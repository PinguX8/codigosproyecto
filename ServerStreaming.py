import asyncio
import threading
import queue
import time
import cv2
import numpy as np
from aiohttp import web, WSMsgType
from ultralytics import YOLO

HOST = "0.0.0.0"
PORT = 8765
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.4
TARGET_CLASS = "traffic light"

# ── Queue de tamaño 1: solo el frame más reciente para YOLO ───────────────────
# No es una queue FIFO normal — es un "slot" que se sobreescribe siempre
latest_frame_lock = threading.Lock()
latest_frame_for_yolo = {"frame": None, "id": 0}

# Resultado YOLO compartido (se actualiza cada vez que YOLO termina)
yolo_result_lock = threading.Lock()
yolo_result = {"detections": [], "frame_id": -1}

# Frame más reciente para mostrar (se actualiza con cada frame recibido)
display_lock = threading.Lock()
display_frame = {"frame": None}

LIGHT_COLORS = {
    "red":     (0, 0, 220),
    "yellow":  (0, 215, 255),
    "green":   (0, 200, 60),
    "unknown": (180, 180, 180),
}


def detect_light_color(frame, x1, y1, x2, y2) -> str:
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"
    top_roi = roi[:max(roi.shape[0] // 3, 1)]
    hsv = cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV)
    masks = {
        "red":    cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
                | cv2.inRange(hsv, (160, 120, 120), (180, 255, 255)),
        "yellow": cv2.inRange(hsv, (20, 100, 100), (35, 255, 255)),
        "green":  cv2.inRange(hsv, (40, 80, 80),   (90, 255, 255)),
    }
    counts = {k: cv2.countNonZero(v) for k, v in masks.items()}
    dominant = max(counts, key=counts.get)
    return dominant if counts[dominant] > 50 else "unknown"


# ── Thread YOLO: siempre procesa el frame más reciente ───────────────────────
def yolo_worker():
    model = YOLO(MODEL_PATH)
    print(f"[YOLO] Modelo cargado: {MODEL_PATH}")
    last_processed_id = -1

    while True:
        # Esperar a que haya un frame nuevo
        with latest_frame_lock:
            frame = latest_frame_for_yolo["frame"]
            frame_id = latest_frame_for_yolo["id"]

        if frame is None or frame_id == last_processed_id:
            time.sleep(0.01)
            continue

        last_processed_id = frame_id
        frame_copy = frame.copy()

        results = model(frame_copy, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = []

        for box in results.boxes:
            if model.names[int(box.cls)] != TARGET_CLASS:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color_name = detect_light_color(frame_copy, x1, y1, x2, y2)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": round(float(box.conf), 3),
                "color": color_name,
            })

        with yolo_result_lock:
            yolo_result["detections"] = detections
            yolo_result["frame_id"] = frame_id

        if detections:
            print(f"[YOLO] frame#{frame_id}: {detections}")


# ── Thread de visualización: muestra frames a máxima velocidad ───────────────
def display_worker():
    cv2.namedWindow("ESP32-CAM | YOLO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ESP32-CAM | YOLO", 640, 480)
    fps_time = time.time()
    fps_count = 0
    fps_display = 0.0

    while True:
        # Obtener frame más reciente
        with display_lock:
            frame = display_frame["frame"]

        if frame is None:
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue

        display = frame.copy()

        # Superponer último resultado YOLO disponible (aunque sea viejo)
        with yolo_result_lock:
            detections = yolo_result["detections"][:]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color_name = det["color"]
            conf = det["conf"]
            bgr = LIGHT_COLORS[color_name]

            # Escalar bbox si el frame fue redimensionado
            cv2.rectangle(display, (x1, y1), (x2, y2), bgr, 2)
            label = f"{color_name.upper()} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 6, y1), bgr, -1)
            cv2.putText(display, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS real de la ventana
        fps_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps_display = fps_count / (now - fps_time)
            fps_count = 0
            fps_time = now

        hud = f"Display: {fps_display:.1f} fps  |  Semaforos: {len(detections)}"
        cv2.putText(display, hud, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, hud, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

        # Upscale QVGA → 640x480
        display = cv2.resize(display, (640, 480), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("ESP32-CAM | YOLO", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# ── Handler WebSocket ─────────────────────────────────────────────────────────
async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=5 * 1024 * 1024, heartbeat=None)
    try:
        await ws.prepare(request)
    except Exception as e:
        print(f"[WS] Handshake fallido: {e}")
        return ws

    client = request.remote
    print(f"[WS] Conectado: {client}")
    await ws.send_str("ready")

    frame_id = 0
    frames_rx = 0
    t_start = time.time()

    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            data = msg.data
            if len(data) < 100:
                continue

            np_arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_id += 1
            frames_rx += 1

            # Actualizar frame para display (siempre, sin bloqueo largo)
            with display_lock:
                display_frame["frame"] = frame

            # Actualizar slot para YOLO (sobreescribe el anterior si no fue procesado)
            with latest_frame_lock:
                latest_frame_for_yolo["frame"] = frame
                latest_frame_for_yolo["id"] = frame_id

            # Stats cada 60 frames
            if frames_rx % 60 == 0:
                elapsed = time.time() - t_start
                print(f"[WS] {frames_rx} frames | {frames_rx/elapsed:.1f} fps entrada")

            # Responder con detección actual
            with yolo_result_lock:
                dets = yolo_result["detections"]
            if dets:
                colors = ",".join(d["color"] for d in dets)
                try:
                    await ws.send_str(f"{len(dets)}:{colors}")
                except Exception:
                    pass

        elif msg.type == WSMsgType.ERROR:
            print(f"[WS] Error: {ws.exception()}")
            break

    print(f"[WS] Desconectado: {client} ({frames_rx} frames)")
    return ws


async def main():
    app = web.Application()
    app.router.add_get("/", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, HOST, PORT).start()
    print(f"[WS] Servidor en ws://{HOST}:{PORT}/")
    await asyncio.Event().wait()


if __name__ == "__main__":
    threading.Thread(target=yolo_worker, daemon=True).start()
    threading.Thread(target=display_worker, daemon=True).start()
    asyncio.run(main())