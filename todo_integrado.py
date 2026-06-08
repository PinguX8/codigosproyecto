from inference_sdk import InferenceHTTPClient
import supervision as sv
import socket
import struct
import numpy as np
import cv2
import threading

# --- CLIENTE ROBOFLOW ---
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="CkPSpW1CLqS1f1bvEJEU",
)

# --- SERVIDOR UDP ---
UDP_IP = "0.0.0.0"
UDP_PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

# --- VARIABLES COMPARTIDAS ---
ultimo_frame = None
ultimo_resultado = None
lock = threading.Lock()
frames_db = {}

# --- HILO DE INFERENCIA (background) ---
def hilo_inferencia():
    global ultimo_frame, ultimo_resultado
    while True:
        with lock:
            frame = ultimo_frame.copy() if ultimo_frame is not None else None

        if frame is not None:
            try:
                result = client.infer(frame, model_id="lentes-para-sofi/8")
                with lock:
                    ultimo_resultado = result
            except Exception as e:
                print(f"Error inferencia: {e}")

thread = threading.Thread(target=hilo_inferencia, daemon=True)
thread.start()

# --- ANOTADORES ---
annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

print("✅ Servidor UDP escuchando en puerto", UDP_PORT)
print("   Presiona ESC para salir.")

# --- LOOP PRINCIPAL ---
while True:
    # Recibir paquete UDP de la ESP32-CAM
    data, addr = sock.recvfrom(65536)
    
    if len(data) < 8:
        continue
    
    # Desempaquetar header
    frame_id, total_chunks, chunk_idx, payload_size = struct.unpack("!HHHH", data[:8])
    payload = data[8:8 + payload_size]

    # Guardar fragmento
    if frame_id not in frames_db:
        frames_db[frame_id] = {"total": total_chunks, "chunks": {}}
    frames_db[frame_id]["chunks"][chunk_idx] = payload

    # Verificar si el frame está completo
    print(f"Frame {frame_id}: chunk {chunk_idx+1}/{total_chunks}")
    if len(frames_db[frame_id]["chunks"]) == total_chunks:
        print(f"✅ Frame {frame_id} completo!")
        # Reconstruir imagen
        img_bytes = b""
        for i in range(total_chunks):
            img_bytes += frames_db[frame_id]["chunks"][i]

        # Limpiar frames viejos
        frames_db = {k: v for k, v in frames_db.items() if k >= frame_id}

        # Decodificar JPEG
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("JPEG corrupto, descartando frame")
            continue

        # Corregir orientación
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Actualizar frame para el hilo de inferencia
        with lock:
            ultimo_frame = frame.copy()
            resultado = ultimo_resultado

        # Anotar con el último resultado disponible
        annotated = frame.copy()
        if resultado:
            detections = sv.Detections.from_inference(resultado)
            labels = [
                f"{pred['class']} {pred['confidence']:.0%}"
                for pred in resultado.get("predictions", [])
            ]
            annotated = annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )

        # Mostrar
        cv2.imshow("ESP32-CAM + Roboflow", annotated)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

sock.close()
cv2.destroyAllWindows()
