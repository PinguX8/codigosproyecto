from ultralytics import YOLO
import cv2
import time

# CARGAR MODELO
model = YOLO("yolov8n.pt")

# URL ESP32-CAM

url = "http://192.168.1.69:81/stream"

cap = cv2.VideoCapture(url)

# Buffer pequeño
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# CONTROL FPS

fps = 2
frame_delay = 1 / fps

# CONTADOR FRAMES
contador = 0

# LOOP PRINCIPAL

while True:

    start_time = time.time()

    ret, frame = cap.read()

    if not ret:
        print("Error leyendo frame")
        continue

    contador += 1

    # SOLO PROCESAR ALGUNOS FRAMES
    if contador % 5 != 0:

        cv2.imshow("ESP32-CAM", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        continue


    # REDUCIR TAMAÑO IMAGEN

    frame_small = cv2.resize(frame, (320, 240))

    # EJECUTAR YOLO

    results = model(
        frame_small,
        verbose=False,
        imgsz=320,
        conf=0.35
    )

    # DETECCIONES

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            # SOLO SEMAFOROS
            if label == "traffic light":

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # REESCALAR coordenadas
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 240

                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                # RECTANGULO VERDE
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                # TEXTO
                cv2.putText(
                    frame,
                    "SEMAFORO",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                print("SEMAFORO DETECTADO")

    # MOSTRAR VIDEO

    cv2.imshow("ESP32-CAM + YOLO", frame)

    # ESC para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # LIMITAR FPS

    elapsed = time.time() - start_time

    sleep_time = frame_delay - elapsed

    if sleep_time > 0:
        time.sleep(sleep_time)


# CERRAR TODO
cap.release()
cv2.destroyAllWindows()
