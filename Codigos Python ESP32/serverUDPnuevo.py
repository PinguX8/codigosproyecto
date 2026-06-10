import socket
import struct
import numpy as np
import cv2
from ultralytics import YOLO

# UDP SERVER

UDP_IP = "0.0.0.0"
UDP_PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# YOLO

model = YOLO("yolov8n.pt")
frames_db = {} # Estructura: { frame_id: { "total": X, "chunks": { idx: data } } }

print("Servidor UDP escuchando...")
print("Esperando imagenes...")

# BUFFER JPEG

buffer = b''
frame = None
while True:

    # RECIBIR DATOS UDP

    data, addr = sock.recvfrom(2000) # Buffer un poco más grande que el paquete
    
    if len(data) < 8:
        continue # Paquete corrupto o basura

    # Desempaquetar el header de 8 bytes (4 enteros de 16 bits sin signo 'H')
    frame_id, total_chunks, chunk_idx, payload_size = struct.unpack("!HHHH", data[:8])
    payload = data[8:8+payload_size]

    # Si es un frame nuevo, inicializamos su espacio
    if frame_id not in frames_db:
        frames_db[frame_id] = {"total": total_chunks, "chunks": {}}

    # Guardamos el fragmento en su posición correspondiente
    frames_db[frame_id]["chunks"][chunk_idx] = payload

    # Verificar si ya tenemos todos los fragmentos de este frame
    if len(frames_db[frame_id]["chunks"]) == total_chunks:
        # Reconstruir la imagen ordenando los fragmentos
        img_bytes = b""
        for i in range(total_chunks):
            img_bytes += frames_db[frame_id]["chunks"][i]

        # Limpiar memoria borrando este frame y los anteriores que hayan quedado viejos
        frames_db = {k: v for k, v in frames_db.items() if k >= frame_id}

        # Convertir bytes a imagen de OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # DECODIFICAR JPEG

    #npdata = np.frombuffer(jpg, dtype=np.uint8)

    #frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)

        if frame is None:
            print("JPEG corrupto")
            continue

    # CORREGIR ORIENTACION

        frame = cv2.rotate(frame, cv2.ROTATE_180)

    # YOLO
   
        results = model(frame, conf=0.10, verbose=False)

        for r in results:
            print("Objetos detectados:", len(r.boxes))
            for box in r.boxes:
              cls = int(box.cls[0])
              label = model.names[cls]
              conf = float(box.conf[0])
            # Ignorar detecciones muy débiles
              if conf < 0.20:
                continue
            if label == "traffic light":
                print(f"SEMAFORO DETECTADO ({conf:.2f})")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"SEMAFORO {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # MOSTRAR VIDEO

        cv2.imshow("UDP ESP32", frame)

        if cv2.waitKey(1) & 0xFF == 27:
             break

cv2.destroyAllWindows()
