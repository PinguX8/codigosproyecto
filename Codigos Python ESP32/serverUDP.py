import socket
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
print("antes del while")


    # =========================
    # RECIBIR IMAGEN
    # =========================

    #data, addr = sock.recvfrom(65535)
    #print(len(data))
    #print(data[:2])
    #print(data[-2:])

    # =========================
    # CONVERTIR JPEG
    # =========================

    #npdata = np.frombuffer(data, dtype=np.uint8)

    #frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)

buffer = b''

while True:

    data, addr = sock.recvfrom(65535)

    buffer += data
    if len(buffer) > 10000:
     print("Buffer demasiado grande, reiniciando")
     buffer = b''
     continue



    print("Buffer:", len(buffer)) #chatg cambios
    print("Inicio:", buffer.find(b'\xff\xd8')) #chatg cambios
    print("Fin:", buffer.find(b'\xff\xd9')) #chatg cambios
    

    #inicio = buffer.find(b'\xff\xd8')
    #fin = buffer.find(b'\xff\xd9')
    #fin = buffer.rfind(b'\xff\xd9') #chatg cambios
    #print("Buffer:", len(buffer))  #chatg cambios

    inicio = buffer.find(b'\xff\xd8')

    if inicio > 0:
     buffer = buffer[inicio:]
     inicio = 0

    fin = buffer.find(b'\xff\xd9', inicio)

    if inicio == -1:
     buffer = b''
     continue

    if fin != -1 and fin < inicio:
     buffer = buffer[inicio:]
     continue



    if len(buffer) > 4000:
     print("Ultimos bytes:", buffer[-10:])

    if inicio != -1 and fin != -1 and fin > inicio:

        jpg = buffer[inicio:fin+2]

        buffer = buffer[fin+2:]

        siguiente = buffer.find(b'\xff\xd8')

        if siguiente > 0:
         buffer = buffer[siguiente:]

        npdata = np.frombuffer(jpg, dtype=np.uint8)
        print("JPEG reconstruido:", len(jpg))

        frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)

        if frame is None:
            print("JPEG corrupto")
            continue

        print("IMAGEN OK")

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # =========================
        # YOLO
        # =========================

        results = model(frame, conf=0.10, verbose=False)

        for r in results:
            print("Objetos detectados:", len(r.boxes))
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                print(model.names[cls], conf)


                if label == "traffic light":

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0,255,0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "SEMAFORO",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2
                    )

        cv2.imshow("UDP ESP32", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break



cv2.destroyAllWindows()
