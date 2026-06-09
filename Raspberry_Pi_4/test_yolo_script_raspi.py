import cv2
from ultralytics import YOLO

# 1. Cargar el modelo YOLOv8 Nano pre-entrenado
model = YOLO("yolov8n.pt")

# 2. Ejecutar la inferencia sobre una imagen
# (Puedes cambiar la URL por la ruta de una foto local en tu Raspberry)
url_imagen = "https://ultralytics.com/images/bus.jpg"
results = model(url_imagen)

# 3. Mostrar los resultados en la terminal
for result in results:
    boxes = result.boxes  # Cajas de los objetos detectados
    for box in boxes:
        # Obtener el nombre de la clase (persona, bus, etc.) y la confianza
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        print(f"Detectado: {label} con un {conf*100:.2f}% de certeza.")
