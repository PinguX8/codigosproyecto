from inference_sdk import InferenceHTTPClient
import supervision as sv
import cv2
import threading

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="CkPSpW1CLqS1f1bvEJEU",
)

# --- Variables compartidas entre hilos ---
ultimo_frame = None
ultimo_resultado = None
lock = threading.Lock()

def hilo_inferencia():
    """Corre en segundo plano, solo hace inferencia"""
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

# Iniciar hilo de inferencia en background
thread = threading.Thread(target=hilo_inferencia, daemon=True)
thread.start()

# --- Captura de video ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

print("✅ Corriendo. Presiona Q para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Actualizar frame para el hilo de inferencia
    with lock:
        ultimo_frame = frame.copy()
        resultado = ultimo_resultado

    # Mostrar detecciones del último resultado disponible
    annotated = frame.copy()
    if resultado:
        detections = sv.Detections.from_inference(resultado)
        labels = [
            f"{pred['class']} {pred['confidence']:.0%}"
            for pred in resultado.get("predictions", [])
        ]
        annotated = annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    cv2.imshow("Lentes para Sofi", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
