"""
=============================================================
  LENTES ASISTIVOS - DETECCIÓN DE SEMÁFORO CON ROBOFLOW
  Versión 2 — Hilo de inferencia en paralelo (fluida)
  Hardware: Raspberry Pi + Webcam USB + Buzzer en GPIO 18
=============================================================
INSTALAR:
  pip install inference-sdk opencv-python numpy
  (En Raspi agregar: RPi.GPIO)

CORRER:
  python3 semaforo_v2.py
=============================================================
"""

import cv2
import numpy as np
import subprocess
import threading
import time
from inference_sdk import InferenceHTTPClient

# ══ TUS DATOS DE ROBOFLOW ════════════════════════════════
API_KEY  = "CkPSpW1CLqS1f1bvEJEU"
MODEL_ID = "lentes-para-sofi/9"
# ═════════════════════════════════════════════════════════

CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)

# ── GPIO (solo Raspi) ─────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT)
    pwm = GPIO.PWM(18, 100)
    pwm.start(0)
    GPIO_OK = True
except ImportError:
    print("[AVISO] GPIO no disponible - modo prueba en PC")
    GPIO_OK = False

# ── Configuración ─────────────────────────────────────────
CAMARA_INDEX  = 0
COOLDOWN_SEG  = 3
CONFIANZA_MIN = 0.5

# ⚠️  Revisar en Roboflow → Dataset → Classes
# y poner exactamente como aparecen ahí
CLASE_ROJO  = "red"
CLASE_VERDE = "green"

# ── Estado compartido entre hilos ─────────────────────────
ultimo_resultado   = {"predictions": []}
frame_para_inferir = None
lock = threading.Lock()


# ═══════════════════════════════════════════════════════════
#  HILO DE INFERENCIA — corre en paralelo, no bloquea cámara
# ═══════════════════════════════════════════════════════════

def hilo_inferencia():
    global ultimo_resultado, frame_para_inferir
    while True:
        with lock:
            frame = frame_para_inferir.copy() if frame_para_inferir is not None else None

        if frame is not None:
            try:
                resultado = CLIENT.infer(frame, model_id=MODEL_ID)
                with lock:
                    ultimo_resultado = resultado
            except Exception as e:
                print(f"[ERROR Roboflow] {e}")

        time.sleep(0.1)  # ~10 inferencias por segundo


# ═══════════════════════════════════════════════════════════
#  ALERTAS AUDIO
# ═══════════════════════════════════════════════════════════

MENSAJES = {
    "ROJO":  "Semáforo en rojo. Detenerse.",
    "VERDE": "Semáforo en verde. Puede cruzar.",
}

def alerta_audio(color):
    msg = MENSAJES.get(color)
    if msg:
        threading.Thread(
            target=lambda: subprocess.run(["espeak", "-v", "es", "-s", "130", msg]),
            daemon=True
        ).start()


# ═══════════════════════════════════════════════════════════
#  ALERTAS VIBRACIÓN / BUZZER
# ═══════════════════════════════════════════════════════════

PATRONES = {
    "ROJO":  (1, 0.8),   # 1 pulso largo  → no cruces
    "VERDE": (3, 0.15),  # 3 pulsos cortos → puedes cruzar
}

def vibrar(color):
    patron = PATRONES.get(color)
    if not patron:
        return
    def _run():
        pulsos, dur = patron
        for _ in range(pulsos):
            if GPIO_OK:
                pwm.ChangeDutyCycle(50)
                time.sleep(dur)
                pwm.ChangeDutyCycle(0)
            else:
                print(f"  [BUZZ] {'█' * int(dur * 10)}")
                time.sleep(dur)
            time.sleep(0.05) #esto lo cambie de 0.1 a 0.05 😁
    threading.Thread(target=_run, daemon=True).start()


# ═══════════════════════════════════════════════════════════
#  INTERPRETAR PREDICCIÓN
# ═══════════════════════════════════════════════════════════

def interpretar_prediccion(resultado):
    mejor_clase, mejor_conf = None, 0.0
    for pred in resultado.get("predictions", []):
        clase     = pred.get("class", "").lower()
        confianza = pred.get("confidence", 0)
        if confianza >= CONFIANZA_MIN and confianza > mejor_conf:
            mejor_conf  = confianza
            mejor_clase = clase

    if mejor_clase is None:
        return None
    if mejor_clase in [CLASE_ROJO.lower(), "red", "rojo", "stop", "pare"]:
        return "ROJO"
    if mejor_clase in [CLASE_VERDE.lower(), "green", "verde", "go", "cruce"]:
        return "VERDE"
    return None


# ═══════════════════════════════════════════════════════════
#  DIBUJAR DETECCIONES
# ═══════════════════════════════════════════════════════════

COLORES_BGR = {"ROJO": (0, 0, 255), "VERDE": (0, 255, 0)}

def dibujar(frame, resultado, color_estado):
    for pred in resultado.get("predictions", []):
        x = int(pred["x"] - pred["width"]  / 2)
        y = int(pred["y"] - pred["height"] / 2)
        w, h = int(pred["width"]), int(pred["height"])
        c = COLORES_BGR.get(color_estado, (200, 200, 200))
        cv2.rectangle(frame, (x, y), (x+w, y+h), c, 2)
        cv2.putText(frame, f"{pred['class']} {pred['confidence']:.0%}",
                    (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    label = color_estado if color_estado else "Buscando semaforo..."
    c     = COLORES_BGR.get(color_estado, (180, 180, 180))
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, c, 3)
    return frame


# ═══════════════════════════════════════════════════════════
#  BUCLE PRINCIPAL
# ═══════════════════════════════════════════════════════════

def main():
    global frame_para_inferir

    cap = cv2.VideoCapture(CAMARA_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Iniciar hilo de inferencia en paralelo
    threading.Thread(target=hilo_inferencia, daemon=True).start()

    print("Sistema iniciado. Presiona Q para salir.")

    ultimo_color  = None
    ultimo_tiempo = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pasar frame al hilo de inferencia
        with lock:
            frame_para_inferir = frame.copy()
            resultado = ultimo_resultado.copy()

        color = interpretar_prediccion(resultado)
        ahora = time.time()

        if color:
            if (color != ultimo_color) or (ahora - ultimo_tiempo >= COOLDOWN_SEG):
                print(f"[SEMÁFORO] {color}")
                alerta_audio(color)
                vibrar(color)
                ultimo_color  = color
                ultimo_tiempo = ahora

        frame = dibujar(frame, resultado, color)
        cv2.imshow("Semaforo Asistivo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if GPIO_OK:
        pwm.stop()
        GPIO.cleanup()
    print("Sistema detenido.")


if __name__ == "__main__":
    main()
