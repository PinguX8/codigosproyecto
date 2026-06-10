"""
=============================================================
  LENTES ASISTIVOS - DETECCIÓN DE SEMÁFORO CON ROBOFLOW
  Versión: Hosted API
  Hardware: Raspberry Pi + Webcam USB + Buzzer en GPIO 18
=============================================================

INSTALAR:
  pip install inference-sdk opencv-python numpy RPi.GPIO

CONFIGURAR:
  1. Entrar a roboflow.com → tu proyecto → Deploy
  2. Copiar: API_KEY, MODEL_ID (ej: "semaforos/3")

CORRER:
  python3 semaforo_roboflow.py
=============================================================
"""

import cv2
import numpy as np
import subprocess
import threading
import time

# ── Roboflow ──────────────────────────────────────────────
from inference_sdk import InferenceHTTPClient

# ══ CONFIGURA ESTOS 3 VALORES ════════════════════════════
API_KEY   = "CkPSpW1CLqS1f1bvEJEU"       # Roboflow → Settings → API Key
MODEL_ID  = "lentes-para-sofi/9"     # Ej: "semaforos-peatonales/3"
# ═════════════════════════════════════════════════════════

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# ── GPIO ──────────────────────────────────────────────────
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
CAMARA_INDEX    = 0
COOLDOWN_SEG    = 3
CONFIANZA_MIN   = 0.5     # 0.0 a 1.0 — ignorar detecciones bajo este %
MOSTRAR_VENTANA = True

# Clases que puede devolver tu modelo (ajustar según Roboflow)
# Entrar a Roboflow → Dataset → Classes para ver los nombres exactos
CLASE_ROJO  = "red"     # o "rojo", "stop", según tu modelo
CLASE_VERDE = "green"   # o "verde", "go", según tu modelo


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
    "ROJO":  (1, 0.8),    # 1 pulso largo
    "VERDE": (3, 0.15),   # 3 pulsos cortos
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
            time.sleep(0.12)

    threading.Thread(target=_run, daemon=True).start()


# ═══════════════════════════════════════════════════════════
#  INTERPRETAR RESULTADO DE ROBOFLOW
# ═══════════════════════════════════════════════════════════

def interpretar_prediccion(resultado):
    """
    Devuelve 'ROJO', 'VERDE' o None según las predicciones del modelo.
    Toma la detección con mayor confianza que supere CONFIANZA_MIN.
    """
    predicciones = resultado.get("predictions", [])

    mejor_clase      = None
    mejor_confianza  = 0.0

    for pred in predicciones:
        clase      = pred.get("class", "").lower()
        confianza  = pred.get("confidence", 0)

        if confianza < CONFIANZA_MIN:
            continue

        if confianza > mejor_confianza:
            mejor_confianza = confianza
            mejor_clase     = clase

    if mejor_clase is None:
        return None

    # Mapear clase del modelo → ROJO / VERDE
    if mejor_clase in [CLASE_ROJO.lower(), "red", "rojo", "stop", "pare"]:
        return "ROJO"
    elif mejor_clase in [CLASE_VERDE.lower(), "green", "verde", "go", "cruce"]:
        return "VERDE"

    return None


# ═══════════════════════════════════════════════════════════
#  DIBUJAR DETECCIONES EN EL FRAME
# ═══════════════════════════════════════════════════════════

COLORES_BGR = {"ROJO": (0, 0, 255), "VERDE": (0, 255, 0)}

def dibujar_detecciones(frame, resultado, color_estado):
    for pred in resultado.get("predictions", []):
        x = int(pred["x"] - pred["width"]  / 2)
        y = int(pred["y"] - pred["height"] / 2)
        w = int(pred["width"])
        h = int(pred["height"])
        clase     = pred["class"]
        confianza = pred["confidence"]

        color_caja = COLORES_BGR.get(color_estado, (200, 200, 200))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_caja, 2)
        cv2.putText(frame, f"{clase} {confianza:.0%}",
                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_caja, 2)

    # Estado general en pantalla
    label = color_estado if color_estado else "Buscando semaforo..."
    color_label = COLORES_BGR.get(color_estado, (180, 180, 180))
    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color_label, 3)

    return frame


# ═══════════════════════════════════════════════════════════
#  BUCLE PRINCIPAL
# ═══════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(CAMARA_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    print("Sistema iniciado. Presiona Q para salir.")

    ultimo_color  = None
    ultimo_tiempo = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Enviar frame a Roboflow ──────────────────────
        try:
            resultado = CLIENT.infer(frame, model_id=MODEL_ID)
        except Exception as e:
            print(f"[ERROR Roboflow] {e}")
            resultado = {"predictions": []}

        # ── Interpretar resultado ────────────────────────
        color = interpretar_prediccion(resultado)
        ahora = time.time()

        if color:
            if (color != ultimo_color) or (ahora - ultimo_tiempo >= COOLDOWN_SEG):
                print(f"[SEMÁFORO] {color}")
                alerta_audio(color)
                vibrar(color)
                ultimo_color  = color
                ultimo_tiempo = ahora

        # ── Mostrar ventana ──────────────────────────────
        if MOSTRAR_VENTANA:
            frame = dibujar_detecciones(frame, resultado, color)
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
