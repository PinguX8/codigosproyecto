"""
=============================================================
  LENTES ASISTIVOS - DETECCIÓN DE SEMÁFORO CON YOLO LOCAL
  Versión local — sin internet, detección en tiempo real
  Hardware: Raspberry Pi + Webcam USB + Buzzer en GPIO 18
=============================================================
INSTRUCCIONES:
INSTALAR:
  pip install ultralytics opencv-python numpy
  (En Raspi agregar: RPi.GPIO)

ESTRUCTURA DE CARPETAS:
  lentes-para-sofi/
  ├── semaforo_local.py   ← este archivo
  └── best.pt             ← tu modelo entrenado

CORRER:
  python3 semaforo_local.py
=============================================================
"""

import cv2
import numpy as np
import subprocess
import threading
import time
from ultralytics import YOLO

# ══ CONFIGURACIÓN ════════════════════════════════════════
MODELO_PATH   = "best.pt"   # debe estar en la misma carpeta
CAMARA_INDEX  = 0
COOLDOWN_SEG  = 3
CONFIANZA_MIN = 0.5

# ⚠️ Revisar en Roboflow → Dataset → Classes
# y poner exactamente como aparecen ahí (en minúsculas)
CLASE_ROJO  = "red"
CLASE_VERDE = "green"
# ═════════════════════════════════════════════════════════

# Cargar modelo local
print("Cargando modelo YOLO...")
modelo = YOLO(MODELO_PATH)
print("Modelo cargado OK ✓")

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
            time.sleep(0.05)
    threading.Thread(target=_run, daemon=True).start()


# ═══════════════════════════════════════════════════════════
#  INTERPRETAR PREDICCIÓN YOLO
# ═══════════════════════════════════════════════════════════

COLORES_BGR = {"ROJO": (0, 0, 255), "VERDE": (0, 255, 0)}

def procesar_resultado(results, frame):
    """
    Interpreta los resultados de YOLO, dibuja los cuadros
    y retorna el color detectado con mayor confianza.
    """
    mejor_clase = None
    mejor_conf  = 0.0

    for result in results:
        for box in result.boxes:
            confianza = float(box.conf[0])
            clase_idx = int(box.cls[0])
            clase     = result.names[clase_idx].lower()

            if confianza < CONFIANZA_MIN:
                continue

            # Determinar color del estado
            if clase in [CLASE_ROJO.lower(), "red", "rojo", "stop", "pare"]:
                estado = "ROJO"
            elif clase in [CLASE_VERDE.lower(), "green", "verde", "go", "cruce"]:
                estado = "VERDE"
            else:
                estado = None

            # Dibujar caja
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color_caja = COLORES_BGR.get(estado, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_caja, 2)
            cv2.putText(frame, f"{clase} {confianza:.0%}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_caja, 2)

            # Guardar el de mayor confianza
            if estado and confianza > mejor_conf:
                mejor_conf  = confianza
                mejor_clase = estado

    # Etiqueta general en pantalla
    label = mejor_clase if mejor_clase else "Buscando semaforo..."
    color_label = COLORES_BGR.get(mejor_clase, (180, 180, 180))
    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color_label, 3)

    return mejor_clase, frame


# ═══════════════════════════════════════════════════════════
#  BUCLE PRINCIPAL
# ═══════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(CAMARA_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    print("Sistema iniciado. Presiona Q para salir.")

    ultimo_color  = None
    ultimo_tiempo = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inferencia local — directo en cada frame, sin internet
        results = modelo(frame, verbose=False)
        color, frame = procesar_resultado(results, frame)

        ahora = time.time()
        if color:
            if (color != ultimo_color) or (ahora - ultimo_tiempo >= COOLDOWN_SEG):
                print(f"[SEMÁFORO] {color}")
                alerta_audio(color)
                vibrar(color)
                ultimo_color  = color
                ultimo_tiempo = ahora

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
