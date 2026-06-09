"""
=============================================================
  LENTES ASISTIVOS - DETECCIÓN DE SEMÁFORO PEATONAL
  Raspberry Pi + Webcam USB
  Detecta: ROJO y VERDE
  Alertas: Audio (espeak) + Vibración (GPIO 18)
=============================================================

HARDWARE NECESARIO:
  - Raspberry Pi (cualquier modelo con GPIO)
  - Webcam USB
  - Motor vibrador o buzzer conectado a GPIO 18 y GND

INSTALAR DEPENDENCIAS ANTES DE CORRER:
  sudo apt update
  sudo apt install espeak -y
  pip install opencv-python numpy RPi.GPIO

CORRER CON:
  python3 semaforo_main.py
=============================================================
"""

import cv2
import numpy as np
import subprocess
import time
import threading

# ── Intentar importar GPIO (solo funciona en Raspberry Pi) ──
try:
    import RPi.GPIO as GPIO
    GPIO_DISPONIBLE = True
except ImportError:
    print("[AVISO] RPi.GPIO no disponible. Vibracion desactivada (modo prueba en PC).")
    GPIO_DISPONIBLE = False


# ═══════════════════════════════════════════════════════════
#  CONFIGURACIÓN — ajusta estos valores si es necesario
# ═══════════════════════════════════════════════════════════

CAMARA_INDEX     = 0      # 0 = primera webcam USB conectada
GPIO_VIBRADOR    = 18     # Pin GPIO para el motor vibrador
UMBRAL_PIXELES   = 500    # Mínimo de píxeles del color para confirmar detección
COOLDOWN_SEG     = 3      # Segundos mínimos entre alertas repetidas
MOSTRAR_VENTANA  = True   # False si corres sin pantalla (modo headless)


# ═══════════════════════════════════════════════════════════
#  CONFIGURACIÓN GPIO
# ═══════════════════════════════════════════════════════════

if GPIO_DISPONIBLE:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_VIBRADOR, GPIO.OUT)
    pwm = GPIO.PWM(GPIO_VIBRADOR, 100)
    pwm.start(0)


# ═══════════════════════════════════════════════════════════
#  DETECCIÓN DE COLOR
# ═══════════════════════════════════════════════════════════

def detectar_semaforo(frame):
    """
    Recibe un frame de la cámara y retorna 'ROJO', 'VERDE' o None.
    Usa espacio de color HSV para mayor robustez ante cambios de luz.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rojo tiene dos rangos en HSV (cruza los 0°/180°)
    mascara_rojo1 = cv2.inRange(hsv, np.array([0,   120, 70]),  np.array([10,  255, 255]))
    mascara_rojo2 = cv2.inRange(hsv, np.array([170, 120, 70]),  np.array([180, 255, 255]))
    mascara_rojo  = mascara_rojo1 + mascara_rojo2

    # Verde (semáforo peatonal suele ser verde brillante)
    mascara_verde = cv2.inRange(hsv, np.array([40, 70, 70]), np.array([90, 255, 255]))

    pixeles_rojo  = cv2.countNonZero(mascara_rojo)
    pixeles_verde = cv2.countNonZero(mascara_verde)

    # Gana el que tenga más píxeles (siempre que supere el umbral)
    if pixeles_rojo > UMBRAL_PIXELES and pixeles_rojo > pixeles_verde:
        return "ROJO"
    elif pixeles_verde > UMBRAL_PIXELES and pixeles_verde > pixeles_rojo:
        return "VERDE"

    return None


# ═══════════════════════════════════════════════════════════
#  ALERTAS DE AUDIO
# ═══════════════════════════════════════════════════════════

MENSAJES_AUDIO = {
    "ROJO":  "Semáforo en rojo. Detenerse.",
    "VERDE": "Semáforo en verde. Puede cruzar.",
}

def alerta_audio(color):
    """Reproduce el mensaje de voz en español (no bloquea el loop principal)."""
    mensaje = MENSAJES_AUDIO.get(color)
    if mensaje:
        # -s 130 = velocidad más lenta para mayor claridad
        hilo = threading.Thread(
            target=lambda: subprocess.run(["espeak", "-v", "es", "-s", "130", mensaje]),
            daemon=True
        )
        hilo.start()


# ═══════════════════════════════════════════════════════════
#  ALERTAS DE VIBRACIÓN
# ═══════════════════════════════════════════════════════════

# Patrón: (cantidad de pulsos, duración de cada pulso en segundos)
PATRONES_VIBRACION = {
    "ROJO":  (1, 0.8),   # 1 pulso largo  → PELIGRO, no cruces
    "VERDE": (3, 0.15),  # 3 pulsos cortos → LIBRE, puedes cruzar
}

def vibrar(color):
    """Activa el motor vibrador en un hilo separado para no bloquear."""
    if not GPIO_DISPONIBLE:
        print(f"[VIBRACIÓN SIMULADA] Patrón para: {color}")
        return

    patron = PATRONES_VIBRACION.get(color)
    if not patron:
        return

    def _vibrar():
        pulsos, duracion = patron
        for _ in range(pulsos):
            pwm.ChangeDutyCycle(50)   # encender vibrador
            time.sleep(duracion)
            pwm.ChangeDutyCycle(0)    # apagar vibrador
            time.sleep(0.12)          # pausa entre pulsos

    hilo = threading.Thread(target=_vibrar, daemon=True)
    hilo.start()


# ═══════════════════════════════════════════════════════════
#  BUCLE PRINCIPAL
# ═══════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(CAMARA_INDEX)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara. Verifica el índice CAMARA_INDEX.")
        return

    print("=" * 50)
    print("  Sistema de detección de semáforo iniciado")
    print("  Presiona Q para salir")
    print("=" * 50)

    ultimo_color   = None
    ultimo_tiempo  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer frame de la cámara.")
            break

        color_detectado = detectar_semaforo(frame)
        ahora = time.time()

        # Lanzar alerta si:
        #   a) Se detectó un color nuevo, O
        #   b) Pasó el tiempo de cooldown y el mismo color sigue activo
        if color_detectado:
            if (color_detectado != ultimo_color) or (ahora - ultimo_tiempo >= COOLDOWN_SEG):
                print(f"[SEMÁFORO] {color_detectado}")
                alerta_audio(color_detectado)
                vibrar(color_detectado)
                ultimo_color  = color_detectado
                ultimo_tiempo = ahora

        # ── Mostrar ventana de debug (opcional) ──
        if MOSTRAR_VENTANA:
            color_texto = {
                "ROJO":  (0, 0, 255),
                "VERDE": (0, 255, 0),
                None:    (200, 200, 200),
            }
            label = color_detectado if color_detectado else "Sin detección"
            cv2.putText(frame, label, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        color_texto[color_detectado], 3)
            cv2.imshow("Semaforo - Deteccion", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Saliendo...")
                break
        else:
            # Sin ventana: pequeña pausa para no saturar la CPU
            time.sleep(0.05)

    # ── Limpieza ──
    cap.release()
    if MOSTRAR_VENTANA:
        cv2.destroyAllWindows()
    if GPIO_DISPONIBLE:
        pwm.stop()
        GPIO.cleanup()
    print("Sistema detenido correctamente.")


if __name__ == "__main__":
    main()
