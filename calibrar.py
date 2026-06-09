"""
=============================================================
  HERRAMIENTA DE CALIBRACIÓN - AJUSTE DE COLORES HSV
  Usar ANTES de la demo para ajustar los umbrales de
  detección según la iluminación del lugar.
=============================================================

CÓMO USAR:
  python3 calibrar.py

  Mueve los sliders hasta que SOLO se ilumine el color
  del semáforo que quieres detectar en la máscara.
  Anota los valores H_min, H_max, S_min, S_max, V_min, V_max
  y actualiza los np.array en semaforo_main.py
=============================================================
"""

import cv2
import numpy as np

CAMARA_INDEX = 0

def nada(x):
    pass

cap = cv2.VideoCapture(CAMARA_INDEX)
cv2.namedWindow("Calibración HSV")

# Sliders para ajustar el rango HSV
cv2.createTrackbar("H min", "Calibración HSV",   0, 180, nada)
cv2.createTrackbar("H max", "Calibración HSV", 180, 180, nada)
cv2.createTrackbar("S min", "Calibración HSV",  50, 255, nada)
cv2.createTrackbar("S max", "Calibración HSV", 255, 255, nada)
cv2.createTrackbar("V min", "Calibración HSV",  50, 255, nada)
cv2.createTrackbar("V max", "Calibración HSV", 255, 255, nada)

print("Mueve los sliders para calibrar. Presiona Q para salir.")
print("Los valores aparecerán en consola mientras ajustas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("H min", "Calibración HSV")
    h_max = cv2.getTrackbarPos("H max", "Calibración HSV")
    s_min = cv2.getTrackbarPos("S min", "Calibración HSV")
    s_max = cv2.getTrackbarPos("S max", "Calibración HSV")
    v_min = cv2.getTrackbarPos("V min", "Calibración HSV")
    v_max = cv2.getTrackbarPos("V max", "Calibración HSV")

    mascara = cv2.inRange(hsv,
                          np.array([h_min, s_min, v_min]),
                          np.array([h_max, s_max, v_max]))

    pixeles = cv2.countNonZero(mascara)
    print(f"\r H:[{h_min}-{h_max}] S:[{s_min}-{s_max}] V:[{v_min}-{v_max}] | Píxeles: {pixeles}   ", end="")

    resultado = cv2.bitwise_and(frame, frame, mask=mascara)

    cv2.imshow("Calibración HSV", resultado)
    cv2.imshow("Original",        frame)
    cv2.imshow("Máscara",         mascara)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\n\nValores finales:")
print(f"  np.array([{h_min}, {s_min}, {v_min}]),  # mínimo")
print(f"  np.array([{h_max}, {s_max}, {v_max}])   # máximo")

cap.release()
cv2.destroyAllWindows()
