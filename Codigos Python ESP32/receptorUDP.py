import socket
import time
import numpy as np
import struct
import cv2
# Debe coincidir con:
# udp.beginPacket(IP_RECEPTOR_CAMARA, PUERTO_RECEPTOR_CAMARA);
LOCAL_IP = "192.168.100.14"
# Debe coincidir con
# #define PUERTO_RECEPTOR_CAMARA 12345
LOCAL_PORT = 12345
BUFFER_SIZE = 1500


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

try:
    sock.bind((LOCAL_IP, LOCAL_PORT))
    print(f"Servidor UDP iniciado y escuchando en {LOCAL_IP}:{LOCAL_PORT}")
except Exception as e:
    print(f"Error al asignar el socket: {e}")
    exit()


ultimo_id_imagen = -1
ultima_longitud_imagen = 0
diccionario_imagen = {}


tiempo_anterior = time.time()
contador_frames = 0
while True:
    try:
        bytes_data, address = sock.recvfrom(
            BUFFER_SIZE)
        id_imagen_host = struct.unpack('>H', bytes_data[:2])[0]
        indice_fragmento = struct.unpack('>H', bytes_data[2:4])[0]
        bytes_totales = struct.unpack('>L', bytes_data[4:8])[0]
        total_fragmentos = struct.unpack('>H', bytes_data[8:10])[0]
        fragmento_imagen = bytes_data[10:]

        # print(f"id {id_imagen_host} indice_Fragmento {indice_fragmento} bytes totales {bytes_totales} total_fragmentos {total_fragmentos}")
        if id_imagen_host != ultimo_id_imagen:
            # Ya es al menos la segunda vez que estamos recibiendo
            # imagen.append(fragmento_imagen)
            ultima_longitud_imagen = bytes_totales
            diccionario_imagen = {}
            ultimo_id_imagen = id_imagen_host
            # print("Comenzada la recepción")
        diccionario_imagen[indice_fragmento] = fragmento_imagen

        if len(diccionario_imagen) == total_fragmentos:
            fragmentos_ordenados = [diccionario_imagen[i]
                                    for i in sorted(diccionario_imagen.keys())]
            imagen_final = b"".join(fragmentos_ordenados)
            if len(imagen_final) == ultima_longitud_imagen:
                # print(f"{ultimo_id_imagen} Armado bien mide {len(imagen_final)}")

                imagen_array = np.frombuffer(imagen_final, dtype=np.uint8)
                imagen_cv = cv2.imdecode(imagen_array, cv2.IMREAD_COLOR)

                if imagen_cv is not None:
                    tiempo_actual = time.time()
                    diferencia = tiempo_actual - tiempo_anterior
                    if diferencia > 0:
                        fps = 1.0 / diferencia
                    else:
                        fps = 0
                    tiempo_anterior = tiempo_actual

                    fps_texto = f"FPS: {fps:.2f}"
                    cv2.putText(
                        imagen_cv,
                        fps_texto,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    cv2.imshow("Video en Vivo", imagen_cv)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    """
                nombre_archivo = f"imagen_id_{ultimo_id_imagen}.jpeg"
                try:
                    with open(nombre_archivo, "wb") as f:
                        f.write(imagen_final)
                    print(
                        f"Imagen guardada exitosamente como: {nombre_archivo}")
                except IOError as e:
                    print(f"Error al intentar guardar el archivo: {e}")
                """
            else:
                print(
                    f"Mide {len(imagen_final)} pero debería medir {ultima_longitud_imagen}")
            diccionario_imagen = {}

    except KeyboardInterrupt:
        print("\nCerrando servidor UDP...")
        break
    except Exception as e:
        print(f"Error durante la recepción: {e}")
        break

sock.close()
