import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 12345))
sock.settimeout(5.0)

print("Escuchando en puerto 12345...")

while True:
    try:
        data, addr = sock.recvfrom(2000)
        print(f"✅ Datos recibidos de {addr}, tamaño: {len(data)} bytes")
    except socket.timeout:
        print("⏱️ Sin datos...")
