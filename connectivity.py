import socket

HOST = "192.168.133.20"  # Linux machine IP
PORT = 502

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

message = "Hello from Windows!"
client_socket.sendall(message.encode())

data = client_socket.recv(1024)
print(f"📩 Received from server: {data.decode()}")

client_socket.close()
