import socket
import sys

def send_images(server_address, server_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((server_address, server_port))
        for i in range(1, 50001):
            with open(f"/path/to/images/ILSVRC2012_val_000{i:05d}.bin", 'rb') as file:
                image_data = file.read(150528)

                if not image_data:
                    print("All images sent.")
                    sys.exit(0)

                s.sendall(image_data)

                print(f"Sent image of size {len(image_data)} bytes")
                response = s.recv(1024)
                if response != b'ACK':
                    print("Did not receive acknowledgment from server.")
                    break
                print("Received acknowledgment from server.")


if __name__ == "__main__":
    server_address = "127.0.0.1" # NOTE: your IP address
    server_port = 65432          # NOTE: your port number
    send_images(server_address, server_port)

