import socket
import os
import struct

def host(ip, port, debug=False):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_address = (ip, port)
    server_socket.bind(server_address)

    if debug: print(f'Server listening on {server_address[0]}:{server_address[1]}')
    server_socket.listen(1)

    client_socket, cliend_address = server_socket.accept()
    if debug: print('Connection from', cliend_address)

    return client_socket

def connect(ip, port, debug=False):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = (ip, port)
    client_socket.connect(server_address)
    if debug: print('Connected to {}:{}'.format(*server_address))

    return client_socket

def send_file(socket, file_name):
    # Send file size
    file_size = int(os.path.getsize(file_name))
    file_size = struct.pack('I', file_size)
    socket.sendall(file_size)

    # Send File data
    with open(file_name, 'rb') as file:
        data = file.read()
        socket.sendall(data)

def receive_file(socket, file_name):
    # Receive file size
    file_size = socket.recv(4)
    file_size = struct.unpack('I', file_size)[0]

    # Receive file data
    error = False
    with open(file_name, 'wb') as file:
        while file_size:
            data = socket.recv(min(1024, file_size))

            if not data:
                error = True
                print('Error while receiving file.')
                break

            file.write(data)
            file_size -= len(data)
    if error:
        os.remove(file_name)