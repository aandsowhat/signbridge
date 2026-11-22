"""Packets start with an unsigned 32-bit big-endian byte count."""
import struct


def read_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("Connection closed during packet read")
        data.extend(chunk)
    return bytes(data)


def receive(sock, limit):
    size, = struct.unpack("!I", read_exact(sock, 4))
    if not 0 < size <= limit:
        raise ValueError(f"Invalid packet size {size} (limit {limit})")
    return read_exact(sock, size)


def send(sock, payload):
    sock.sendall(struct.pack("!I", len(payload)) + payload)
