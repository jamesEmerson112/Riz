"""Talk to the arm host: newline-delimited JSON over a Unix socket, one request, one reply.

    from so101_host import client
    client.call("skill", name="wave")
"""
import json
import os
import socket
import tempfile


def socket_path():
    """$SO101_SOCK, else <temp dir>/so101-arm.sock. macOS caps Unix socket paths at 104 bytes,
    so the socket lives in the temp dir, not in the repo."""
    return os.environ.get("SO101_SOCK") or os.path.join(tempfile.gettempdir(), "so101-arm.sock")


def call(op, *, sock=None, timeout=60.0, **fields):
    path = str(sock or socket_path())
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(path)
        s.sendall((json.dumps({"op": op, **fields}) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf or b'{"ok": false, "error": "no reply"}')
