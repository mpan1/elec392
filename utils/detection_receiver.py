"""
UDP receiver for Coral detections.
"""

import json
import socket
import time

class DetectionReceiver:
    def __init__(self, port=5005, timeout=0.2, stale_after=0.5):
        """
        timeout: socket timeout (seconds)
        stale_after: how old data can be before considered invalid
        """
        self.addr = ("127.0.0.1", port)
        self.timeout = timeout
        self.stale_after = stale_after

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.addr)
        self.sock.settimeout(self.timeout)

        self.latest = None

    def update(self):
        """Attempt to receive new data (non-blocking-ish)."""
        try:
            data, _ = self.sock.recvfrom(65535)
            self.latest = json.loads(data.decode("utf-8"))
        except socket.timeout:
            pass

    def get_latest(self):
        """Return latest detections or None if stale/missing."""
        if self.latest is None:
            return None

        age = time.time() - self.latest["timestamp"]
        if age > self.stale_after:
            return None

        return self.latest