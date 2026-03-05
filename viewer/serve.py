#!/usr/bin/env python3
"""Simple HTTP server for the DINO 3D Spatial Viewer.

Adds gzip Content-Encoding for large files (.ply) and proper MIME types.
Intended for local development only — do not expose to a network.
"""
import argparse
import gzip
import http.server
import functools
import os

# Files above this threshold get gzip-compressed on the fly
GZIP_THRESHOLD = 10 * 1024 * 1024  # 10 MB

EXTRA_MIME_TYPES = {
    ".ply": "application/octet-stream",
    ".splat": "application/octet-stream",
    ".spz": "application/octet-stream",
}


class GzipHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that serves large files with gzip Content-Encoding."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().do_GET()

        size = os.path.getsize(path)
        accept_gzip = "gzip" in self.headers.get("Accept-Encoding", "")

        if size > GZIP_THRESHOLD and accept_gzip:
            self._serve_gzipped(path)
        else:
            super().do_GET()

    def _serve_gzipped(self, path):
        content_type = self.guess_type(path)

        try:
            with open(path, "rb") as f:
                raw = f.read()
        except FileNotFoundError:
            self.send_error(404, "File not found")
            return
        except PermissionError:
            self.send_error(403, "Permission denied")
            return
        except OSError as e:
            self.log_error("Error reading %s: %s", path, e)
            self.send_error(500, "Internal server error")
            return

        try:
            compressed = gzip.compress(raw, compresslevel=6)
        except (MemoryError, OSError) as e:
            self.log_error("Gzip compression failed for %s: %s", path, e)
            self.send_error(500, "Compression failed")
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(compressed)))
        self.end_headers()
        try:
            self.wfile.write(compressed)
        except (BrokenPipeError, ConnectionResetError):
            self.log_error("Client disconnected during transfer of %s", path)

    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in EXTRA_MIME_TYPES:
            return EXTRA_MIME_TYPES[ext]
        return super().guess_type(path)


def main():
    parser = argparse.ArgumentParser(description="DINO Viewer dev server")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    handler = functools.partial(GzipHandler, directory=".")
    with http.server.HTTPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"DINO Viewer serving at http://localhost:{args.port}")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
