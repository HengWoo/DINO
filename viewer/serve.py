#!/usr/bin/env python3
"""Simple HTTP server for the DINO 3D Spatial Viewer."""
import argparse
import http.server
import functools


def main():
    parser = argparse.ArgumentParser(description="DINO Viewer dev server")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=".")
    with http.server.HTTPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"DINO Viewer serving at http://localhost:{args.port}")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
