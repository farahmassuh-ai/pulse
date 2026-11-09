"""Serve the live training dashboard on your own computer.

Run from the repo root:
    python src/serve_dashboard.py
then open http://localhost:8765/dashboard/ in a browser and start
    python src/train_live.py configs/final.yaml
in a second terminal. The page refreshes itself every second.

It serves only dashboard/ and live_runs/, to this computer only (127.0.0.1),
and tells the browser never to cache, so the dashboard always reads the newest snapshot.
"""
from __future__ import annotations

import argparse
import functools
import posixpath
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent


ALLOWED = ("dashboard", "live_runs")         # nothing else in the repo (no .env, no data) is served


def allowed(raw_path: str) -> bool:
    """Judge the path the way the file server will read it: decoded (%2e%2e is ..) and
    cleaned up (a/../b is b). Only files inside dashboard/ or live_runs/ pass."""
    path = unquote(raw_path.split("?", 1)[0].split("#", 1)[0])
    if "\\" in path:                                # no Windows-style separators
        return False
    clean = posixpath.normpath(path).lstrip("/")
    return clean.split("/", 1)[0] in ALLOWED


class NoCacheHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if not allowed(self.path):
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):                              # HEAD must follow the same rule as GET
        if not allowed(self.path):
            self.send_error(404)
            return
        super().do_HEAD()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):          # keep the terminal quiet (one poll per second)
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Serve the live training dashboard.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    handler = functools.partial(NoCacheHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"dashboard: http://localhost:{args.port}/dashboard/   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
