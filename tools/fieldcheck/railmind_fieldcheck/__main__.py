"""CLI: `python -m railmind_fieldcheck [--host H] [--port P] [--no-open]`."""
from __future__ import annotations

import argparse

from .server import serve


def main(argv=None):
    ap = argparse.ArgumentParser(description="RailMind Field Input Check (standalone)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)

    httpd = serve(args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"RailMind Field Input Check (standalone) — serving at {url}")
    print("Pick your data source and click 'Run check'. Ctrl-C to stop.")
    if not args.no_open:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
