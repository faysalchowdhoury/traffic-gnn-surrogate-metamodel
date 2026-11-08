"""``gnnsurrogate generate | train | serve``."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gnnsurrogate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--config", default="configs/corridor.yaml")
    g.add_argument("--out", default="data/scenarios.jsonl")
    g.add_argument("-n", type=int, default=1500)
    g.add_argument("--workers", type=int, default=2)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/scenarios.jsonl")
    t.add_argument("--out", default="results/surrogate")
    t.add_argument("--registry", default="registry")
    s = sub.add_parser("serve")
    s.add_argument("--registry", default="registry")
    s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        from .data import generate

        print(generate(a.config, a.out, a.n, workers=a.workers))
    elif a.cmd == "train":
        from .train import run, write_report

        print(write_report(run(a.data, a.out, a.registry), Path(a.out)).read_text(encoding="utf-8"))
    else:
        import uvicorn

        from .api.app import create_app

        uvicorn.run(create_app(a.registry), host="127.0.0.1", port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
