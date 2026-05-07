#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "figures"
DST = ROOT / "results" / "figures"
DST.mkdir(parents=True, exist_ok=True)

def main() -> None:
    copied = 0
    for png in SRC.glob("*.png"):
        shutil.copy2(png, DST / png.name)
        copied += 1
    print(f"copied {copied} figure(s) to {DST}")

if __name__ == "__main__":
    main()
