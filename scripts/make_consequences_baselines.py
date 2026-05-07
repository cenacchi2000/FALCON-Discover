#!/usr/bin/env python
from __future__ import annotations
from _common import load, OUT_TABLES

def main() -> None:
    a = load("consequences_calibration_table5A.csv")
    b = load("consequences_region_table5B.csv")
    c = load("consequences_baselines_table5C.csv")
    d = load("weighted_calibration_full.csv")
    text = "# Tables 5A–5C — Consequences and baselines\n\n"
    text += "## A. Downstream calibration\n\n" + a.to_markdown(index=False) + "\n\n"
    text += "## B. Region quality\n\n" + b.to_markdown(index=False) + "\n\n"
    text += "## C. Stronger comparator baselines\n\n" + c.to_markdown(index=False) + "\n\n"
    text += "## Full weighted calibration across all seven datasets\n\n" + d.to_markdown(index=False) + "\n"
    (OUT_TABLES / "make_consequences_baselines.md").write_text(text)
    print("wrote make_consequences_baselines.md")

if __name__ == "__main__":
    main()
