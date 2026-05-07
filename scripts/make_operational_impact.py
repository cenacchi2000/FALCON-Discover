#!/usr/bin/env python
from __future__ import annotations
from _common import load, write_markdown

def main() -> None:
    df = load("operational_impact_table2.csv")
    write_markdown(df, "make_operational_impact.md", "Table 2 — Operational impact and threshold robustness")
    print("wrote", "make_operational_impact.md")

if __name__ == "__main__":
    main()
