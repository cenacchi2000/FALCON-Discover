#!/usr/bin/env python
from __future__ import annotations
from _common import load, write_markdown

def main() -> None:
    df = load("family_ablation_table3.csv")
    write_markdown(df, "make_family_ablation.md", "Table 3 — Discrepancy-family ablation")
    print("wrote", "make_family_ablation.md")

if __name__ == "__main__":
    main()
