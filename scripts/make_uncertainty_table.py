#!/usr/bin/env python
from __future__ import annotations
from _common import load, write_markdown

def main() -> None:
    df = load("uncertainty_intervals_table14.csv")
    write_markdown(df, "make_uncertainty_table.md", "Table 14 — Uncertainty intervals")
    print("wrote", "make_uncertainty_table.md")

if __name__ == "__main__":
    main()
