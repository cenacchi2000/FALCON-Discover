#!/usr/bin/env python
from __future__ import annotations
from _common import load, write_markdown

def main() -> None:
    df = load("null_concentration_table15.csv")
    write_markdown(df, "make_null_concentration.md", "Table 15 — Random-review null comparison")
    print("wrote", "make_null_concentration.md")

if __name__ == "__main__":
    main()
