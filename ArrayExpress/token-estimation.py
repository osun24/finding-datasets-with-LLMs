#!/usr/bin/env python3
"""Estimate tiktoken token counts for ArrayExpress context_text files."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
import tiktoken


DEFAULT_MODEL = "gpt-4o-mini"
CONTEXT_GLOB = "*-screened/context_text/*.txt"


def get_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            return tiktoken.get_encoding("o200k_base")
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")


def context_files(root: Path) -> list[Path]:
    return sorted(path for path in root.glob(CONTEXT_GLOB) if path.is_file())


def estimate_tokens(path: Path, encoding) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    source_list = path.parent.parent.name
    accession = path.stem
    return {
        "source_list": source_list,
        "accession": accession,
        "path": str(path),
        "characters": len(text),
        "tokens": len(encoding.encode(text)),
    }


def print_summary(rows: list[dict[str, object]]) -> None:
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source_list"]].append(int(row["tokens"]))

    print("Token estimates by context_text directory:")
    for source_list in sorted(by_source):
        counts = by_source[source_list]
        print(
            f"  {source_list}: "
            f"files={len(counts)}, total={sum(counts):,}, "
            f"avg={sum(counts) / len(counts):,.1f}, max={max(counts):,}"
        )

    all_counts = [int(row["tokens"]) for row in rows]
    if all_counts:
        print(
            f"\nOverall: files={len(all_counts)}, total={sum(all_counts):,}, "
            f"avg={sum(all_counts) / len(all_counts):,.1f}, max={max(all_counts):,}"
        )


def main() -> int:
    if tiktoken is None:
        print(
            "Missing dependency: tiktoken. Install it with `python3 -m pip install tiktoken` "
            "or install the repo requirements.",
            file=sys.stderr,
        )
        return 1

    root = Path(__file__).resolve().parent
    encoding = get_encoding(DEFAULT_MODEL)
    files = context_files(root)

    if not files:
        print(f"No files matched {root / CONTEXT_GLOB}")
        return 1

    rows = [estimate_tokens(path, encoding) for path in files]

    print(f"Token estimates using {DEFAULT_MODEL} tokenizer:\n")
    for row in rows:
        print(f"{row['source_list']}/{row['accession']}: {int(row['tokens']):,} tokens")

    print()
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
