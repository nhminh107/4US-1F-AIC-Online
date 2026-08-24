from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BackEnd.app.retrieval_v2.corpus_stats import CorpusDocument, CorpusStats


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic Retrieval V2 corpus statistics from JSONL.",
    )
    parser.add_argument("input", type=Path, help="Input JSONL document file")
    parser.add_argument("output", type=Path, help="Output versioned JSON stats file")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--video-field", default="video_id")
    parser.add_argument("--text-field", default="text")
    return parser


def _documents(args: argparse.Namespace) -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    with args.input.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                documents.append(
                    CorpusDocument(
                        document_id=str(payload[args.id_field]),
                        video_id=str(payload[args.video_field]),
                        text=str(payload[args.text_field]),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"Missing field {exc.args[0]!r} on JSONL line {line_number}"
                ) from exc
    return documents


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stats = CorpusStats.from_documents(_documents(args))
    stats.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
