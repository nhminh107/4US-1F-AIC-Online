"""Top-level runner for indexing captions into Elasticsearch."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BackEnd.scripts.index_captions_to_elasticsearch import main

if __name__ == "__main__":
    main()
