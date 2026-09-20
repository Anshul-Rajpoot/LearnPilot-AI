from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from config import DATA_DIR, EMBEDDINGS_PATH
from llm.ollama import OllamaError, embed


def main() -> None:
    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No transcript JSON files found in {DATA_DIR}")

    rows: list[dict] = []
    for path in files:
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        chunks = payload.get("chunks", [])
        print(f"Embedding {path.name}: {len(chunks)} chunks")
        embeddings = embed([str(chunk.get("text", "")) for chunk in chunks])
        for chunk, vector in zip(chunks, embeddings):
            item = dict(chunk)
            item["chunk_id"] = len(rows)
            item["embedding"] = vector
            rows.append(item)

    df = pd.DataFrame.from_records(rows)
    joblib.dump(df, EMBEDDINGS_PATH)
    print(f"Saved {len(df)} chunks to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except OllamaError as exc:
        raise SystemExit(f"Embedding failed: {exc}") from exc
