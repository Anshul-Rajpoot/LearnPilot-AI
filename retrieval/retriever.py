from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DATA_DIR
from llm.ollama import OllamaError, embed, is_available


class Retriever:
    """Hybrid course retriever: Ollama embeddings first, TF-IDF fallback."""

    def __init__(self, embeddings_path: Path):
        self.embeddings_path = embeddings_path
        self.df = self._load()
        self._tfidf: TfidfVectorizer | None = None
        self._tfidf_matrix = None

    def _load_from_jsons(self) -> pd.DataFrame:
        json_files = sorted(DATA_DIR.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No transcript JSON files found in {DATA_DIR}")

        rows: list[dict[str, Any]] = []
        for path in json_files:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            for chunk in payload.get("chunks", []):
                if not isinstance(chunk, dict):
                    continue
                item = dict(chunk)
                item.setdefault("text", "")
                rows.append(item)

        if not rows:
            raise ValueError(f"No course chunks found in {DATA_DIR}")

        df = pd.DataFrame.from_records(rows)
        if "text" not in df.columns:
            raise ValueError("Knowledge base must contain a 'text' column")
        return df

    def _load(self) -> pd.DataFrame:
        if not self.embeddings_path.exists():
            return self._load_from_jsons()

        # Git LFS pointer means the large binary was not included in a source archive.
        if self.embeddings_path.stat().st_size < 1024:
            head = self.embeddings_path.read_text(encoding="utf-8", errors="ignore")[:80]
            if head.startswith("version https://git-lfs.github.com/spec/v1"):
                return self._load_from_jsons()

        try:
            df = joblib.load(self.embeddings_path)
        except Exception:
            return self._load_from_jsons()

        if not isinstance(df, pd.DataFrame):
            raise TypeError("embeddings.joblib must contain a pandas DataFrame")
        if "text" not in df.columns:
            raise ValueError("Knowledge base must contain a 'text' column")
        if "embedding" in df.columns:
            df = df.copy()
            df["embedding"] = df["embedding"].apply(np.asarray)
        return df

    @property
    def video_count(self) -> int:
        return int(self.df["number"].nunique()) if "number" in self.df.columns else 0

    @property
    def chunk_count(self) -> int:
        return len(self.df)

    def _tfidf_search(self, query: str, top_k: int) -> pd.DataFrame:
        texts = tuple(self.df["text"].fillna("").astype(str).tolist())
        if self._tfidf is None:
            self._tfidf = TfidfVectorizer()
            self._tfidf_matrix = self._tfidf.fit_transform(texts)
        query_vector = self._tfidf.transform([query])
        scores = cosine_similarity(self._tfidf_matrix, query_vector).ravel()
        indices = scores.argsort()[::-1][:top_k]
        result = self.df.iloc[indices].copy()
        result["score"] = scores[indices]
        result["retrieval_method"] = "tfidf"
        return result

    def search(self, query: str, top_k: int = 5, prefer_semantic: bool = True) -> pd.DataFrame:
        if not query or not query.strip():
            return self.df.head(0).copy()
        top_k = max(1, min(top_k, len(self.df)))

        if prefer_semantic and "embedding" in self.df.columns and is_available():
            try:
                query_embedding = embed([query])[0]
                matrix = np.vstack(self.df["embedding"].to_numpy())
                scores = cosine_similarity(matrix, [query_embedding]).ravel()
                indices = scores.argsort()[::-1][:top_k]
                result = self.df.iloc[indices].copy()
                result["score"] = scores[indices]
                result["retrieval_method"] = "semantic"
                return result
            except (OllamaError, ValueError):
                pass

        return self._tfidf_search(query, top_k)

    def records(self, result: pd.DataFrame) -> list[dict[str, Any]]:
        columns = [c for c in ["title", "number", "start", "end", "text", "score", "retrieval_method"] if c in result.columns]
        return result[columns].to_dict(orient="records")
