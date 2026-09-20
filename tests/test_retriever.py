import json
from pathlib import Path

import joblib
import pandas as pd

import retrieval.retriever as retriever_module
from retrieval.retriever import Retriever


def test_tfidf_fallback_retrieval(tmp_path: Path):
    path = tmp_path / "embeddings.joblib"
    df = pd.DataFrame([
        {"title": "HTML", "number": 1, "start": 0, "end": 5, "text": "HTML structures a webpage."},
        {"title": "CSS", "number": 2, "start": 0, "end": 5, "text": "CSS controls presentation and layout."},
    ])
    joblib.dump(df, path)
    retriever = Retriever(path)
    result = retriever.search("HTML webpage", top_k=1, prefer_semantic=False)
    assert len(result) == 1
    assert result.iloc[0]["title"] == "HTML"
    assert result.iloc[0]["retrieval_method"] == "tfidf"


def test_missing_lfs_pointer_falls_back_to_jsons(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "jsons"
    data_dir.mkdir()
    payload = {
        "chunks": [
            {"title": "HTML Basics", "number": 1, "start": 0, "end": 5, "text": "HTML structures a webpage."},
            {"title": "CSS Basics", "number": 2, "start": 0, "end": 5, "text": "CSS controls presentation and layout."},
        ]
    }
    (data_dir / "course.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(retriever_module, "DATA_DIR", data_dir)

    path = tmp_path / "embeddings.joblib"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n", encoding="utf-8")

    retriever = Retriever(path)
    result = retriever.search("HTML webpage", top_k=1, prefer_semantic=False)

    assert len(result) == 1
    assert result.iloc[0]["title"] == "HTML Basics"
    assert result.iloc[0]["retrieval_method"] == "tfidf"
