"""Command-line retrieval smoke test for the LearnPilot AI knowledge base."""

from config import EMBEDDINGS_PATH
from retrieval.retriever import Retriever


def main() -> None:
    query = input("Ask a course question: ").strip()
    retriever = Retriever(EMBEDDINGS_PATH)
    results = retriever.search(query, top_k=5)
    for _, row in results.iterrows():
        print(f"\nVideo {row.get('number','?')} — {row.get('title','Untitled')}")
        print(f"{row.get('start','?')}s–{row.get('end','?')}s")
        print(row.get("text", ""))


if __name__ == "__main__":
    main()
