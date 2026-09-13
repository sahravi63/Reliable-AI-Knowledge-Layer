"""
Chunk the synthetic policy corpus and load into ChromaDB, attaching the
metadata (version, effective_date, scope, department, supersedes) that the
Conflict & Authority Resolution node (src/graph/nodes/conflict.py) depends on.

Run: python -m scripts.ingest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, DATA_DIR, settings

CHUNK_SIZE = 400          # characters per chunk (small corpus -> keep chunks generous)
CHUNK_OVERLAP = 80
COLLECTION_NAME = "policy_docs"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window chunker on paragraphs, falling back to raw char windows."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= chunk_size:
            buf = f"{buf}\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
            if len(para) > chunk_size:
                # paragraph itself too long: hard-window it
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i : i + chunk_size])
                buf = ""
            else:
                buf = para
    if buf:
        chunks.append(buf)
    return chunks


def main():
    with open(DATA_DIR / "documents.json") as f:
        documents = json.load(f)

    print(f"Loading embedding model: {settings.embedding_model}")
    embedder = SentenceTransformer(settings.embedding_model)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Fresh collection each run so re-ingestion is idempotent.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    all_ids, all_docs, all_metas = [], [], []

    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['doc_id']}::chunk-{i}"
            metadata = {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "version": doc.get("version", ""),
                "effective_date": doc.get("effective_date", ""),
                "scope": doc.get("scope", ""),
                "department": doc.get("department") or "",
                "supersedes": doc.get("supersedes", "") or "",
                "superseded_by": doc.get("superseded_by", "") or "",
                "chunk_index": i,
            }
            all_ids.append(chunk_id)
            all_docs.append(chunk)
            all_metas.append(metadata)

    print(f"Embedding {len(all_docs)} chunks from {len(documents)} documents...")
    embeddings = embedder.encode(all_docs, show_progress_bar=True).tolist()

    collection.add(ids=all_ids, documents=all_docs, metadatas=all_metas, embeddings=embeddings)

    print(f"Ingested {len(all_docs)} chunks into ChromaDB at {CHROMA_DIR}")
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    main()
