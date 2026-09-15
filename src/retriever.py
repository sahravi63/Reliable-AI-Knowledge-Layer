from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.config import CHROMA_DIR, settings

COLLECTION_NAME = "policy_docs"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    distance: float = 0.0
    relevant: Optional[bool] = None       # set by the relevance grader
    grader_reasoning: str = ""


class Retriever:
    """Thin wrapper around the ChromaDB collection built by scripts/ingest.py."""

    def __init__(self, top_k: int = None):
        self.top_k = top_k or settings.top_k
        self._client = None
        self._collection = None
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(settings.embedding_model)
        return self._embedder

    @property
    def collection(self):
        if self._collection is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self._collection = self._client.get_collection(COLLECTION_NAME)
        return self._collection

    def search(self, query: str, top_k: Optional[int] = None) -> list[Chunk]:
        k = top_k or self.top_k
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)

        chunks: list[Chunk] = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=meta, distance=dist))
        return chunks
