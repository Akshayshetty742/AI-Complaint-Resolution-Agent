import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.config import FAISS_INDEX_DIR, EMBEDDING_MODEL
from src.document_loader import load_documents, chunk_documents


class VectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL, index_dir: Path = FAISS_INDEX_DIR):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_file = self.index_dir / "index.faiss"
        self.metadata_file = self.index_dir / "chunks.json"
        self.embedder = SentenceTransformer(self.model_name)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []

    def build_index(self, force_rebuild: bool = False) -> int:
        """Loads policy documents, chunks them, creates embeddings, and saves FAISS index."""
        if not force_rebuild and self.is_index_available():
            self.load_index()
            return len(self.chunks)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        docs = load_documents()
        if not docs:
            raise ValueError("No policy documents found in knowledge base!")

        self.chunks = chunk_documents(docs)
        texts = [chunk["text"] for chunk in self.chunks]

        # Generate normalized embeddings for cosine similarity
        embeddings = self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product of normalized vectors = Cosine similarity
        self.index.add(embeddings)

        # Save to disk
        faiss.write_index(self.index, str(self.index_file))
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

        return len(self.chunks)

    def is_index_available(self) -> bool:
        return self.index_file.exists() and self.metadata_file.exists()

    def load_index(self):
        """Loads FAISS index and chunk metadata from disk."""
        if not self.is_index_available():
            raise FileNotFoundError(f"FAISS index files not found in {self.index_dir}")

        self.index = faiss.read_index(str(self.index_file))
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top_k most similar chunks for a query."""
        if self.index is None or not self.chunks:
            if self.is_index_available():
                self.load_index()
            else:
                self.build_index()

        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        k = min(top_k, len(self.chunks))
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx != -1 and idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(score)
                chunk["rank"] = rank + 1
                results.append(chunk)

        return results
