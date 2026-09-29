from typing import List, Dict, Any, Tuple
from src.vector_store import VectorStore
from src.config import RETRIEVAL_TOP_K


class PolicyRetriever:
    def __init__(self, vector_store: VectorStore = None, top_k: int = RETRIEVAL_TOP_K):
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieves matching policy chunks for a customer complaint."""
        k = top_k or self.top_k
        return self.vector_store.search(query=query, top_k=k)

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into a clean, numbered context string for prompt injection."""
        if not retrieved_chunks:
            return "No relevant company policy documents found."

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("source", "Unknown Document")
            text = chunk.get("text", "").strip()
            score = chunk.get("score", 0.0)
            context_blocks.append(
                f"[Document {i}: {source} (Relevance Score: {score:.3f})]\n{text}"
            )

        return "\n\n".join(context_blocks)

    def get_context_for_complaint(self, complaint: str, top_k: int = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Convenience method returning both the formatted prompt context and raw chunk metadata."""
        chunks = self.retrieve(complaint, top_k=top_k)
        formatted = self.format_context(chunks)
        return formatted, chunks
