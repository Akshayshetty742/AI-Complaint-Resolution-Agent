from pathlib import Path
from typing import List, Dict, Any
from src.config import KNOWLEDGE_BASE_DIR


def load_documents(directory: Path = KNOWLEDGE_BASE_DIR) -> List[Dict[str, Any]]:
    """Loads all text files from the knowledge base directory."""
    documents = []
    if not directory.exists():
        return documents

    for file_path in sorted(directory.glob("*.txt")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    documents.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "content": content
                    })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return documents


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Chunks documents into semantic blocks by paragraphs/sections or character windows.
    Returns list of chunks with source metadata.
    """
    chunks = []
    for doc in documents:
        content = doc["content"]
        filename = doc["filename"]

        # First try splitting by double newline (logical sections/numbered paragraphs)
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            if not current_chunk:
                current_chunk = para
            elif len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += "\n\n" + para
            else:
                chunks.append({
                    "text": current_chunk,
                    "source": filename,
                    "chunk_id": chunk_index
                })
                chunk_index += 1
                # Add overlap if paragraph is long, or start new
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                current_chunk = (overlap_text + "\n" + para).strip() if overlap_text else para

        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "source": filename,
                "chunk_id": chunk_index
            })

    return chunks
