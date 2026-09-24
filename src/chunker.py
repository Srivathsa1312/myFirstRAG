"""
Chunking Module
----------------
In RAG, chunking breaks down documents into smaller, semantically coherent passages.

Key Concepts:
1. chunk_size: Target length (in characters or tokens) for each passage.
2. chunk_overlap: Overlapping text retained across consecutive chunks to preserve
   context across boundaries.
3. Clean boundaries: Splitting on spaces/newlines instead of cutting mid-word.
"""

from typing import List, Dict, Any


def chunk_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80
) -> List[Dict[str, Any]]:
    """
    Splits input text into overlapping chunks, attempting to split on word/newline boundaries.
    
    Args:
        text (str): The raw text to chunk.
        chunk_size (int): Target maximum characters per chunk.
        chunk_overlap (int): Overlap characters between chunks.
        
    Returns:
        List[Dict[str, Any]]: Chunks with unique ID, text content, and metadata.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)
    chunk_id = 0

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunk_content = text[start:].strip()
            if chunk_content:
                chunks.append({
                    "id": f"chunk_{chunk_id:03d}",
                    "text": chunk_content,
                    "char_count": len(chunk_content)
                })
            break

        # Look for the last natural break (newline or space) within the chunk window
        natural_break = text.rfind("\n", start + chunk_size // 2, end)
        if natural_break == -1:
            natural_break = text.rfind(" ", start + chunk_size // 2, end)

        if natural_break != -1 and natural_break > start:
            end = natural_break

        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append({
                "id": f"chunk_{chunk_id:03d}",
                "text": chunk_content,
                "char_count": len(chunk_content)
            })
            chunk_id += 1

        # Advance start position by (length - overlap)
        step = max(1, (end - start) - chunk_overlap)
        start += step

    return chunks


if __name__ == "__main__":
    sample_text = (
        "NovaTech Dynamics offers a 90-day remote work program called Work-From-Anywhere-Quarter. "
        "Employees get a $350 stipend for internet and co-working spaces.\n\n"
        "Project Chimera is led by Dr. Elena Rostova and will launch in November 2026. "
        "It uses lattice-based cryptography and the Aegis-7 KEM mechanism."
    )
    test_chunks = chunk_text(sample_text, chunk_size=120, chunk_overlap=30)
    print(f"Generated {len(test_chunks)} clean chunks:")
    for c in test_chunks:
        print(f"[{c['id']}] ({c['char_count']} chars): {c['text']!r}")
