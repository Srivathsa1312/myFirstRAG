"""
Vector Store & Ingestion Module
--------------------------------
In RAG, the Vector Store acts as the "long-term memory".

How it works:
1. Takes the text chunks produced by the chunker.
2. Uses an embedding model to convert each chunk into a vector.
   (Automatically uses Google Gemini if GEMINI_API_KEY is set, or our fast offline BM25/TF-IDF vectorizer).
3. Indexes the vectors in ChromaDB so we can perform fast semantic similarity searches (Cosine distance).
"""

import os
import sys
from typing import List, Dict, Any, Optional
import chromadb

# Ensure local imports work cleanly
sys.path.append(os.path.dirname(__file__))
from embeddings import get_embedding_function, BM25TFIDFEmbeddingFunction

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "novatech_knowledge"

# Cached embedding function instance
_EMBEDDING_FUNCTION = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Returns a persistent ChromaDB client."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_active_embedding_function(corpus_docs: Optional[List[str]] = None):
    global _EMBEDDING_FUNCTION
    if _EMBEDDING_FUNCTION is None or corpus_docs is not None:
        _EMBEDDING_FUNCTION = get_embedding_function(corpus_docs)
    return _EMBEDDING_FUNCTION


def get_or_create_collection(client: Optional[chromadb.PersistentClient] = None, corpus_docs: Optional[List[str]] = None):
    """
    Returns the target ChromaDB collection configured with our embedding function.
    """
    if client is None:
        client = get_chroma_client()
    ef = get_active_embedding_function(corpus_docs)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )


def index_chunks(chunks: List[Dict[str, Any]], reset: bool = True) -> int:
    """
    Indexes a list of text chunks into ChromaDB.
    
    Args:
        chunks (List[Dict[str, Any]]): Chunks containing 'id' and 'text'.
        reset (bool): If True, clears existing collection items before indexing.
        
    Returns:
        int: Number of chunks indexed.
    """
    client = get_chroma_client()
    
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    corpus_docs = [c["text"] for c in chunks]
    collection = get_or_create_collection(client=client, corpus_docs=corpus_docs)

    ids = [c["id"] for c in chunks]
    documents = corpus_docs
    metadatas = [{"chunk_id": c["id"], "char_count": len(c["text"])} for c in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    return collection.count()


def query_collection(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Performs semantic vector search for a query string.
    
    Args:
        query_text (str): The search query.
        top_k (int): Number of most similar chunks to return.
        
    Returns:
        List[Dict[str, Any]]: Retrieved chunks with their text, metadata, and similarity score.
    """
    collection = get_or_create_collection()
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    if results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, distances):
            retrieved.append({
                "text": doc,
                "metadata": meta,
                "distance": round(float(dist), 4),
                "similarity": round(1.0 - float(dist), 4)
            })

    return retrieved


if __name__ == "__main__":
    from chunker import chunk_text

    print("--- [Step 1] Loading Knowledge Base ---")
    kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_base.txt")
    with open(kb_path, "r", encoding="utf-8") as f:
        content = f.read()

    print("--- [Step 2] Chunking Knowledge Base ---")
    chunks = chunk_text(content, chunk_size=350, chunk_overlap=60)
    print(f"Generated {len(chunks)} chunks.")

    print("--- [Step 3] Indexing Chunks in ChromaDB ---")
    count = index_chunks(chunks, reset=True)
    print(f"Successfully stored {count} chunks in ChromaDB!")

    print("\n--- [Step 4] Testing Semantic Retrieval Queries ---")
    test_queries = [
        "Can I bring my pet dog to the office?",
        "What are the details of Project Chimera?",
        "What do I do if there is a security leak?"
    ]

    for q in test_queries:
        print(f"\n🔍 Query: '{q}'")
        hits = query_collection(q, top_k=1)
        if hits:
            best = hits[0]
            print(f"👉 Top Match (Similarity Score: {best['similarity']}):")
            print(f"   {best['text'][:180]}...")
