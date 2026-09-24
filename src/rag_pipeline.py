"""
End-to-End RAG Pipeline
-----------------------
Combines:
1. Retrieval (Semantic search via ChromaDB)
2. Augmentation (Prompt assembly with context injection)
3. Generation (LLM answering strictly using the retrieved context)
"""

import os
import sys
from typing import Dict, Any

sys.path.append(os.path.dirname(__file__))
from vector_store import query_collection, index_chunks
from chunker import chunk_text
from generator import generate_response, build_augmented_prompt


class RAGPipeline:
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_base.txt")
        self.data_path = data_path

    def initialize_knowledge_base(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """Loads and indexes the knowledge base into ChromaDB."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Knowledge base not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        indexed_count = index_chunks(chunks, reset=True)
        return indexed_count

    def ask(self, question: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline for a given question.
        
        Step 1: Retrieve top_k semantically similar chunks.
        Step 2: Augment prompt with retrieved chunks.
        Step 3: Generate grounded answer.
        """
        # Step 1: Retrieval
        retrieved_chunks = query_collection(question, top_k=top_k)

        # Step 2 & 3: Augmentation & Generation
        result = generate_response(question, retrieved_chunks)

        return {
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "answer": result["answer"],
            "source": result["source"],
            "augmented_prompt": result["prompt"]
        }


if __name__ == "__main__":
    rag = RAGPipeline()
    rag.initialize_knowledge_base()

    sample_question = "What hardware do new employees receive?"
    print(f"\n================ Question: {sample_question} ================\n")
    response = rag.ask(sample_question, top_k=2)

    print("--- [Retrieved Context Chunks] ---")
    for i, chunk in enumerate(response["retrieved_chunks"], 1):
        print(f"\nChunk #{i} (Similarity: {chunk['similarity']}):")
        print(chunk["text"])

    print("\n--- [Final Generated Answer] ---")
    print(response["answer"])
