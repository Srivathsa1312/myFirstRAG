"""
Interactive RAG Console
-----------------------
Allows you to ask questions against the NovaTech knowledge base in real time.
Displays the retrieved chunks, similarity scores, and the generated response.
"""

import os
import sys

# Ensure project directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.rag_pipeline import RAGPipeline


def print_banner():
    print("=" * 65)
    print("         🤖 Welcome to MyFirstRAG Interactive Console")
    print("=" * 65)
    print("Ask any question about NovaTech Dynamics (policies, hardware,")
    print("Project Chimera, incident response, or out-of-domain topics).")
    print("Type 'exit' or 'quit' to stop.\n")


def main():
    print("Initializing RAG pipeline and indexing documents...")
    rag = RAGPipeline()
    count = rag.initialize_knowledge_base()
    print(f"✅ Indexed {count} chunks in ChromaDB.\n")

    print_banner()

    SUGGESTIONS = {
        "1": "What is the remote work policy and stipend?",
        "2": "Can I bring my pet dog to the office?",
        "3": "Who leads Project Chimera and what is the target date?",
        "4": "What command do I run if there is a security leak?",
        "5": "What is the population of Tokyo? (Out-of-domain test)",
    }

    # Pre-canned suggestions
    print("💡 Example questions you can try (type question or number 1-5):")
    for num, text in SUGGESTIONS.items():
        print(f" {num}. {text}")
    print()

    while True:
        try:
            query = input("❓ Your Question (or 1-5): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        # If user entered a number shortcut, map to the actual question
        if query in SUGGESTIONS:
            print(f"👉 Selected #{query}: \"{SUGGESTIONS[query]}\"")
            query = SUGGESTIONS[query]

        print("\n🔎 Searching vector database & generating answer...")
        result = rag.ask(query, top_k=3)

        print("\n" + "-" * 60)
        print(f"📥 RETRIEVED CHUNKS ({len(result['retrieved_chunks'])} matches):")
        print("-" * 60)
        for i, chunk in enumerate(result["retrieved_chunks"], 1):
            print(f"Chunk #{i} [Similarity: {chunk['similarity']:.4f}]:")
            print(f"\"{chunk['text']}\"\n")

        print("-" * 60)
        print(f"🎯 FINAL ANSWER (via {result['source']}):")
        print("-" * 60)
        print(result["answer"])
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
