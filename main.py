"""
Interactive RAG Console
-----------------------
Allows you to ask questions against the NovaTech knowledge base in real time.
Displays the retrieved chunks, similarity scores, and the generated response.
"""

import os
import sys

# Ensure src folder is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
try:
    from src.rag_pipeline import RAGPipeline
except ImportError:
    # pyrefly: ignore [missing-import]
    from rag_pipeline import RAGPipeline


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

    # Pre-canned suggestions
    print("💡 Example questions you can try:")
    print(" 1. What is the remote work policy and stipend?")
    print(" 2. Can I bring my pet dog to the office?")
    print(" 3. Who leads Project Chimera and what is the target date?")
    print(" 4. What command do I run if there is a security leak?")
    print(" 5. What is the population of Tokyo? (Out-of-domain test)\n")

    while True:
        try:
            query = input("❓ Your Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print("\n🔎 Searching vector database & generating answer...")
        result = rag.ask(query, top_k=2)

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
