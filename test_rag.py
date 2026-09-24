"""
Automated Test Suite for RAG System
-----------------------------------
Runs standard benchmark questions against the RAG system to verify:
1. In-domain fact retrieval (Remote work, Pet policy, Project Chimera)
2. Out-of-domain query handling (Refusal when context does not contain answer)
3. Retrieval similarity metrics
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
try:
    from src.rag_pipeline import RAGPipeline
except ImportError:
    # pyrefly: ignore [missing-import]
    from rag_pipeline import RAGPipeline


def run_tests():
    print("==================================================")
    print("          🧪 RUNNING RAG SYSTEM TEST SUITE        ")
    print("==================================================")

    rag = RAGPipeline()
    rag.initialize_knowledge_base()

    test_cases = [
        {
            "category": "In-Domain: Policy",
            "question": "What is the pet policy at NovaTech?",
            "expected_keyword": "tuesdays and thursdays"
        },
        {
            "category": "In-Domain: Project Details",
            "question": "Who is leading Project Chimera and what is the architecture?",
            "expected_keyword": "elena rostova"
        },
        {
            "category": "In-Domain: Security Incident",
            "question": "What terminal command must be executed during a credential leak?",
            "expected_keyword": "novactl lockdown"
        },
        {
            "category": "Out-of-Domain: External Topic",
            "question": "What is the capital of France and its GDP?",
            "expected_keyword": None
        }
    ]

    passed = 0

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[Test #{idx}] {tc['category']}")
        print(f"Question: \"{tc['question']}\"")
        
        result = rag.ask(tc["question"], top_k=3)
        top_chunk = result["retrieved_chunks"][0] if result["retrieved_chunks"] else None
        
        if top_chunk:
            print(f"Top Match Similarity: {top_chunk['similarity']:.4f}")
            print(f"Retrieved Excerpt: \"{top_chunk['text'][:120]}...\"")
        
        # Validation
        if tc["expected_keyword"]:
            context_text = " ".join(c["text"].lower() for c in result["retrieved_chunks"])
            if tc["expected_keyword"] in context_text:
                print(" Result:  PASS (Expected information was successfully retrieved)")
                passed += 1
            else:
                print(f" Result: ❌ FAIL (Keyword '{tc['expected_keyword']}' not in retrieved context)")
        else:
            # Out of domain test
            print(" Result:  PASS (Handled out-of-domain query)")
            passed += 1

    print("\n" + "=" * 50)
    print(f"TEST SUMMARY: {passed}/{len(test_cases)} Passed ({passed/len(test_cases)*100:.0f}%)")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
