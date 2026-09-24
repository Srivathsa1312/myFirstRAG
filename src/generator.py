"""
Generator & Prompt Augmentation Module
---------------------------------------
In RAG, this module performs the "Augmented Generation":
1. AUGMENTATION: Formats the retrieved chunks into a structured prompt context.
2. GENERATION: Passes the enriched prompt to the LLM (Google Gemini or offline fallback).
3. GROUNDING: Constrains the LLM so it answers ONLY using facts from the retrieved chunks,
   eliminating hallucinations.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


def build_augmented_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Constructs the prompt by injecting retrieved context chunks.
    
    This is the core concept of RAG: Transforming a generic question into
    a context-grounded question for the LLM.
    """
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        context_blocks.append(f"[Document Chunk #{idx} - Similarity: {chunk.get('similarity', 'N/A')}]\n{chunk['text']}")

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful, factual company knowledge assistant.
Answer the user's question using ONLY the provided context below.
If the answer cannot be found in the context, say: "I do not have enough information in the provided documents to answer that question."

--- RETRIEVED CONTEXT START ---
{context_str}
--- RETRIEVED CONTEXT END ---

User Question: {query}
Helpful Answer:"""

    return prompt


def generate_response(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates an answer based on retrieved context.
    Uses Google Gemini if GEMINI_API_KEY is available; otherwise provides
    educational offline extraction and shows the complete augmented prompt.
    """
    prompt = build_augmented_prompt(query, retrieved_chunks)
    gemini_key = os.getenv("GEMINI_API_KEY")

    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )
            return {
                "source": "gemini-3.6-flash",
                "answer": response.text.strip(),
                "prompt": prompt
            }
        except Exception as e:
            return {
                "source": "offline-fallback",
                "answer": f"[API Error: {e}]. Retrieved relevant context was:\n" + "\n---\n".join(c['text'] for c in retrieved_chunks),
                "prompt": prompt
            }

    # Offline educational synthesizer when no API key is configured yet
    combined_context = "\n".join(c["text"] for c in retrieved_chunks)
    return {
        "source": "offline-grounded-synthesizer",
        "answer": (
            "💡 [Offline Grounded Mode]\n"
            "Here is the exact information retrieved from the knowledge base for your query:\n\n"
            + "\n\n".join(f"• {c['text']}" for c in retrieved_chunks)
            + "\n\n(Tip: Add GEMINI_API_KEY to your .env file to enable live Gemini LLM generation!)"
        ),
        "prompt": prompt
    }
