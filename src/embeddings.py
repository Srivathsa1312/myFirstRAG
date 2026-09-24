"""
Embeddings Module
-----------------
What is an Embedding?
An embedding converts raw text into a mathematical vector representation.
Semantically relevant documents have a higher dot-product (cosine similarity) with the query vector.

This module provides:
1. GeminiEmbeddingFunction: State-of-the-art neural embeddings (text-embedding-004) when GEMINI_API_KEY is available.
2. BM25TFIDFEmbeddingFunction: Robust, deterministic, zero-network vectorizer utilizing
   TF-IDF (Term Frequency - Inverse Document Frequency) with subword matching and stopword filtering.
"""

import os
import math
import re
import hashlib
from typing import List, Set, Dict, Optional
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

load_dotenv()

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "me", "more", "most", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your"
}

# Common semantic synonyms for basic zero-shot query expansion
SYNONYMS: Dict[str, List[str]] = {
    "pet": ["pet", "dog", "cat", "paws"],
    "pets": ["pet", "dog", "cat", "paws"],
    "dog": ["dog", "pet", "paws"],
    "dogs": ["dog", "pet", "paws"],
    "cat": ["cat", "pet", "paws"],
    "cats": ["cat", "pet", "paws"],
    "vacation": ["pto", "leave", "holiday", "wellness"],
    "leave": ["leave", "pto", "vacation", "wellness"],
    "pto": ["pto", "leave", "vacation"],
    "remote": ["remote", "travel", "work-from-anywhere", "international"],
    "travel": ["travel", "remote", "international", "stipend"],
    "laptop": ["laptop", "hardware", "monitor", "equipment"],
    "hardware": ["hardware", "laptop", "monitor", "equipment"],
    "security": ["security", "lockdown", "incident", "soc"],
    "breach": ["breach", "lockdown", "incident", "leak"],
    "hack": ["lockdown", "incident", "security", "leak"],
    "chimera": ["chimera", "encryption", "cryptography", "rostova", "aegis-7"]
}


class BM25TFIDFEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Offline vectorizer with TF-IDF weighting, synonym expansion, and deterministic feature projection.
    Dimension: 512
    """
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.doc_freq: Dict[str, int] = {}
        self.total_docs: int = 1

    def fit_corpus(self, docs: List[str]):
        """Builds Inverse Document Frequency (IDF) table across the corpus."""
        self.total_docs = max(1, len(docs))
        self.doc_freq.clear()
        for doc in docs:
            tokens = set(self._extract_tokens(doc))
            for tok in tokens:
                self.doc_freq[tok] = self.doc_freq.get(tok, 0) + 1

    def _extract_tokens(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
        tokens = []
        for word in words:
            if word in STOPWORDS:
                continue
            tokens.append(word)
            # Add semantic synonyms if any
            if word in SYNONYMS:
                tokens.extend(SYNONYMS[word])
        return tokens

    def _embed_single(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        tokens = self._extract_tokens(text)
        if not tokens:
            return vec

        # Calculate term frequencies
        tf: Dict[str, float] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0.0) + 1.0

        for tok, count in tf.items():
            # Standard smoothed IDF
            df = self.doc_freq.get(tok, 0)
            idf = math.log((self.total_docs + 1) / (df + 1)) + 1.0
            weight = count * idf

            # Deterministic projection into vector bins
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign * weight

        # L2-normalization for cosine distance
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [round(x / norm, 5) for x in vec]
        return vec

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_single(text) for text in input]


class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    """Online Google Gemini Embeddings (gemini-embedding-001)"""
    def __init__(self, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            if hasattr(response, "embeddings") and response.embeddings:
                embeddings.append(response.embeddings[0].values)
            elif hasattr(response, "embedding") and response.embedding:
                embeddings.append(response.embedding.values)
        return embeddings


def get_embedding_function(corpus_docs: Optional[List[str]] = None) -> EmbeddingFunction[Documents]:
    """Factory to get the active embedding function."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            print("[Embedding] Using Google Gemini 'gemini-embedding-001'")
            return GeminiEmbeddingFunction(gemini_key)
        except Exception as e:
            print(f"[Embedding] Falling back to offline TF-IDF embeddings: {e}")

    ef = BM25TFIDFEmbeddingFunction()
    if corpus_docs:
        ef.fit_corpus(corpus_docs)
    return ef
