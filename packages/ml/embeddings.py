from __future__ import annotations

import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns (N, 384) float32 array with L2-normalized embeddings."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for unit-normalized vectors — just dot product."""
    return float(np.dot(a, b))


def mean_pairwise_cosine_sim(embeddings: np.ndarray) -> float:
    """
    Mean upper-triangle cosine similarity. Used as semantic_redundancy signal.
    High value = content repeats itself = more compressible.
    """
    if len(embeddings) < 2:
        return 0.0
    # embeddings already normalized → gram matrix = cosine sims
    gram = embeddings @ embeddings.T
    n = len(gram)
    upper = gram[np.triu_indices(n, k=1)]
    return float(np.mean(upper))
