"""Second-stage ranking: a cross-encoder reads the question together with each candidate chunk and re-scores it.

Search (fast, rough) finds ~30 candidates; the reranker (slower, more precise) keeps the best 5.
"""
from fastembed.rerank.cross_encoder import TextCrossEncoder

RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # 80 MB, runs on the CPU

_model: TextCrossEncoder | None = None


def get_model() -> TextCrossEncoder:
    """Load the model once (the first call downloads it if it is not cached)."""
    global _model
    if _model is None:
        _model = TextCrossEncoder(model_name=RERANK_MODEL)
    return _model


def rerank(question: str, hits: list[dict], top_n: int = 5) -> list[dict]:
    """Return the top_n hits ordered by relevance to the question, each with a "rerank_score"."""
    if not hits:
        return []
    scores = list(get_model().rerank(question, [h["text"] for h in hits]))
    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
    return [{**hit, "rerank_score": float(score)} for hit, score in ranked[:top_n]]
