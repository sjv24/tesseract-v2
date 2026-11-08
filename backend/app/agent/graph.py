from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.router import Category, classify
from app.core.config import settings
from app.generation.generator import generate
from app.retrieval.search import search

# the structure-aware index; the comments have no v2 and always use the original collection
V2_COLLECTIONS = {
    "script": "Interstellar_Script_v2",
    "textbook": "Interstellar_Textbook_v2",
}

SOURCES = {
    "script": {
        "collection": "Interstellar_Script",
        "embedder": "ollama",
        "prompt": (
            "You answer questions about the plot, characters and dialogue of the "
            "film Interstellar using ONLY the provided screenplay excerpts. "
            "If the excerpts do not contain the answer, say you don't know."
        ),
    },
    "textbook": {
        "collection": "Interstellar_Textbook",
        "embedder": "ollama",
        "prompt": (
            "You answer science questions using ONLY the provided excerpts from "
            "'The Science of Interstellar'. If the excerpts do not contain the "
            "answer, say you don't know."
        ),
    },
    "comments": {
        "collection": "Interstellar_Comments",
        "embedder": "openai",
        "prompt": (
            "You summarise what viewers think about the film Interstellar using "
            "ONLY the provided YouTube comments. Describe the range of opinions. "
            "If the comments do not cover the question, say you don't know."
        ),
    },
}

MULTI_PROMPT = (
    "You answer questions about the film Interstellar using ONLY the provided "
    "excerpts. Each excerpt is tagged with its source: [script] is the screenplay, "
    "[textbook] is 'The Science of Interstellar', [comments] are viewer comments. "
    "Combine the sources where they help and mention which source a point comes "
    "from. If the excerpts do not contain the answer, say you don't know."
)


class AgentState(TypedDict, total=False):
    question: str
    category: Category
    answer: str
    citations: list[dict]


def collection_name(source_name: str) -> str:
    """The Weaviate collection to search, following INDEX_VERSION from .env."""
    if settings.index_version == "v2" and source_name in V2_COLLECTIONS:
        return V2_COLLECTIONS[source_name]
    return SOURCES[source_name]["collection"]


def classify_node(state: AgentState) -> dict:
    return {"category": classify(state["question"])}


TOP_K = 5
# hybrid search and reranking were measured on these two sources only (the comments have no labelled evidence)
IMPROVED_RETRIEVAL_SOURCES = {"script", "textbook"}


def _retrieve(source_name: str, question: str) -> list[dict]:
    source = SOURCES[source_name]
    collection = collection_name(source_name)
    improved = source_name in IMPROVED_RETRIEVAL_SOURCES
    rerank_on = improved and settings.rerank_enabled
    hits = search(
        collection,
        question,
        k=settings.rerank_candidates if rerank_on else TOP_K,
        embedder=source["embedder"],
        mode=settings.retrieval_mode if improved else "vector",
        alpha=settings.hybrid_alpha,
    )
    if rerank_on:
        from app.retrieval.rerank import rerank  # imported here so the model only loads when reranking is switched on

        hits = rerank(question, hits, top_n=TOP_K)
    return [{**hit, "source": source_name, "collection": collection} for hit in hits]


def _citation(hit: dict) -> dict:
    # the score's scale depends on how the chunk was ranked: reranker score, else hybrid score, else 1 - cosine distance
    if hit.get("rerank_score") is not None:
        score = hit["rerank_score"]
    elif hit.get("distance") is not None:
        score = 1 - hit["distance"]
    else:
        score = hit["score"]
    return {
        "source": hit["source"],
        "collection": hit["collection"],
        "chunk_id": hit["chunk_id"],
        "section": hit["section"],
        "page": hit["page"],
        "score": round(score, 4),
        "snippet": hit["text"][:200],
    }


def _single_source_agent(source_name: str, state: AgentState) -> dict:
    hits = _retrieve(source_name, state["question"])
    answer = generate(state["question"], [h["text"] for h in hits], SOURCES[source_name]["prompt"])
    return {"answer": answer, "citations": [_citation(h) for h in hits]}


def script_agent(state: AgentState) -> dict:
    return _single_source_agent("script", state)


def textbook_agent(state: AgentState) -> dict:
    return _single_source_agent("textbook", state)


def comments_agent(state: AgentState) -> dict:
    return _single_source_agent("comments", state)


def multi_agent(state: AgentState) -> dict:
    question = state["question"]
    hits = [hit for source_name in SOURCES for hit in _retrieve(source_name, question)]
    chunks = [f"[{h['source']}] {h['text']}" for h in hits]
    return {"answer": generate(question, chunks, MULTI_PROMPT), "citations": [_citation(h) for h in hits]}


def route(state: AgentState) -> str:
    return state["category"]


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify_node)
    builder.add_node("script", script_agent)
    builder.add_node("textbook", textbook_agent)
    builder.add_node("comments", comments_agent)
    builder.add_node("multi", multi_agent)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route,
        {
            "script": "script",
            "textbook": "textbook",
            "comments": "comments",
            "multi": "multi",
        },
    )
    for node in ("script", "textbook", "comments", "multi"):
        builder.add_edge(node, END)

    return builder.compile()


graph = build_graph()
