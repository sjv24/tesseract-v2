from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.router import Category, classify
from app.generation.generator import generate
from app.retrieval.search import search

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


def classify_node(state: AgentState) -> dict:
    return {"category": classify(state["question"])}


def _retrieve(source_name: str, question: str) -> list[str]:
    source = SOURCES[source_name]
    hits = search(source["collection"], question, embedder=source["embedder"])
    return [hit["text"] for hit in hits]


def _single_source_agent(source_name: str, state: AgentState) -> dict:
    chunks = _retrieve(source_name, state["question"])
    answer = generate(state["question"], chunks, SOURCES[source_name]["prompt"])
    return {"answer": answer}


def script_agent(state: AgentState) -> dict:
    return _single_source_agent("script", state)


def textbook_agent(state: AgentState) -> dict:
    return _single_source_agent("textbook", state)


def comments_agent(state: AgentState) -> dict:
    return _single_source_agent("comments", state)


def multi_agent(state: AgentState) -> dict:
    question = state["question"]
    chunks = []
    for source_name in SOURCES:
        for text in _retrieve(source_name, question):
            chunks.append(f"[{source_name}] {text}")
    return {"answer": generate(question, chunks, MULTI_PROMPT)}


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
