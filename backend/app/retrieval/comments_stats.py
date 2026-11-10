"""Counting questions about the YouTube comments ("how many mention X", "what share are negative").

Vector search can only return the five most similar comments, so it cannot count. These functions count over ALL
comments instead. They are plain Python over a list of rows, so they are easy to test; load_comments() fills that
list from Weaviate once and keeps it in memory.
"""
import ast
import re
from collections import Counter
from functools import lru_cache

from app.retrieval.store import get_client

COMMENTS_COLLECTION = "Interstellar_Comments"
# words that are the topic of every comment, so they say nothing when ranking "most mentioned topics"
GENERIC_ASPECTS = {
    "movie", "movies", "film", "films", "video", "videos", "interstellar", "content", "guy", "man", "thing", "things",
    "people", "one", "way", "lot", "part",
}


def parse_row(properties: dict) -> dict:
    """Weaviate stores the label lists as text such as "['Negative']"; turn them back into real lists."""
    return {
        "text": properties["text"],
        "sentiment": ast.literal_eval(properties["sentiment"]),
        "aspects": ast.literal_eval(properties["aspects"]),
    }


@lru_cache(maxsize=1)
def load_comments() -> tuple[dict, ...]:
    """Every comment, fetched once (about 41,000 rows)."""
    collection = get_client().collections.use(COMMENTS_COLLECTION)
    return tuple(parse_row(o.properties) for o in collection.iterator())


def mentions(row: dict, keyword: str) -> bool:
    """Whole-word, case-insensitive match, so 'ending' does not match 'pending'."""
    return re.search(rf"\b{re.escape(keyword)}\b", row["text"], re.IGNORECASE) is not None


def select(rows, sentiment: str | None = None, keyword: str | None = None) -> list[dict]:
    """Comments that carry the sentiment label (a comment can carry several) and mention the keyword."""
    return [
        r for r in rows
        if (sentiment is None or sentiment in r["sentiment"]) and (keyword is None or mentions(r, keyword))
    ]


def count(rows, sentiment=None, keyword=None) -> int:
    return len(select(rows, sentiment, keyword))


def share(rows, sentiment=None, keyword=None) -> dict:
    """Matching comments as a share of a stated base: all comments, or, when a sentiment is asked for,
    the comments that have any sentiment label at all (about a third of them have none)."""
    matching = count(rows, sentiment, keyword)
    base_rows = [r for r in rows if r["sentiment"]] if sentiment else rows
    base = len(base_rows)
    base_name = "comments with a sentiment label" if sentiment else "all comments"
    return {"matching": matching, "base": base, "base_name": base_name, "share": matching / base if base else 0.0}


def ratio(rows, sentiment: str, versus: str, keyword=None) -> dict:
    top, bottom = count(rows, sentiment, keyword), count(rows, versus, keyword)
    return {"top": top, "bottom": bottom, "ratio": top / bottom if bottom else None}


def top_aspects(rows, sentiment=None, keyword=None, n: int = 5) -> list[tuple[str, int]]:
    """The most frequent topic words among the matching comments (case-folded, generic words skipped)."""
    counts = Counter()
    for row in select(rows, sentiment, keyword):
        counts.update({a.lower() for a in row["aspects"]} - GENERIC_ASPECTS)
    return counts.most_common(n)
