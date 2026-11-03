"""Rebuild the "_v2" index from the source files with ONE command.

    cd backend
    python -m ingestion.run_ingest                    # screenplay and textbook
    python -m ingestion.run_ingest --source script    # only one of them
    python -m ingestion.run_ingest --recreate         # delete the _v2 collections and rebuild from scratch

Steps for each source: chunk -> embed -> write to Weaviate -> remove stale objects -> verify.
It is safe to run repeatedly: chunks that did not change are replaced, not duplicated.
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import ollama
import weaviate
from weaviate.exceptions import WeaviateConnectionError

from app.core.config import ROOT_DIR
from ingestion import chunk, embed, index

SOURCE_FILES = {"script": "pdf.json", "textbook": "textbook.json"}
VERSION_FILE = Path(__file__).with_name("index_version.json")


def git_state() -> dict:
    """Which commit was this index built from (and was the working tree clean)?"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT_DIR, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT_DIR).stdout.strip()
        )
        return {"commit": commit, "uncommitted_changes": dirty}
    except Exception:
        return {"commit": "unknown", "uncommitted_changes": None}


def ingest_source(client, source: str, recreate: bool) -> dict:
    """Run the whole pipeline for one source and return a summary of what happened."""
    path = ROOT_DIR / "data" / "processed" / SOURCE_FILES[source]
    name = index.COLLECTIONS[source]
    print(f"\n== {source} -> {name} ==")

    chunks = chunk.build_chunks(source, path)
    print(f"  chunked: {len(chunks)} chunks")

    started = time.time()
    vectors = embed.embed_texts([c.text for c in chunks])
    seconds = time.time() - started
    dimensions = len(vectors[0])
    print(f"  embedded: {len(vectors)} vectors of {dimensions} numbers in {seconds:.1f} s ({embed.EMBED_MODEL})")

    collection = index.create_collection(client, name, recreate=recreate)
    index.upsert_chunks(collection, chunks, vectors)
    removed = index.prune_stale(collection, {index.chunk_uuid(c.id) for c in chunks})
    stored = index.count_objects(collection)
    rate = index.self_retrieval_rate(collection, chunks, vectors)
    print(f"  stored in Weaviate: {stored} (expected {len(chunks)}) | stale objects removed: {removed}")
    print(f"  self-retrieval check: {rate:.0%} of sampled chunks found themselves as the best match")

    return {
        "collection": name,
        "chunks": len(chunks),
        "stored": stored,
        "dimensions": dimensions,
        "embed_seconds": round(seconds, 1),
        "ok": stored == len(chunks) and rate == 1.0,
    }


def describe_build(result: dict) -> dict:
    """Everything needed to know how ONE collection was built (its own date, commit, model and settings)."""
    return {
        **result,
        "built_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": git_state(),
        "embedding": {"model": embed.EMBED_MODEL, "input": "raw chunk text (no prefix)"},
        "chunking": {
            "splitter": "langchain RecursiveCharacterTextSplitter",
            "langchain_text_splitters": version("langchain-text-splitters"),
            "chunk_tokens": chunk.CHUNK_TOKENS,
            "chars_per_token": chunk.CHARS_PER_TOKEN,
            "chunk_chars": chunk.CHUNK_CHARS,
            "overlap_chars": chunk.OVERLAP_CHARS,
            "min_chars": chunk.MIN_CHARS,
        },
    }


def update_version_file(results: dict, path: Path = VERSION_FILE) -> None:
    """Record the sources that were just built. Sources that were NOT part of this run keep their old entry."""
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    collections = existing.get("collections", {})
    for source, result in results.items():
        collections[source] = describe_build(result)
    path.write_text(json.dumps({"collections": collections}, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the _v2 Weaviate index.")
    parser.add_argument("--source", choices=[*SOURCE_FILES, "all"], default="all")
    parser.add_argument("--recreate", action="store_true", help="delete and rebuild the _v2 collections")
    args = parser.parse_args(argv)
    sources = list(SOURCE_FILES) if args.source == "all" else [args.source]

    try:
        client = weaviate.connect_to_local()
    except WeaviateConnectionError:
        print("Weaviate is not reachable on localhost:8080. Start Docker Desktop, then run: docker start weaviate")
        return 2
    try:
        results = {s: ingest_source(client, s, args.recreate) for s in sources}
    except ollama.ResponseError as error:
        print(f"Ollama rejected the request: {error}")
        return 2
    except ConnectionError:
        print("Ollama is not reachable on localhost:11434. Start the Ollama app and try again.")
        return 2
    finally:
        client.close()

    update_version_file(results)
    ok = all(r["ok"] for r in results.values())
    print(f"\nrecorded in {VERSION_FILE.name}")
    print("ALL CHECKS PASSED" if ok else "SOMETHING FAILED (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
