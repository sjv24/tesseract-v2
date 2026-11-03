"""Print quality statistics for the chunker, so a change to its settings can be judged in one command.

    cd backend
    python -m ingestion.chunk_stats

Exits with code 1 if a hard check fails (duplicate IDs or lost text), so it can also run in CI.
"""
import re
import statistics
import sys

from app.core.config import ROOT_DIR
from ingestion import chunk as c

DATA = ROOT_DIR / "data" / "processed"
SOURCES = {"script": "pdf.json", "textbook": "textbook.json"}
SENTENCE_END = re.compile(r"[.!?\"')\]”’]\s*$")
OLD_INDEX_SENTENCE_END = {"script": "13%", "textbook": "6%"}  # measured on the 245-character index


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "n/a"


def check_source(source: str, filename: str) -> bool:
    path = DATA / filename
    if not path.exists():
        print(f"{source}: {path} not found (copy the source files into data/processed/)")
        return False

    chunks = c.build_chunks(source, path)
    if not chunks:
        print(f"{source}: no chunks produced")
        return False

    sizes = sorted(len(x.text) for x in chunks)
    ends = sum(1 for x in chunks if SENTENCE_END.search(x.text))
    unique_ids = len({x.id for x in chunks}) == len(chunks)

    cleaned = c.clean_elements(c.load_elements(path), source)
    blob = " ".join(x.text.replace("\n", " ") for x in chunks)
    lost = [e["text"][:50] for e in cleaned if len(e["text"]) > 3 and e["text"][:40] not in blob]

    print("=" * 72)
    print(f"{source}: {len(chunks)} chunks from {len(cleaned)} cleaned elements")
    print(f"  size (chars): min={sizes[0]}  p10={sizes[len(sizes) // 10]}  median={statistics.median(sizes):.0f}  "
          f"p90={sizes[9 * len(sizes) // 10]}  max={sizes[-1]}   (target {c.CHUNK_CHARS}, hard max {c.MAX_CHUNK_CHARS})")
    print(f"  end at a sentence end: {ends}/{len(chunks)} = {percent(ends, len(chunks))}   "
          f"(old 245-char index: {OLD_INDEX_SENTENCE_END[source]})")
    print(f"  under {c.MIN_CHARS} chars: {sum(1 for s in sizes if s < c.MIN_CHARS)}")
    print(f"  distinct sections: {len({x.section for x in chunks})}")
    if source == "textbook":
        chapters = sorted({int(x.chapter.split()[0]) for x in chunks if x.chapter})
        gaps = [n for n in range(1, max(chapters) + 1) if n not in chapters] if chapters else []
        print(f"  chapters detected: {len(chapters)}" + (f"  (missing: {gaps})" if gaps else "  (no gaps)"))
    print(f"  ids unique: {unique_ids}")
    print(f"  text lost: {len(lost)} elements" + (f"  e.g. {lost[:3]}" if lost else ""))

    sample = chunks[len(chunks) // 2]
    print(f"  sample (chunk {sample.position}, section={sample.section!r}, page={sample.page}):")
    print("    " + sample.text[:260].replace("\n", " ⏎ ") + " ...")
    return unique_ids and not lost


def main() -> int:
    ok = True
    for source, filename in SOURCES.items():
        ok = check_source(source, filename) and ok
    print("=" * 72)
    print("ALL CHECKS PASSED" if ok else "SOMETHING FAILED (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
