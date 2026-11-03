"""Structure-aware chunking for the Interstellar sources (screenplay and textbook).

Pipeline:
    load elements -> clean -> split into sections -> pack small sections together
    -> split oversize sections with LangChain -> attach stable IDs
"""
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Settings: the knobs we will tune later with the evals ---------------------------------
CHARS_PER_TOKEN = 4  # rule of thumb for English text
CHUNK_TOKENS = 400  # target chunk size
OVERLAP_RATIO = 0.125  # share of a chunk repeated at the start of the next one
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN  # 1600 characters
OVERLAP_CHARS = int(CHUNK_CHARS * OVERLAP_RATIO)  # 200 characters
MIN_CHARS = 300  # a section smaller than this is always merged with the next one

# Element categories that are noise (page numbers, running titles, OCR junk from figures)
DROP_CATEGORIES = {"script": {"Header"}, "textbook": {"Header", "Image"}}
# How elements are glued back together: screenplay lines stay on their own line, book text flows as prose
JOINERS = {"script": "\n", "textbook": " "}

SCENE_RE = re.compile(r"^\s*(INT\.|EXT\.|INT/EXT\.|INT\./EXT\.)", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^\d{1,2}\s+[A-Z]")
PAGE_NUMBER_RE = re.compile(r"^\d{1,3}\.?$")
BARE_NUMBER_RE = re.compile(r"^\d{1,2}$")
# whitespace that comes right after the end of a sentence: where we prefer to cut
SENTENCE_BREAK = r"(?<=[.!?…”’\"])\s+"
# the book title printed at the top of pages, damaged by OCR into things like "TH E S CIE NCE O F I NTERSTELLAR"
RUNNING_TITLE = "THESCIENCEOFINTERSTELLAR"


@dataclass
class Section:
    title: str
    chapter: str
    page: int
    lines: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    source: str
    position: int  # 1-based, in reading order
    text: str
    section: str  # scene heading or book section where the chunk starts
    chapter: str  # textbook chapter ("" for the screenplay)
    page: int  # page where the chunk starts


def load_elements(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data["text"]


def clean_elements(elements: list[dict], source: str) -> list[dict]:
    kept = []
    for element in elements:
        text = " ".join(element.get("text", "").split())  # collapse any run of whitespace to one space
        if not text:
            continue
        is_chapter_number_candidate = source == "textbook" and BARE_NUMBER_RE.match(text)
        if element.get("category") in DROP_CATEGORIES[source] and not is_chapter_number_candidate:
            continue  # (a bare number labelled "Image" can still be a chapter number: keep those)
        if source == "script" and PAGE_NUMBER_RE.match(text):
            continue  # page numbers (in the textbook a bare number may be a chapter number: decided later)
        if re.sub(r"\s+", "", text).upper() == RUNNING_TITLE:
            continue
        kept.append({"page": element.get("page", 0), "category": element.get("category", ""), "text": text})
    return kept


def looks_like_heading(text: str) -> bool:
    starts_ok = text[0].isupper() or bool(CHAPTER_RE.match(text))
    return 3 <= len(text) <= 100 and starts_ok and not text.endswith((".", ",", ";", ":"))


def is_heading(element: dict, source: str) -> bool:
    if source == "script":
        return bool(SCENE_RE.match(element["text"]))
    return element["category"] == "Title" and looks_like_heading(element["text"])


def split_into_sections(elements: list[dict], source: str) -> list[Section]:
    first_page = elements[0]["page"] if elements else 0
    sections = [Section(title="Front matter", chapter="", page=first_page)]
    chapter = ""
    next_chapter = 1  # chapters are numbered 1, 2, 3...: only the NEXT number counts as a chapter marker
    pending_number = None  # a chapter number printed alone; the chapter title is the following element
    for element in elements:
        if source == "textbook":
            if BARE_NUMBER_RE.match(element["text"]):
                if int(element["text"]) == next_chapter:
                    pending_number = element["text"]
                    next_chapter += 1
                continue  # any other bare number is a stray page or footnote number: drop it
            if pending_number is not None:
                element = {**element, "category": "Title", "text": f"{pending_number} {element['text']}"}
                pending_number = None
                chapter = element["text"]
            elif CHAPTER_RE.match(element["text"]) and element["category"] == "Title":
                if int(element["text"].split()[0]) == next_chapter:
                    chapter = element["text"]
                    next_chapter += 1
        if is_heading(element, source):
            sections.append(Section(title=element["text"], chapter=chapter, page=element["page"]))
        sections[-1].lines.append(element["text"])
    return [s for s in sections if s.lines]


def pack_sections(sections: list[Section], source: str, max_chars: int) -> list[Document]:
    """Merge neighbouring small sections so we do not end up with tiny chunks."""
    joiner = JOINERS[source]
    documents: list[Document] = []
    current: Document | None = None
    for section in sections:
        text = joiner.join(section.lines)
        fits = current is not None and len(current.page_content) + len(joiner) + len(text) <= max_chars
        too_small = current is not None and len(current.page_content) < MIN_CHARS
        if current is not None and (fits or too_small):
            current.page_content += joiner + text
        else:
            if current is not None:
                documents.append(current)
            current = Document(
                page_content=text,
                metadata={"section": section.title, "chapter": section.chapter, "page": section.page},
            )
    if current is not None:
        documents.append(current)
    return documents


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS,
        chunk_overlap=OVERLAP_CHARS,
        length_function=len,
        separators=["\n\n", SENTENCE_BREAK, "\n", " ", ""],  # biggest meaning first, then sentence ends
        is_separator_regex=True,
    )


def chunk_id(source: str, position: int, text: str) -> str:
    raw = f"{source}|{position}|{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def build_chunks(source: str, path: Path) -> list[Chunk]:
    elements = clean_elements(load_elements(path), source)
    sections = split_into_sections(elements, source)
    documents = pack_sections(sections, source, CHUNK_CHARS)
    pieces = make_splitter().split_documents(documents)
    return [
        Chunk(id=chunk_id(source, position, piece.page_content), source=source, position=position,
              text=piece.page_content, **piece.metadata)
        for position, piece in enumerate(pieces, start=1)
    ]
