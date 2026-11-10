"""Tests for ingestion/chunk.py.

Most tests use small MADE-UP data (never text from the book or the screenplay, which are copyrighted).
The last group runs on the real files in data/processed/ and is skipped when they are not present.
"""
import json
import re

import pytest

from app.core.config import ROOT_DIR
from ingestion import chunk as c


# --- helpers ---------------------------------------------------------------------------------
def el(text, category="NarrativeText", page=1):
    """Build one made-up element, the way pdf.json stores them."""
    return {"page": page, "category": category, "text": text}


def write_source(tmp_path, elements, name="source.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"title": "test", "text": elements}), encoding="utf-8")
    return path


SENTENCE_END = re.compile(r"[.!?\"')\]”’]\s*$")


# --- clean_elements --------------------------------------------------------------------------
def test_clean_collapses_whitespace_and_drops_blank_items():
    cleaned = c.clean_elements([el("  The   black hole\nbends   light.  "), el("   "), el("")], "textbook")
    assert [e["text"] for e in cleaned] == ["The black hole bends light."]


def test_clean_drops_headers_and_page_numbers_in_the_screenplay():
    elements = [el("10.", "Header"), el("12."), el("A real line of action."), el("COOPER", "Title")]
    cleaned = c.clean_elements(elements, "script")
    assert [e["text"] for e in cleaned] == ["A real line of action.", "COOPER"]


def test_clean_drops_image_junk_but_keeps_a_bare_number_in_the_textbook():
    cleaned = c.clean_elements([el("D<T", "Image"), el("9", "Image"), el("Some prose.")], "textbook")
    assert [e["text"] for e in cleaned] == ["9", "Some prose."]  # the "9" may be a chapter number


def test_clean_removes_the_ocr_damaged_running_title():
    cleaned = c.clean_elements([el("TH E S CIE NCE O F I NTERSTELLAR", "Title"), el("Real text.")], "textbook")
    assert [e["text"] for e in cleaned] == ["Real text."]


# --- headings and sections -------------------------------------------------------------------
def test_script_scene_headings_start_new_sections():
    elements = [el("TITLE PAGE"), el("INT. KITCHEN - DAY"), el("Someone cooks."), el("EXT. FIELD - NIGHT"), el("Wind.")]
    sections = c.split_into_sections(elements, "script")
    assert [s.title for s in sections] == ["Front matter", "INT. KITCHEN - DAY", "EXT. FIELD - NIGHT"]
    assert sections[1].lines == ["INT. KITCHEN - DAY", "Someone cooks."]


def test_textbook_only_real_titles_are_headings():
    assert c.is_heading(el("Tidal Gravity", "Title"), "textbook")
    assert not c.is_heading(el("of our universe's birth.", "Title"), "textbook")  # fragment, not a heading
    assert not c.is_heading(el("Tidal Gravity", "NarrativeText"), "textbook")  # wrong category
    assert not c.is_heading(el("Tidal Gravity", "Title"), "script")


def test_chapter_number_printed_alone_is_merged_with_the_next_item():
    elements = [
        el("1", "Title", 3),
        el("First Chapter", "NarrativeText", 3),
        el("Some intro text.", "NarrativeText", 3),
        el("A Section", "Title", 4),
        el("Section text.", "NarrativeText", 4),
    ]
    sections = c.split_into_sections(elements, "textbook")
    titles = [s.title for s in sections]
    assert "1 First Chapter" in titles
    assert sections[-1].title == "A Section"
    assert sections[-1].chapter == "1 First Chapter"  # later sections inherit the chapter


def test_only_the_next_expected_chapter_number_counts():
    elements = [
        el("7", "Title"),  # stray number: we are waiting for chapter 1
        el("Ignored words."),
        el("1 Real Chapter", "Title"),  # a chapter number and title in ONE item
        el("Body."),
        el("42", "UncategorizedText"),  # stray page number: must be dropped
        el("More body."),
    ]
    sections = c.split_into_sections(elements, "textbook")
    chapters = {s.chapter for s in sections if s.chapter}
    assert chapters == {"1 Real Chapter"}
    all_lines = [line for s in sections for line in s.lines]
    assert "42" not in all_lines and "7" not in all_lines


# --- pack_sections ---------------------------------------------------------------------------
def section(title, text):
    return c.Section(title=title, chapter="", page=1, lines=[text])


def test_small_neighbouring_sections_are_merged():
    sections = [section("A", "x" * 100), section("B", "y" * 100), section("C", "z" * 100)]
    docs = c.pack_sections(sections, "textbook", max_chars=1000)
    assert len(docs) == 1
    assert docs[0].metadata["section"] == "A"  # tagged with the first section
    assert len(docs[0].page_content) == 100 * 3 + 2  # three texts and two joining spaces


def test_a_section_that_does_not_fit_starts_a_new_document():
    sections = [section("A", "x" * 600), section("B", "y" * 600)]
    docs = c.pack_sections(sections, "textbook", max_chars=1000)
    assert [d.metadata["section"] for d in docs] == ["A", "B"]


def test_a_tiny_section_is_never_left_alone():
    heading_only = section("Heading", "Usual Viewpoint")  # well under MIN_CHARS
    big = section("Body", "y" * 1500)  # does not fit next to it
    docs = c.pack_sections([heading_only, big], "textbook", max_chars=1000)
    assert len(docs) == 1  # merged anyway (the splitter will cut it later)
    assert docs[0].page_content.startswith("Usual Viewpoint")


def test_screenplay_sections_are_joined_with_newlines():
    docs = c.pack_sections([section("A", "line one"), section("B", "line two")], "script", max_chars=1000)
    assert docs[0].page_content == "line one\nline two"


def test_textbook_sections_are_joined_with_spaces_because_the_book_flows_as_prose():
    docs = c.pack_sections([section("A", "part one"), section("B", "part two")], "textbook", max_chars=1000)
    assert docs[0].page_content == "part one part two"


# --- the splitter ----------------------------------------------------------------------------
def test_splitter_prefers_to_cut_between_sentences():
    text = " ".join(f"This is sentence number {i} of the test." for i in range(200))
    pieces = c.make_splitter().split_text(text)
    assert len(pieces) > 3
    assert all(SENTENCE_END.search(p) for p in pieces)


def test_splitter_never_exceeds_the_size_limit_even_without_any_spaces():
    pieces = c.make_splitter().split_text("x" * 10_000)
    assert len(pieces) > 1
    assert all(len(p) <= c.CHUNK_CHARS for p in pieces)


# --- merge_small_tail ------------------------------------------------------------------------
def doc(text):
    return c.Document(page_content=text, metadata={"section": "S", "chapter": "", "page": 1})


def test_a_short_tail_is_glued_on_and_the_duplicated_overlap_is_removed():
    previous = " ".join(f"word{i:03d}" for i in range(150))  # a long piece made of distinct words
    overlap = previous[-60:]  # the splitter starts the next piece with a copy of the previous ending
    tail = overlap + " unique ending"
    merged = c.merge_small_tail([doc(previous), doc(tail)], " ")
    assert len(merged) == 1
    assert merged[0].page_content == previous + " unique ending"  # the overlap appears once, not twice


def test_a_short_tail_without_any_overlap_is_joined_with_the_joiner():
    merged = c.merge_small_tail([doc("first " * 100), doc("unrelated leftover")], "\n")
    assert merged[0].page_content == "first " * 100 + "\n" + "unrelated leftover"


def test_a_tail_that_is_long_enough_is_left_alone():
    pieces = [doc("a" * 1000), doc("b" * c.MIN_CHARS)]
    assert len(c.merge_small_tail(pieces, " ")) == 2


def test_a_single_short_piece_is_left_alone():
    assert len(c.merge_small_tail([doc("short")], " ")) == 1


@pytest.mark.parametrize("sentences", range(55, 130, 3))
def test_no_section_length_ever_leaves_a_short_leftover_chunk(tmp_path, sentences):
    elements = [el("1 Chapter", "Title")] + [el(f"Sentence number {i:04d} is right here.") for i in range(sentences)]
    chunks = c.build_chunks("textbook", write_source(tmp_path, elements))
    if len(chunks) > 1:
        assert all(len(x.text) >= c.MIN_CHARS for x in chunks)
    assert all(len(x.text) <= c.MAX_CHUNK_CHARS for x in chunks)
    blob = " ".join(x.text for x in chunks)
    assert all(f"Sentence number {i:04d} is right here." in blob for i in range(sentences))  # nothing lost


# --- chunk_id --------------------------------------------------------------------------------
def test_chunk_id_is_stable_and_changes_with_every_input():
    base = c.chunk_id("script", 1, "hello")
    assert base == c.chunk_id("script", 1, "hello")
    assert base != c.chunk_id("script", 1, "hellp")
    assert base != c.chunk_id("script", 2, "hello")
    assert base != c.chunk_id("textbook", 1, "hello")
    assert re.fullmatch(r"[0-9a-f]{16}", base)


# --- build_chunks on made-up files -----------------------------------------------------------
def long_source(tmp_path):
    elements = [el("INT. ROOM - DAY", page=1)]
    elements += [el(f"Sentence number {i} happens in this scene and nothing else.", page=1 + i // 40) for i in range(300)]
    elements += [el("EXT. GARDEN - NIGHT", page=9), el("A short scene.", page=9)]
    return write_source(tmp_path, elements)


def test_build_chunks_respects_size_and_numbers_chunks_from_one(tmp_path):
    chunks = c.build_chunks("script", long_source(tmp_path))
    assert len(chunks) > 3
    assert [x.position for x in chunks] == list(range(1, len(chunks) + 1))
    assert all(len(x.text) <= c.MAX_CHUNK_CHARS for x in chunks)


def test_build_chunks_is_deterministic_and_ids_are_unique(tmp_path):
    path = long_source(tmp_path)
    first = c.build_chunks("script", path)
    second = c.build_chunks("script", path)
    assert [x.id for x in first] == [x.id for x in second]
    assert len({x.id for x in first}) == len(first)


def test_neighbouring_chunks_of_a_long_section_overlap(tmp_path):
    chunks = c.build_chunks("script", long_source(tmp_path))
    first, second = chunks[0].text, chunks[1].text
    shared = [n for n in range(40, c.OVERLAP_CHARS + 60) if first.endswith(second[:n])]
    assert shared, "the start of chunk 2 should repeat the end of chunk 1"


def test_build_chunks_loses_no_text(tmp_path):
    path = long_source(tmp_path)
    cleaned = c.clean_elements(c.load_elements(path), "script")
    blob = " ".join(x.text.replace("\n", " ") for x in c.build_chunks("script", path))
    missing = [e["text"] for e in cleaned if e["text"] not in blob]
    assert missing == []


def test_chunks_carry_section_chapter_and_page(tmp_path):
    elements = [el("1 Opening", "Title", 5), el("Body of the chapter. " * 20, "NarrativeText", 5)]
    chunk = c.build_chunks("textbook", write_source(tmp_path, elements))[0]
    assert (chunk.section, chunk.chapter, chunk.page) == ("1 Opening", "1 Opening", 5)
    assert chunk.source == "textbook"


def test_an_empty_source_gives_no_chunks(tmp_path):
    assert c.build_chunks("script", write_source(tmp_path, [])) == []


# --- real data (skipped when the copyrighted source files are not on this machine) ------------
DATA = ROOT_DIR / "data" / "processed"
needs_data = pytest.mark.skipif(
    not (DATA / "pdf.json").exists() or not (DATA / "textbook.json").exists(),
    reason="source files are not in data/processed/ (they are git-ignored)",
)


@needs_data
def test_real_screenplay_is_chunked_sensibly():
    chunks = c.build_chunks("script", DATA / "pdf.json")
    assert 90 <= len(chunks) <= 150
    assert all(len(x.text) <= c.MAX_CHUNK_CHARS for x in chunks)
    assert all(len(x.text) >= c.MIN_CHARS for x in chunks)  # no tiny chunks in the screenplay
    ends = sum(1 for x in chunks if SENTENCE_END.search(x.text))
    assert ends / len(chunks) >= 0.75  # the old index managed 13%


@needs_data
def test_real_textbook_is_chunked_sensibly_and_all_31_chapters_are_found():
    chunks = c.build_chunks("textbook", DATA / "textbook.json")
    assert 350 <= len(chunks) <= 520
    assert all(len(x.text) <= c.MAX_CHUNK_CHARS for x in chunks)
    assert all(len(x.text) >= c.MIN_CHARS for x in chunks)  # short leftovers are merged back
    numbers = {int(x.chapter.split()[0]) for x in chunks if x.chapter}
    assert numbers == set(range(1, 32))
    ends = sum(1 for x in chunks if SENTENCE_END.search(x.text))
    assert ends / len(chunks) >= 0.8  # the old index managed 6%


@needs_data
def test_real_ids_are_unique_and_reproducible():
    first = c.build_chunks("textbook", DATA / "textbook.json")
    second = c.build_chunks("textbook", DATA / "textbook.json")
    assert [x.id for x in first] == [x.id for x in second]
    assert len({x.id for x in first}) == len(first)
