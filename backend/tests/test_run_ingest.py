"""Tests for the version record written by ingestion/run_ingest.py."""
import json

from ingestion import run_ingest


def entry(name, chunks):
    return {"collection": name, "chunks": chunks, "stored": chunks, "dimensions": 768, "embed_seconds": 1.0, "ok": True}


def test_each_collection_records_how_it_was_built(tmp_path):
    path = tmp_path / "index_version.json"
    run_ingest.update_version_file({"script": entry("Interstellar_Script_v2", 117)}, path)
    record = json.loads(path.read_text(encoding="utf-8"))["collections"]["script"]
    assert record["chunks"] == 117
    assert record["embedding"]["model"] == "embeddinggemma"
    assert record["chunking"]["chunk_tokens"] == 400
    assert "built_at" in record and "git" in record


def test_a_partial_run_keeps_the_other_sources_entry(tmp_path):
    path = tmp_path / "index_version.json"
    run_ingest.update_version_file({"script": entry("Interstellar_Script_v2", 117)}, path)
    run_ingest.update_version_file({"textbook": entry("Interstellar_Textbook_v2", 430)}, path)
    collections = json.loads(path.read_text(encoding="utf-8"))["collections"]
    assert set(collections) == {"script", "textbook"}  # the second run did not erase the first

    run_ingest.update_version_file({"script": entry("Interstellar_Script_v2", 120)}, path)
    collections = json.loads(path.read_text(encoding="utf-8"))["collections"]
    assert collections["script"]["chunks"] == 120  # rebuilt: its entry is replaced
    assert collections["textbook"]["chunks"] == 430  # untouched
