# Evaluation log

Every change to the retrieval pipeline gets a row here, measured on the same golden set
(`backend/evals/dataset.jsonl`, 100 questions). Numbers are from this project's own runs on a laptop CPU
(AMD Radeon 780M integrated graphics, so `qwen3:8b` runs on the CPU).

## How to reproduce

```bash
cd backend
python -m evals.run_evals --mode retrieval   # about 1-2 minutes: is the evidence in the top 5 chunks?
python -m evals.run_evals --mode routing     # about 4 minutes: does the router pick the right source?
python -m evals.run_evals --mode full        # about 1.5 hours: router + answer for all 100 questions, both indexes
```

`--configs "v2 hybrid + rerank" ...` picks which retrieval setups to compare (see `RETRIEVAL_CONFIGS` in `evals/run_evals.py`).

## What is measured

| Metric | Meaning |
|---|---|
| hit@5 | share of questions where a quote of the correct evidence appears in one of the top-5 chunks |
| MRR | mean of 1/rank of the first chunk containing the evidence (1.0 = always first, 0 = never found) |
| routing accuracy | the router picked the source the question was labelled with |
| correct / false refusal | on unanswerable / answerable questions, the answer says it does not know (matched by phrase, so rough) |
| p50 / p95 latency | median / 95th percentile seconds to produce an answer (excluding the router) |

Evidence labels are short quotes, not chunk IDs, so the same labels score any chunking. 49 answerable script and
textbook questions have evidence; 3 are excluded from retrieval scoring because the old 245-character chunks cut the
quote in half (q048, q051, q077), leaving 46. **One question is about 2 percentage points, so differences of a few
points are noise.**

## Row 1: structure-aware chunks (v2) vs the original index (v1)

What changed: script and textbook re-chunked from scratch (scene and chapter aware, about 400 tokens with 12.5% overlap,
sentence-aware splits, short tails merged) into new `_v2` collections. Comments are unchanged. Same embedding model.

| | v1 (old) | v2 (new) |
|---|---|---|
| Chunks (script / textbook) | 655 / 2275 | 117 / 420 |
| Characters per chunk | about 245 | about 1,600 |
| hit@5 (46 questions) | 57% | 74% |
| MRR | 0.42 | 0.51 |
| Correct refusal (16 unanswerable) | 75% | 88% |
| False refusal (84 answerable) | 9.5% | 9.5% |
| Latency p50 / p95 | 13.6 s / 47 s | 40.1 s / 113 s |

**Fairness check (equal text).** A v2 chunk holds about 6 times more text, so five v2 chunks can contain a quote more
easily. Giving v1 as many chunks as fit in v2's top-5 text (about 25 chunks, 6,366 characters) finds the evidence in
**76%** of questions, against 74% for v2. **So the retrieval gain in this table comes from bigger chunks, not from better
chunk boundaries.** What the v2 chunks do provide is whole-sentence passages, scene and chapter names and page numbers
(used for citations). The cost is about 3 times slower answers, because the model reads 6 times more text.

## Row 2: router prompt

What changed: the router prompt now describes each source (the textbook also covers how the film was made and the
scientists behind it) and says `comments` is only for questions about what viewers, fans or audiences think or about
counts of comments. Changes were guided by the failure patterns, not by copying questions.

| | before | after |
|---|---|---|
| Routing accuracy, golden set (83 questions) | 87% | **92%** |
| Routing accuracy, hold-out set (24 new questions, `routing_holdout.jsonl`) | 96% | **100%** |

The hold-out questions were written before looking at any router output. Remaining errors on the golden set are
mostly genuinely ambiguous ("What is the film really about?", "Why is the ending so confusing?") or physics-sounding
questions about Miller's planet that the router sends to the script. The hold-out set is small (24), so 100% is not proof
the router is solved.

## Row 3 and 4: hybrid search and reranking

What changed: *hybrid* mixes meaning-based search with exact keyword matching (BM25; alpha 0.75 means 75% meaning,
25% keywords). *Reranking* fetches the top 30 candidates and re-scores each with a cross-encoder
(`ms-marco-MiniLM-L-6-v2`, 80 MB, about 1.5 s for 30 chunks on this CPU), keeping the best 5.

| Setup | hit@5 | MRR | script | textbook |
|---|---|---|---|---|
| v1 vector (original) | 57% | 0.42 | 72% | 38% |
| v2 vector | 74% | 0.51 | 80% | 67% |
| v2 hybrid (alpha 0.75) | 76% | 0.56 | 84% | 67% |
| v2 vector + rerank | 78% | 0.60 | 84% | 71% |
| **v2 hybrid + rerank** | **83%** | **0.61** | 84% | **81%** |
| v1 hybrid + rerank | 65% | 0.52 | 76% | 52% |

- Each step helps a little; together they raise hit@5 from 74% to 83% and MRR from 0.51 to 0.61 on v2.
  The textbook gains most (67% to 81%), where exact names and numbers matter.
- Reranking also helps the old index (57% to 67%), so it is not only compensating for v2's bigger chunks.
- Caveats: alpha 0.5 and 0.75 were compared on this same set (the 0.5 versus 0.75 difference is one question), and
  with 21 to 25 questions per route these are rough. The hybrid and rerank settings were chosen with this set in view,
  so the numbers are probably a little optimistic.
- In the chatbot these are switched on with `RETRIEVAL_MODE=hybrid` and `RERANK_ENABLED=true` in `.env` (they apply to
  script and textbook only; the comments source has no labelled evidence, so it stays on plain vector search).
  Answer-level effects (faithfulness, latency with reranking) have not been run yet.

## Open items

- Answer quality is not measured yet (refusals are matched by phrase; nothing judges whether an answer is correct).
- q058 (box office) and q059 (Dunkirk) are labelled unanswerable, but the comments mention sales and Nolan, so those two
  labels are debatable.
- Run `--mode full` again with `v2 + hybrid + rerank + new router` to get end-to-end numbers.
