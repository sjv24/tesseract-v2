# Evaluation log

Every change to the retrieval pipeline gets a row here, measured on the same golden set
(`backend/evals/dataset.jsonl`, 100 questions). Numbers are from this project's own runs, on a laptop CPU
(AMD Radeon 780M integrated graphics, so `qwen3:8b` runs on the CPU).

## How to reproduce

```bash
cd backend
python -m evals.run_evals --mode retrieval   # about 1 minute: is the evidence in the top 5 chunks?
python -m evals.run_evals --mode full        # about 1.5 hours: router + answer for all 100 questions, both indexes
```

## What is measured

| Metric | Meaning |
|---|---|
| hit@5 | share of questions where a quote of the correct evidence appears in one of the top-5 chunks |
| MRR | mean of 1/rank of the first chunk containing the evidence (1.0 = always first, 0 = never found) |
| routing accuracy | the router picked the source the question was labelled with |
| correct refusal | on unanswerable questions, the answer says it does not know (matched by phrase, so rough) |
| false refusal | on answerable questions, the answer says it does not know |
| p50 / p95 latency | median / 95th percentile seconds to produce an answer (excluding the router) |

Evidence labels are short quotes, not chunk IDs, so the same labels score any chunking. 49 answerable script and
textbook questions have evidence; 3 are excluded from retrieval scoring because the old 245-character chunks cut the
quote in half (q048, q051, q077), leaving 46.

## Row 1: structure-aware chunks (v2) vs the original index (v1)

What changed: script and textbook re-chunked from scratch (scene and chapter aware, about 400 tokens with 12.5% overlap,
sentence-aware splits, short tails merged) and stored in new `_v2` collections. Comments are unchanged. Embeddings are the
same model (`embeddinggemma`).

| | v1 (old) | v2 (new) |
|---|---|---|
| Chunks (script / textbook) | 655 / 2275 | 117 / 420 |
| Characters per chunk | about 245 | about 1,600 |
| **hit@5, all 46 scored questions** | 57% | **74%** |
| hit@5, script (25) | 72% | 80% |
| hit@5, textbook (21) | 38% | 67% |
| MRR, all | 0.42 | 0.51 |
| Correct refusal (16 unanswerable) | 75% | 88% |
| False refusal (84 answerable, see note) | 9.5% | 9.5% |
| **Latency p50 / p95** | **13.6 s / 47 s** | **40.1 s / 113 s** |

Routing accuracy (same router for both): 87% (72 of 83 questions with a labelled route).

### What this does and does not show

- **v2 retrieves the evidence more often**, mostly on textbook questions. But a v2 chunk holds about 6 times more text, so
  five v2 chunks can contain a quote more easily than five v1 chunks. Part of the gain is simply more text per result.
  A fairer comparison at equal total text (for example v1 top-30 against v2 top-5) has not been run yet.
- **v2 is about 3 times slower** per answer, because the model must read about 6 times more text on a CPU. This is the cost
  of the larger chunks and should be weighed against the retrieval gain.
- **Most "false refusals" are router mistakes, not retrieval.** 11 questions were sent to the wrong source (for example
  "What did Maarten Schmidt find when he studied the quasar's light?" went to comments), and a wrong source cannot
  contain the answer. Routing is the next thing to fix.
- **Correct refusal improved partly by luck of the label.** Two questions marked unanswerable (q058 box office, q059 Dunkirk)
  were not refused by either index because the comments really do mention sales figures and Nolan; those two labels are
  debatable. The Penrose-process question (q013) is the clearest win: v1 answered it with a passage about a different
  process, v2 said it does not know.
- Refusals are detected by phrase and answers were not read one by one, so answer correctness is not measured yet
  (that needs a judge, planned later).
- With 21 to 25 questions per slice, differences of a few points are noise.

## Planned rows

- Equal-text-budget comparison of v1 and v2 (fairness check for Row 1)
- Router fixes
- Hybrid search (vector + BM25)
- Reranking
