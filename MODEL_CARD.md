# GroundedGuard: model card

GroundedGuard flags the parts of a retrieval-augmented (RAG) answer that its source
context does not support. Given a context and a response, it returns character spans of
the response that are likely hallucinated, plus a response-level decision.

| | default | fast / low-memory option |
|---|---|---|
| checkpoint | `checkpoints/modernbert_lr5e-5_s42` | `checkpoints/B_min_full_e8` |
| encoder | ModernBERT-base (149.6 M params), full context up to 8,192 tokens | DistilBERT-base-uncased (66.4 M), 512-token sliding windows, min over windows |
| threshold (frozen) | **0.85** (`outputs/module5b/frozen_choice.json`) | **0.91** (`outputs/module5/frozen_choice.json`) |
| select with | `--model modernbert` (default) | `--model distilbert` |

Seed 42 is served because it was the pre-registered default; it was not picked by test score.

## Intended use

- Research and engineering use: highlighting response spans in English RAG outputs that a
  reviewer should check against the source, and flagging responses for review.
- Input: a response plus one of: a question with passages (QA), a plain document
  (summary-style), or a structured record (data-to-text, a JSON object).
- Not intended as the sole basis for any decision, and not validated for clinical, legal or
  financial content: the training data contains none of these (see *Domain shift* below).
  A response with no flagged span is **not** verified as correct.

## Training data

[RAGTruth](https://github.com/ParticleMedia/RAGTruth) (Niu et al., ACL 2024), English,
word-level hallucination annotations of responses from six LLMs (GPT-4-0613,
GPT-3.5-turbo-0613, Llama-2-7B/13B/70B-chat, Mistral-7B-Instruct):

- **QA**: MS MARCO questions with retrieved passages
- **Summarization**: CNN/Daily Mail and recent news articles
- **Data-to-text**: Yelp business records (JSON) to text

Responses with `quality == "good"` are kept (as in the official baseline). The official
train split is divided into train (14,047 responses) and dev (895; 50 sources per task,
`random.seed(2024)`, as in the official baseline); test is the official test split
(2,675 responses). Overlapping annotated spans are merged; every gold span counts as a
hallucination, including spans flagged `implicit_true` or `due_to_null`.

## Training procedure

Token classification with one logit per token (BCE, `pos_weight` = neg/pos ≈ 17.1),
AdamW, lr 5e-5 (chosen on dev from {2e-5, 5e-5}), weight decay 0.01, 10% linear warmup,
8 epochs, grad clip 1.0, bf16 autocast, batches of ≤ 65,536 padded tokens. Input:
`[CLS] context [SEP] response [SEP]`; the QA question is placed before the passages; data-to-text
records are serialized as `key: value` lines. The epoch and the threshold were chosen on
dev only (dev All span F1); the test set was scored once after both were frozen.

## Evaluation (RAGTruth test, official metric, mean ± sample std over seeds 42/43/44)

Span level: character overlap between predicted and gold spans, micro-averaged; case level: a response is
positive if it has ≥ 1 span. The metric was validated against the official
`compute_metrics.py` (identical when given the same offsets; the official script locates
predicted strings with `str.find`, ours uses offsets).

| task | model | case P | case R | case F1 | span P | span R | span F1 |
|---|---|---|---|---|---|---|---|
| QA | **ModernBERT (full context)** | 0.5616 ± 0.0184 | 0.7458 ± 0.0509 | 0.6401 ± 0.0217 | 0.6148 ± 0.0494 | 0.6133 ± 0.0563 | **0.6110 ± 0.0037** |
| | DistilBERT (windows) | 0.5495 ± 0.0173 | 0.6750 ± 0.0225 | 0.6056 ± 0.0147 | 0.5504 ± 0.0211 | 0.4620 ± 0.0095 | 0.5022 ± 0.0123 |
| Summary | **ModernBERT** | 0.5877 ± 0.0488 | 0.4608 ± 0.0468 | 0.5140 ± 0.0181 | 0.5194 ± 0.0440 | 0.2524 ± 0.0447 | **0.3370 ± 0.0353** |
| | DistilBERT | 0.4791 ± 0.0152 | 0.3562 ± 0.0172 | 0.4085 ± 0.0155 | 0.3759 ± 0.0054 | 0.1681 ± 0.0070 | 0.2322 ± 0.0058 |
| Data2txt | **ModernBERT** | 0.8328 ± 0.0318 | 0.9102 ± 0.0155 | 0.8694 ± 0.0129 | 0.5017 ± 0.0424 | 0.6269 ± 0.0556 | **0.5547 ± 0.0093** |
| | DistilBERT | 0.8235 ± 0.0093 | 0.8538 ± 0.0115 | 0.8383 ± 0.0069 | 0.4096 ± 0.0110 | 0.4938 ± 0.0107 | 0.4477 ± 0.0058 |
| **All** | **ModernBERT** | 0.7361 ± 0.0314 | 0.7851 ± 0.0238 | **0.7592 ± 0.0115** | 0.5447 ± 0.0411 | 0.5429 ± 0.0531 | **0.5411 ± 0.0054** |
| | DistilBERT | 0.7115 ± 0.0030 | 0.7158 ± 0.0143 | 0.7136 ± 0.0075 | 0.4538 ± 0.0141 | 0.4134 ± 0.0017 | 0.4326 ± 0.0056 |
| All | baseline: whole response | 0.3525 | 1.0000 | 0.5213 | 0.0407 | 1.0000 | 0.0783 |
| All | baseline: lexical overlap (k=2) | 0.3988 | 0.9565 | 0.5629 | 0.0751 | 0.2264 | 0.1128 |

Sources: `outputs/module5b/test_results.json`, `outputs/module5/test_results.json`,
`outputs/module3/baselines.json`.

### Comparison with the RAGTruth paper

Published numbers (in %) from arXiv:2401.00396v2, Tables 5 and 6, checked against the v2 PDF
(every value identical to the HTML version); they were not re-run here. The paper describes
its test set as 450 instances, 150 per task, which matches the 450 test sources (150 per task,
6 responses each) of the official split. Differences to keep in mind: the paper's detector is a fine-tuned
Llama-2-13B (~87× ModernBERT-base's parameters); we did not verify that the paper's test
set uses the same `quality == "good"` filter as the official baseline script; the paper's
metric locates predicted strings with `str.find`.

| | QA F1 | Data2txt F1 | Summary F1 | Overall P | Overall R | Overall F1 |
|---|---|---|---|---|---|---|
| **span level**: paper, fine-tuned Llama-2-13B | 58.2 | 53.5 | 38.8 | 55.6 | 50.2 | 52.7 |
| paper, prompt gpt-4-turbo | 32.6 | 28.2 | 24.1 | 18.4 | 60.9 | 28.3 |
| GroundedGuard ModernBERT (3 seeds) | 61.1 | 55.5 | 33.7 | 54.5 | 54.3 | 54.1 ± 0.5 |
| **response level**: paper, fine-tuned Llama-2-13B | 68.2 | 88.1 | 59.1 | 76.9 | 80.7 | 78.7 |
| paper, prompt gpt-4-turbo | 45.6 | 78.3 | 47.6 | 46.9 | 97.9 | 63.4 |
| GroundedGuard ModernBERT (case level) | 64.0 | 86.9 | 51.4 | 73.6 | 78.5 | 75.9 ± 1.2 |

Read as: at span level, ModernBERT is on par with or above the paper's fine-tuned Llama-2-13B on QA
and data-to-text, and below it on summarization; at response level it is below on all three tasks.

## Efficiency (RTX PRO 6000 Blackwell; CPU = AMD EPYC 7763, 4 threads)

| | ModernBERT | DistilBERT |
|---|---|---|
| `predict()` latency, batch 1, 200 dev responses, GPU bf16: median / p95 | 16.4 / 18.4 ms | 6.2 / 7.9 ms |
| same, CPU fp32 (4 threads): median / p95 | 880.6 / 1,693.0 ms | 408.6 / 1,281.2 ms |
| peak GPU memory per training process | 47.2 GiB | 5.7 GiB |
| train time per epoch | 171.2 s | 66.2 s |
| API (`/detect`, fp32, 1 uvicorn worker), 200 sequential requests: client p50 / p95 / p99, throughput | 16.5 / 22.8 / 30.0 ms, 56.8 req/s | not measured |
| API, 4 concurrent clients × 50 requests: client p50 / p95 / p99, throughput | 60.8 / 72.9 / 76.4 ms, 64.0 req/s | not measured |

Sources: `outputs/module6/latency.json`, `outputs/module5b/comparison.json`, `outputs/module7/load_test.json`.

## Threshold

A token is flagged when its probability is ≥ the threshold; consecutive flagged tokens form one
span; a response is flagged if it has at least one span. The served thresholds (ModernBERT
0.85, DistilBERT 0.91) maximize dev All span F1 over 0.05–0.90 (step 0.05) and 0.91–0.99
(step 0.01) and were frozen, with the checkpoint sha256, before the test set was loaded.
A different threshold can be passed (`threshold=` / `--threshold`), but its test performance
is unmeasured; lower thresholds trade precision for recall.

**Scores are uncalibrated.** A span's `score` is the highest token probability inside it, and
the response `score` is the highest token probability in the response. They come from a
classifier trained with a positive-class weight of ≈ 17, so they are not probabilities that
the text is hallucinated; their only defined use is the comparison with the threshold.

## Serving precision (API default: fp32)

All test numbers above come from bf16 inference, batched as in evaluation. The API serves
**fp32** by default (`GG_PRECISION`), because fp32 output does not depend on batch composition
and was not slower. Measured on **dev only** (test was not re-scored), same frozen threshold 0.85,
which was chosen in bf16 (`outputs/module7/precision_dev.json`):

| | bf16 (evaluated) | fp32 (served) |
|---|---|---|
| dev All span F1 | 0.5042 | 0.5043 (Δ +0.0001) |
| dev All case F1 | 0.7885 | 0.7885 |
| dev responses whose spans differ | 82 / 895 | |
| GPU `predict()` latency, batch 1, 200 dev responses: median / p95 | 16.8 / 18.5 ms | 14.3 / 19.8 ms |

## Limitations

- **Contradictions in QA and summaries are mostly missed.** Char recall on *Evident
  Conflict* spans (seed 42, test): QA 0.082, Summary 0.120 (Data2txt: 0.735). *Subtle
  Conflict* recall is 0.077 overall (16 test spans). Summary span recall is 0.25 overall.
- **Long contexts.** Responses whose context needs > 10 DistilBERT windows (49 test
  responses) have span recall 0.214 (F1 0.307) even with the full context, against
  0.60–0.67 for shorter ones; this is not fixed by removing windowing.
- **Strong generators.** Span F1 on GPT-4 responses is 0.218 and on GPT-3.5 0.301 (seed 42),
  where only ~10% of responses contain a hallucination.
- **Domain shift.** Trained only on MS MARCO QA, CNN/DM and news summaries and Yelp records,
  in English, with 2023-era generators. Other domains (e.g. clinical, legal, code), other
  languages, other prompt formats or newer generators are unevaluated.
- **Inputs above 8,192 tokens** are split into sliding windows and scored with min over
  windows (`"windowed": true`). ModernBERT was never trained or evaluated in this mode
  (all RAGTruth inputs fit in one window); treat those outputs as unvalidated.
- **bf16 numerics.** In bf16, batch-1 results can differ from batched results near the
  threshold: in the parity check, 6.8% of test responses (182/2,675) had a different span set
  at batch 1, almost always 1–2 tokens. fp32 (the API default) is independent of batch
  composition (verified); its dev F1 equals bf16's within 0.0001, but its test score was not
  measured.
- **Span granularity is token runs, not sentences.** A span is a run of consecutive flagged
  tokens. It can cover a whole sentence (as in the demos) or a few words; it is not aligned to
  sentences or claims. On dev (fp32), predicted spans are shorter than gold spans (median 19 vs
  30 characters, mean 37.6 vs 48.0) and start/end on a sentence boundary less often than gold
  (starts 18.3% vs 26.1%, ends 25.7% vs 30.8%).
- **Span edges.** With the byte-level BPE tokenizer, a span can end on a whitespace-only token
  and so include trailing whitespace (7 of 2,784 test spans); `response[start:end] == text`
  always holds.
- **No flag is not proof of support**, and a flag is a pointer for review, not a verdict.

## How to use

```python
from groundedguard.inference.predictor import GroundedGuardPredictor
p = GroundedGuardPredictor("checkpoints/modernbert_lr5e-5_s42")      # frozen threshold 0.85
p.predict(response, question="...", context=passages)                  # QA
p.predict(response, context=document)                                  # summary-style
p.predict(response, structured={"name": "...", "hours": {...}})        # data-to-text
```

```bash
python -m groundedguard.inference.predictor --response "..." --question "..." --context_file passages.txt
bash examples/run_demos.sh            # three hand-written demos (not an evaluation)
```

HTTP API: see the README (`uvicorn api.app:app`, `POST /detect`).

## Reproducibility

Frozen choices with weight sha256: `outputs/module5b/frozen_choice.json` (ModernBERT),
`outputs/module5/frozen_choice.json` (DistilBERT). Predictor parity with the evaluated
predictions: `outputs/module6/parity.json` (batched: 100% identical spans on dev and test
for both models), tests in `tests/test_predictor.py`.
