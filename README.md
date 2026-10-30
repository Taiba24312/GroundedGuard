# GroundedGuard

Span-level hallucination detection for retrieval-augmented generation (RAG).

<!-- Every number outside code blocks is verified by scripts/check_readme_numbers.py against the
     source files listed in the nearest "sources:" comment above it. -->

## Problem

A RAG system answers from retrieved context, but the answer can still contain claims that the
context does not support: invented facts, changed numbers, contradictions. GroundedGuard takes
the context and the response and returns the **character spans of the response that are not
supported**, plus a response-level flag, so a reviewer can see *where* an answer goes wrong.

## Approach

A token classifier reads `[CLS] context [SEP] response [SEP]` and scores every response
token; a token is flagged when its score reaches a threshold frozen on dev, and consecutive
flagged tokens form one span. It is trained on RAGTruth (QA, news summarization,
data-to-text), with three context formats: question + passages, a document, or a JSON record.

```text
DistilBERT (fast option)                         ModernBERT (default)
512-token limit: the context is split            8,192-token limit: every RAGTruth input
into overlapping windows, the response           fits in one window
is repeated in each
                                                 [CLS] full context [SEP] response [SEP]
[CLS] ctx window 1 [SEP] response [SEP] -> s1             |
[CLS] ctx window 2 [SEP] response [SEP] -> s2      per-token score -> threshold -> spans
[CLS] ctx window 3 [SEP] response [SEP] -> s3
                         |
   per-token MIN over windows: a token is supported if ANY window supports it
   (the loss is computed on the MIN, so training matches inference)
                         |
             per-token score -> threshold -> spans
```

## Results

<!-- sources: outputs/module5/test_results.json, outputs/module5b/test_results.json, outputs/module3/baselines.json, references/ragtruth_paper_tables.md, outputs/module2/windowing_report.json -->
RAGTruth test set, official metric (character-level span overlap; a response counts as
flagged when it has at least one span), in %. Our models: mean ± standard deviation over
3 seeds, each scored once with a checkpoint and threshold frozen on dev beforehand.

| detector | response F1 | span P | span R | **span F1** | span F1 QA | span F1 Summary | span F1 Data2txt |
|---|---|---|---|---|---|---|---|
| B1: flag the whole response | 52.1 | 4.1 | 100.0 | 7.8 | 10.2 | 5.7 | 7.7 |
| B2: lexical overlap | 56.3 | 7.5 | 22.6 | 11.3 | 18.7 | 9.4 | 9.1 |
| DistilBERT, 512-token windows + min (ours) | 71.4 ± 0.8 | 45.4 ± 1.4 | 41.3 ± 0.2 | 43.3 ± 0.6 | 50.2 ± 1.2 | 23.2 ± 0.6 | 44.8 ± 0.6 |
| **ModernBERT, full context (ours)** | 75.9 ± 1.2 | 54.5 ± 4.1 | 54.3 ± 5.3 | **54.1 ± 0.5** | 61.1 ± 0.4 | 33.7 ± 3.5 | 55.5 ± 0.9 |
| Fine-tuned Llama-2-13B (paper) | 78.7 | 55.6 | 50.2 | 52.7 | 58.2 | 38.8 | 53.5 |
| Prompted GPT-4-turbo (paper) | 63.4 | 18.4 | 60.9 | 28.3 | 32.6 | 24.1 | 28.2 |

Paper rows: Niu et al., *RAGTruth: A Hallucination Corpus for Developing Trustworthy
Retrieval-Augmented Language Models*, ACL 2024, arXiv:2401.00396v2, Tables 5 and 6
(values checked against the PDF; see `references/ragtruth_paper_tables.md`). They were not
re-run here, and the paper's detector is a 13B-parameter LLM. Full per-task precision and
recall, and the comparison details, are in [MODEL_CARD.md](MODEL_CARD.md).

## Key findings

<!-- sources: outputs/module5b/error_analysis_tables.json -->
- **Contradictions in QA and summaries are mostly missed.** For ModernBERT (seed 42, test),
  character recall on *Evident Conflict* spans is 8.2% in QA and 12.0% in summaries, against
  73.5% in data-to-text and 73.6% on *Evident Baseless Info* in QA. *Subtle Conflict* (16 test
  spans) has 7.7% recall. The model finds added information far more reliably than
  contradicted information in free text.

<!-- sources: outputs/module5b/comparison.json -->
- **Most of the ModernBERT gain comes from the encoder, not from seeing more context.** On the
  807 test responses that fit in a single DistilBERT window (no windowing at all), span F1 still
  rises from 0.429 to 0.564. For the 49 responses with the longest contexts (more than 10
  DistilBERT windows), full context does not help: recall is 0.226 with windows and 0.214
  without. Long contexts are hard because of the task, not because of windowing.

<!-- sources: outputs/module5b/error_analysis_tables.json -->
- **Span F1 tracks the base rate of hallucinations.** Only 9.3% of GPT-4 responses and 10.4%
  of GPT-3.5 responses contain a hallucination, and span F1 on them is 0.218 and 0.301; on
  Mistral-7B responses (57.6% hallucinated) it is 0.624. With few true spans, each false
  positive weighs heavily, so per-generator scores are not comparable as model quality.

## Limitations

<!-- sources: outputs/module5b/test_results.json, outputs/module5b/comparison.json, outputs/module7/precision_dev.json, outputs/module2/windowing_report_modernbert.json -->
- Summaries: span recall is 25.2 ± 4.5%; conflicts in QA and summaries are mostly missed (above).
- Responses with very long contexts (more than 10 DistilBERT windows) have span recall 0.214.
- Trained and evaluated only on RAGTruth: English, MS MARCO QA, CNN/DM and news summaries,
  Yelp records, and the six generators of the RAGTruth release. Other domains (clinical, legal, code), languages and
  generators are unevaluated. A response without spans is not verified as correct.
- Inputs above 8,192 tokens fall back to sliding windows; that mode is unevaluated for
  ModernBERT.
- Scores are uncalibrated (maximum token probability); only the comparison with the threshold
  is meaningful.
- Spans are runs of flagged tokens, not sentences: on dev, predicted spans have a median of 19
  characters against 30 for gold spans.
- The API serves fp32; the test results above are bf16. On dev the two differ by 0.0001 span F1.

## API quickstart

```bash
conda activate groundedguard
CUDA_VISIBLE_DEVICES=0 uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1
# docs with an example per task: http://localhost:8000/docs

curl -s -X POST localhost:8000/detect -H "Content-Type: application/json" -d '{
  "task": "summary",
  "document": "The city council voted 7-2 on Tuesday to extend the bike lane on Main Street by two miles.",
  "response": "The council unanimously approved the bike lane extension."}'
```

Request bodies, selected by `task`:

```json
{"task": "qa",       "question": "...", "passages": "...", "response": "..."}
{"task": "summary",  "document": "...",                    "response": "..."}
{"task": "data2txt", "data": {"name": "...", "hours": {}},  "response": "..."}
```

The response gives `has_hallucination`, `score` (uncalibrated; compare with `threshold`),
`threshold`, `spans` (`start`/`end` are character offsets into the response exactly as sent,
`response[start:end] == text`), `windowed`, `n_windows`, `model`, `latency_ms` and
`request_id`. `POST /detect/batch` takes a JSON array of such requests; `GET /health` reports
the loaded model.

<!-- sources: outputs/module7/load_test.json, outputs/module6/latency.json -->
Measured on one RTX PRO 6000 (ModernBERT, fp32, one worker): 200 sequential requests, p50
16.5 ms and 56.8 requests/s; 4 concurrent clients, 64.0 requests/s with 0 errors. GPU calls
are served one at a time.

| variable | default | |
|---|---|---|
| `GG_MODEL` | `modernbert` | or `distilbert` (faster, less memory); see `configs/inference.yaml` |
| `GG_PRECISION` | `fp32` | or `bf16` (GPU only) |
| `GG_DEVICE` | `auto` | `cuda`, `cuda:N` or `cpu` |
| `GG_MAX_RESPONSE_CHARS` | `10000` | longer responses get `422` |
| `GG_MAX_CONTEXT_CHARS` | `60000` | limit on the serialized context |
| `GG_MAX_BATCH` | `16` | requests per `/detect/batch` call |
| `GG_LOG_CONTENT` | `false` | log request text (off: logs hold ids, lengths, latency, status) |

Errors: `422` (schema, empty response, input over a limit), `503` (model not loaded), `500`
without a stack trace; every response carries `X-Request-ID`. Python and CLI use, and an untested
Dockerfile, are described in [MODEL_CARD.md](MODEL_CARD.md) and the `Dockerfile` header.

## Reproduce

```bash
conda activate groundedguard        # or: pip install -r requirements-lock.txt
bash scripts/reproduce.sh --dry-run # checks every module, flag, config and script; runs nothing
bash scripts/reproduce.sh           # download -> preprocess -> windows -> baselines -> training
                                    # -> freeze on dev -> test once -> error analysis -> comparison
```

Each step in `scripts/reproduce.sh` is commented with its runtime and GPU needs. A full run
needs a fresh checkout (the frozen files and checkpoints are write-once) and repeats GPU
training, which is not bit-exact across runs.

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest      # predictor parity + API tests (needs the checkpoints)
python -m pytest -m "not slow" tests/test_api.py
python scripts/check_readme_numbers.py      # verifies every number in this README
```

## Project structure

| path | contents |
|---|---|
| `groundedguard/data/` | download, exploration, preprocessing, context serialization, windowing |
| `groundedguard/models/` | dataset and batching, window aggregation, training, dev diagnostics |
| `groundedguard/eval/` | official metric, pre-registration on dev, one-time test scoring, model comparison |
| `groundedguard/baselines/`, `groundedguard/analysis/` | heuristic baselines, error analysis |
| `groundedguard/inference/` | `GroundedGuardPredictor` (Python + CLI), latency measurement |
| `api/` | FastAPI service |
| `configs/` | training configs, served checkpoints (`inference.yaml`) |
| `scripts/` | reproduction, README number check, precision check, load test |
| `tests/` | predictor parity and unit tests, API tests |
| `examples/` | hand-written demos (not an evaluation) |
| `references/` | the RAGTruth paper's result tables |
| `outputs/` | reports from every step (numbers only: per-response predictions, large logs and reports that quote dataset text are not committed) |

## License and data

The code in this repository is released under the MIT License (see `LICENSE`).

RAGTruth is released by its authors under the MIT License
(https://github.com/ParticleMedia/RAGTruth). It is built from MS MARCO, CNN/Daily Mail, news
articles and the Yelp Open Dataset, which have their own terms of use. The dataset is not
redistributed here: `data/` is excluded from the repository, and `scripts/reproduce.sh`
downloads it from the official source. Model checkpoints are not included either.
