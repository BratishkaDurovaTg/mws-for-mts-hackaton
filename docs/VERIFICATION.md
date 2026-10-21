# Verification record

Дата: 2026-10-21. Scope: подготовительный software pipeline без official
данных и pretrained weights. Это не production model validation.

## Phase gates

| Gate | Evidence | Result |
| --- | --- | --- |
| Reconnaissance | Empty repo; report/plan committed before Python | PASS |
| Skills | 15 searches,253unique candidates;8installed,52files/555999bytes blob verified | PASS; HF warning documented |
| Contracts/evaluator | Analytic TP/FP/FN, exact IDs, invalid/empty/unlabelled/duplicate/retry/scorer cases | PASS |
| Data/EDA/split | 20hand-labelled synthetic tickets,28facts; templatefamily retained | PASS for detected links; bounded near scan incomplete on long record |
| Baseline | Empty/rules previews: F1=0 /0.590909 | Synthetic mechanics only |
| Installed package | uv sync --locked --no-editable; separate call imports site-packages | PASS |
| Tests | .venv/bin/pytest -q without PYTHONPATH:240passed | PASS |
| Lint/format | Ruff check/format src tests | PASS |
| Wheel | uv build --wheel | PASS; independent replay recorded below |
| HF/SFT contracts | Fake tokenizer/model tests for local load, template/masking/EOS, budgets/retries, split/gates/hashes | PASS contracts only |
| Real LLM baseline/tiny-overfit/SFT | No weights or ML extra installed | NOT RUN |
| Official scorer/submission/runtime | Materials not received | PENDING spec/data/hardware |

## Measured defects corrected during review

| Finding | Before | Correction / regression evidence |
| --- | --- | --- |
| Official attribution | Blank name/provenance became primary | Reject blank fields |
| Failed retries | attempts3,error produced retry rate0 | Count retries before success/error branch |
| Raw input overwrite | Output ancestor could replace train.jsonl | Equal/ancestor and normalization/ledger collision guards |
| Ledger newline | Appending to incomplete final line corrupted JSONL | Fail before mutation; previous bytes preserved |
| Lazy HF timing | Constructor clock omitted load; GPU probe too early | Public prepare before loadclock/CUDA probe; fake GPU test |
| Wheel provenance | Module parent outside checkout looked clean | Invocation checkout, nullable status, package source hash |
| Training weight identity | Same-size content change passed metadata identity | Full local weight hashes in explicit runs/stage fingerprint |
| Failed baseline | Finite0score alone allowed full training | Require measured baseline with zero technical failures |
| Official direction | Comparison assumed maximize | Explicit higher_is_better for official ranking |

These are software corrections, not measured model-quality improvements.

## Interpretation and limits

All preparation scores use provisional raw exact matching on synthetic data.
Fact identity includes type/value/polarity/temporality/certainty; evidence is
audited separately. Lexical support cannot prove semantic truth, cause,
speaker attribution or temporality. Rules remain a limited mention baseline.

This fixture has190possible pairs; all are screened, but one long record
exceeds the sequence-comparison budget. Inspect completeness/limits. Zero
detected cross-split links does not prove independence. Split:15train/5dev,
18connected components. Full-fixture score is a regression check, not held-out
task evaluation.

Cold time excludes process spawn, includes worker imports/read/load/first
prediction. Warm metrics are serial and environment-specific. RAM is fresh
process OS high-water including interpreter/imports. Rules have zero model
weights; no LLM compactness conclusion follows. Inapplicable tokens/VRAM=null.
CUDA paths are contract-tested, not measured on CUDA hardware.

Initial source checks used PYTHONPATH because this host reapplied UF_HIDDEN to
editable .pth files, which CPython ignores. Delivered commands use installed
noneditable packages, verified across separate calls. No manual PYTHONPATH or
hidden cache state is required by that workflow.

## Completion boundary

Local preparation is deliverable after clean replay/artifact checks. The ML
product requires official schema/scorer/data/runtime and actual LLM baseline,
training/generation and resource validation. No weights were downloaded, no
real support data sent to services, no paid/long GPU jobs or autoresearch run.

## Final replay and saved artifacts

Updated after final clean-environment and committed-baseline replay. Measured
artifacts belong in experiments/baselines and append-only experiments/runs.jsonl.
