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
| Tests | uv run --locked --no-editable pytest -q without PYTHONPATH:279passed | PASS |
| Lint/format | Ruff check/format src tests scripts:47files | PASS |
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

Independent core-only wheel installation completed with locked hashed
dependencies in `artifacts/final-wheel-env`. Its interpreter, without source
path injection, executed:

```bash
artifacts/final-wheel-env/bin/python scripts/reproduce_preparation.py --output artifacts/preparation-20261022-v2 --id-prefix prep-20261022 --publish-synthetic
```

The clean source commit was `1e3b83b66f1b840ff0d22d7caf502a8e07539fd5`,
package source SHA256 `179f4ee68e38f4f5c0fcdab3fdf3f1bc60ca72ccf351194dc4995989c6f1ec31`,
fixture SHA256 `27fdf9619af33ff3763368c937711f2f94d3fb93937d2fdca9fff8bc2fbe1840`.
Both run records report `git_dirty=false`. Python3.12.12, macOS26.0.1 arm64;
seed42; batch1 serial,10passes after1warmup,200timed requests per backend.

| Backend | Provisional full-fixture F1 | TP/FP/FN | Warm p50 / p95 (ms) | Process peak RAM (MiB) | Technical failures |
| --- | --- | --- | --- | --- | --- |
| Empty | 0 | 0/0/28 | 0.000750 / 0.000792 | 33.75 | 0/200 |
| Rules | 0.590909 | 13/3/15 | 0.011521 / 0.081110 | 33.90625 | 0/200 |

These are software regression references on all20synthetic records, not
held-out model quality. Generated split remains15train/5dev. Weight bytes=0;
tokens/VRAM=null. Timing is a single host observation with clock/scheduling
noise, not a universal performance claim. Reviewed copies are in
`experiments/baselines/`; the two append-only records in
`experiments/runs.jsonl` preserve original paths and add `published_artifact`.

The public slot suite also ran from this wheel:10records/9gold spans,
zero technical failures. Annotation-to-span round-trip integrity=1.0 is a
converter control, not a model score. Generic technical rules scored0 on
the foreign slot ontology; no MWS-quality inference follows. Public texts,
predictions and detailed public metrics remain ignored under `artifacts/`.

GitHub authorization with `workflow` scope succeeded. The initial Linux
[CI run at 5eadc08](https://github.com/BratishkaDurovaTg/mws-for-mts-hackaton/actions/runs/39848423438)
passed. Subsequent pushes run the full installed preparation replay as well;
their status must be read from the corresponding commit checks.

## Addendum2026-10-22

- Independent fresh wheel environment replay completed on20synthetic tickets:
  validate/infer/evaluate/benchmark; zero technical failures; F1=.590909.
- RELATED_TASKS/PRIOR_ART cover T-Bank, RuREBus, NEREL/RuNNE, FactRuEval,
  conversational slots; rankings are track/version-specific. Research stopped
  at actionable architecture/transfer hypotheses, not a model winner.
- NLU++ pinned public slice:10records/9slots,CC-BY4.0,23,986byte source;
  online/offline conversion bytes match. Source texts remain ignored. RU public
  coverage, relations, negation and long-context public gold remain pending.
- Exact span-copy bridge and occurrence-aware diagnostic are tested; this is
  not a trained span classifier. Four explicit experiment fields are checked
  before computation and persisted in the ledger.
- Replay caught an invalid combination: contextual fact evidence is not
  exact entity-span gold. Synthetic fact baselines use the fact scorer;
  opt-in exact span scoring validates gold before computation and is used
  for public slot spans. No arbitrary first-occurrence span is inferred.
- A regression test forbids entering the benchmark when span gold is invalid;
  final local checks:279passed, Ruff check/format passed, no PYTHONPATH.
- Five candidate checkpoints include one small encoder that still needs a
  trained extraction head. No model weights/large datasets/training jobs run.
