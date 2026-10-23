# Verification record

Initial preparation2026-10-21; real-runtime update2026-10-22. Official data
is unavailable. The real-model checks below establish mechanics, not production quality.

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
| Lint/format | Ruff check/format src tests scripts | PASS |
| Wheel | uv build --wheel | PASS; independent replay recorded below |
| HF/SFT contracts | Fake tokenizer/model tests for local load, template/masking/EOS, budgets/retries, split/gates/hashes | PASS contracts only |
| Real LLM execution/profile | Qwen0.5B,BF16,MPS;6real responses scored | PASS execution;0/6schema-valid, extraction baseline REJECT |
| Real LoRA smoke/tiny/reload | 2updates;100-step tiny;2/2train targets exact after reload | PASS mechanics only;no full SFT |
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
training/generation and resource validation. D17 explicitly authorized one
small local checkpoint and bounded mechanical training. No real support data
was sent to services; full SFT, paid jobs and autoresearch remain unstarted.

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

## Real-runtime iteration2026-10-22 — D17–D19

Current user master freezes infrastructure and prioritizes one real ML slice;
earlier no-weights status above is historical. Locked ML extra installed
unchanged: torch2.14.1/transformers5.20.2/peft0.21.2/accelerate1.15.0/
safetensors0.8.0. Hardware measured: AppleM4,16GiB unified RAM, MPS available,
CUDA unavailable. No external inference API or auxiliary training is used.

Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`
downloaded to ignored `models/`;988097824weight bytes, SHA256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
matches Hub LFS digest. Native local-only safetensors load succeeded in BF16
on MPS:494032768parameters,11.685s including initial imports,1006776576MPS
allocated bytes after load (snapshot, not peak). Acquisition/hardware/token
length evidence lives in `artifacts/real-ml/`. This preflight checkpoint
preceded the measured inference/PEFT results below.

Before generation, selected six existing synthetic cases (empty, HTTP code,
UUID, negation, version/history, two-message correction context). Fixed seed42,
512new-token limit, strict parser, zero retries. This is a mechanics diagnostic,
not an independent validation set or expected MWS quality. All20existing fixture
SFT sequences fit2602tokens; no silent truncation or new guessed ontology.

Day-1 runbook now has timed30m/2h/4h/8h actions and exact-source requirement
capture. Backlog is current. D18 adds three bounded industrial lessons, without
data acquisition or external-data training. The next empirical gate is real
inference/profiling followed by one finite LoRA smoke and adapter reload;
these gates are now completed below.

## Real-runtime results2026-10-22

Reviewed synthetic-derived copies and file hashes:
`experiments/runtime_probe_20261022/manifest.json`. Five new ledger runs preserve
all original commits/dirty flags/commands/hashes; prior history is unchanged.
Raw traces were inspected against the committed fixtures. No model weights,
optimizer checkpoints, public-corpus text or real customer text is published.

Same six diagnostic rows,one timed pass,no warmup; ordinary empty/rules execute
on CPU, Qwen uses MPS. This is a resource/reference table, not a Pareto ranking:
the failed Qwen extractor is ineligible for quality/resource ranking.

| Backend | Fixture F1 | Schema-valid responses | p50 / p95 seconds | OS process peak RAM bytes | Weight bytes |
| --- | --- | --- | --- | --- | --- |
| Empty | 0 | 6/6 | .00000104 / .00000175 | 34766848 | 0 |
| Rules | .666667 | 6/6 | .00002098 / .00031155 | 35078144 | 0 |
| Qwen0.5B strict | 0 | 0/6 | 2.01304 / 3.22251 | 557121536 | 988097824 |

Qwen cold startup=7.16847s (excludes process spawn),load=4.29327s,
generated tokens/sec=64.1475,accepted goodput=0. Six latency samples are a
small mechanical observation, not a stable service percentile. All6raw
outputs were valid JSON syntax; all6failed the schema. `parse_failure_rate=1`
includes schema rejection. FP=0/unsupported-evidence-rate=0 after rejection
does **not** establish absence of hallucination; semantic hallucination rate=null.
Observed: invalid evidence fields/nesting, invented facts on empty input,
unsupported404explanation, invalid qualifiers and history confusion. UUID text
was copied literally once, but no fact passed the extraction contract.

Real LoRA smoke:540672trainable parameters,2optimizer updates,finite gradients;
loss .60619831→.53996742;16.3464s for evaluate/train/generate/save. Adapter
weight file=2175168bytes; tokenizer+metadata make the directory13606698bytes.
Separate-process smoke adapter reload generated6real responses; schema failures
remained. KEEP only the mechanical gate, not an extraction improvement.

One optional tiny stage used the same fresh base/recipe and frozen train split:
100steps,218.987s,loss .60619831→.000911929,finite gradients,2/2training fact
sets exact. Separate-process reload identifies `PeftModelForCausalLM` with
adapter `default`; both real generations pass parser/schema/evidence audit.
These are memorized `syn-action`/`syn-assumption`, not held-out quality.

MPS allocation snapshots≈1.01GB and driver snapshots are saved in raw traces;
they are not peaks. CUDA/VRAM=null. OS RSS does not capture all MPS/unified
memory; do not add overlapping measurements or infer target-hardware fitness.

Locked dependencies needed no changes; no core runtime defect or new unit test
was required. Current pre-spec P0 mechanics and Day-1 readiness are complete.
Stop generic optimization. Full SFT,transfer,quantization,model tournaments and
autoresearch wait for official data/scorer/rules. Highest-value next step:
ingest official materials with `OFFICIAL_ADAPTATION.md` when provided.

Final local checks after the runtime iteration:279tests passed with installed
ML extra, Ruff check/format passed (48Python files), wheel rebuilt. A new,
independent core-only wheel environment validated/inferred/evaluated the six
published diagnostic rows with zero technical failures (rules F1=.666667,
software reference only). No source path injection or model weights required
for this package check. Publication file hashes and unchanged ledger prefix
were verified before commit; corresponding CI is checked on the pushed head.

Publication finalization2026-10-23:32published-file hashes and5run-copy paths
verified; all8published inputs equal committed synthetic fixtures;7unique
ledger IDs retain their four-field contracts and unchanged historical prefix.
No additional model training or speculative preparation was started.
