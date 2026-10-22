# Phased implementation plan

Дата: 2026-10-21. Выполнять последовательно; evidence записывать в
`docs/VERIFICATION.md`. Official-data gates могут оставаться pending, это
не разрешает подменять их synthetic quality.

| Фаза | Работа | Gate перед переходом | Artifacts |
| --- | --- | --- | --- |
| 0 | Read-only repo reconnaissance, official research, skills search/install/read | Evidence / unknowns разделены | PRE_IMPLEMENTATION_REPORT, SKILLS_AUDIT, MODEL_CANDIDATES |
| 1 | Canonical contracts, strict IO/adapters/parser/normalization/evaluator | Analytic TP/FP/FN + malformed/failure/identity tests | package, tests, schemas |
| 2 | Synthetic fixtures, deterministic baselines, EDA/leakage/group split | Annotated-vs-unlabelled distinction; frozen manifest, no detected overlap | fixture data + EDA/split outputs |
| 3 | Single/batch inference, fresh-process benchmark, run ledger | End-to-end CLI smoke, saved metrics/provenance/RAM/timing | experiments/runs.jsonl, artifacts |
| 4 | Local HF prompt-only backend, SFT features/collator + gated LoRA runner | Offline config/masking/failure tests; actual model smoke pending weights | configs, model/training code, runbook |
| 5 | Full tests/lint/build/clean-env replay, error analysis/docs | All local core checks pass; pending ML/official checks explicit | lock, CI, README, VERIFICATION |
| 6 (official data) | Read spec; replace adapters+scorer; EDA; annotation/split review | Frozen validation + official scorer reference cases | official constraints, data/split hashes |
| 7 (official data + hardware) | Prompt-only baseline, tiny overfit, SFT + focused ablations | Generation quality + resources measured on same split | ledger, best-known checkpoint |
| 8 (only if justified) | Constrained decoding, quantization, hybrid | Quality/resource deltas measured; rules permit | Pareto table, KEEP/REJECT |
| 9 | Freeze known-good submission before deadline | Clean offline target-runtime replay + official format check | version/tag, submission, model card |

## Local implementation contract

Package `compactfacts`, Python 3.12.12, src layout. Core dependencies Pydantic
and psutil; pytest/ruff dev tools; pinned ML extra separate. JSON config
validated with `extra=forbid`. No default model winner, automatic checkpoint
downloads, external tracker, remote code or quantization.

Canonical Pydantic models (`compactfacts.schemas`):

- `Message(id: str, role: str, content: str, timestamp: str|None=None)`.
- `Evidence(message_id: str, start: int, end: int, text: str)`.
- `Fact(type: str, value: str, evidence: list[Evidence]=[],
  polarity: affirmed|negated=affirmed, temporality: current|historical|unknown=unknown,
  certainty: asserted|hypothetical=asserted)`.
- `Ticket(id: str, messages: list[Message], facts: list[Fact]|None=None,
  group_id: str|None=None, metadata: dict={})`; `facts=None` unlabelled.
- `FactPayload(facts: list[Fact])`, required field, no default empty response.
- `Prediction(ticket_id: str, status: ok|error, facts: list[Fact],
  error_code: str|None=None, repaired: bool=False, generated_tokens: int|None=None)`.
  Error always visible; errors cannot carry facts.

Core utilities `compactfacts.utils.io`: strict JSON without duplicate keys or
non-finite literals; atomic private JSON/JSONL writes; SHA256. Data adapter
owns external field names. Canonical reader rejects duplicate IDs/oversize
inputs/malformed lines, does not skip records. Official submission adapter
must fail clearly until supplied; canonical diagnostic export supported.

Extractor interface: `predict(ticket: Ticket) -> Prediction`; factory
`create_extractor(config: InferenceConfig)` in `compactfacts.models`.
HF implementation imports heavyweight libraries lazily. Config stores
`backend`, `model_path`, `model_id`, immutable `revision`, `adapter_path`,
`device`, `dtype`, `max_input_tokens`, `max_new_tokens`, `seed`,
`allowed_types`, `parser_mode`, `max_retries`. Shared prompt/target format
for inference and SFT. Generated facts audited against original messages.

Evaluation API: `evaluate(gold: list[Ticket], predictions: list[Prediction],
normalization=None, official_scorer=None) -> dict`. Fact key includes type,
value, polarity, temporality, certainty; evidence excluded from fact identity
but checked separately. Duplicate predictions cannot increase recall.
Unmatched prediction != confirmed hallucination. `official_scorer` receives
canonical objects and returns a finite score with name/provenance; official
format conversion stays in its adapter. Unknown/extra IDs fail explicitly.

Data APIs: `read_tickets(path)`, `write_tickets(path,tickets)`;
`analyze(tickets, tokenizer=None)` returns aggregate report;
`split_tickets(tickets, dev_fraction, seed, near_threshold)` returns train,
dev, immutable manifest. Split joins exact/template/group/near edges using
connected components; refuses impossible splits. Near detection bounded
and explicitly reports completeness/limits. Token lengths null if no real
tokenizer supplied; character/whitespace counts never called model tokens.

Training API: pure `build_features(ticket, tokenizer, max_length)` and
collator tests without torch; opt-in real runner with smoke -> tiny overfit ->
full gates, local models, dataset/config hashes, explicit LoRA settings,
generation validation after overfit, no automatic full training on fixtures.

## Decision log

| ID | Решение | Основание / alternative |
| --- | --- | --- |
| D01 | Separate typed core from ML extra | Core reproducible without GB installs; full ML mandatory install rejected |
| D02 | Preserve values and message-local evidence | Avoid ID corruption/role loss; flatten+normalize rejected |
| D03 | Evaluator before neural training | H2 analytic tests; model-first would lack comparison |
| D04 | Local ledger + frozen split | Small 48h workflow; hosted tracker optional later |
| D05 | Strict parser default; repair explicitly measured | No silent recovery-as-success; constrained engine waits for benchmark |
| D06 | No model winner yet | Metadata is not task validation; 4 compact candidates under review |
| D07 | No autoresearch loop yet | Official data/scorer unavailable; do not optimize synthetic surrogate |
| D08 | Keep failed retries and scorer attribution observable | Adversarial evaluator regression cases |
| D09 | Protect IO boundaries and ledger history | Overwrite/newline defects reproduced before fix |
| D10 | Noneditable install + wheel replay | Host hides editable .pth; installed import verified |
| D11 | Explicit preparation + content-based model gates | Lazy GPU timing and same-size weight identity defects |
| D12 | Require official score direction | Unknown objective cannot be assumed maximized |
| D13 | Frozen public auxiliary suite, separate from MWS/synthetic tests | NLU++ pinned CC-BY-4.0 slots; <=256KiB per source file; exact spans and attribution, no relabelling or automatic training mix. Acceptance: source/hash/span checks and reproducible conversion |
| D14 | Add exact span bridge and occurrence-aware diagnostic | Generative output is not assumed superior; span detector must copy source values. Baseline: canonical set scorer ignores occurrences. Acceptance: nested/repeated-value/Unicode/out-of-bounds tests; no learned-model claim |
| D15 | Four-field experiment contract | New runs require Hypothesis/Change/Expected result/Decision rule. Acceptance: CLI/ledger reject missing fields; previous records remain append-only |
| D16 | Final installed replay and pre-compute span gate | Hypothesis: rejecting contextual evidence before profiling avoids invalid span experiments. Baseline: clean wheel replay at 1e3b83b,278 tests, first Linux CI passed at 5eadc08. Acceptance: forbidden-benchmark regression, installed full replay in CI, reviewed synthetic-only snapshots and final-head checks |
| D17 | Validate one real compact runtime before further preparation | Hypothesis: actual pretrained inference and a minimal LoRA update expose integration risks that doubles cannot. Baseline: e58f94f,279 core tests, real ML not run. Change: install existing locked ML extra, select one immutable native safetensors model after bounded metadata check, existing HF/benchmark/training paths. Expected: real generation, measured resources, finite update and adapter save/reload. KEEP mechanics only on measured evidence; record failed attempts, no full SFT/sweeps/quantization. User explicitly authorized model/dependency acquisition in current master §§9–12. |
| D19 | Operational Day-1 readiness and current backlog | Hypothesis: a short timed ingestion/scorer/baseline runbook reduces adaptation delay. Baseline: existing adaptation checklist and stale GitHub authorization backlog. Acceptance: first30m/2h/4h/8h actions, exact-source requirement capture, current P0 real mechanics and explicit official-data gates; no new framework. |
| D18 | Bounded telecom/cloud/support prior-art update under infrastructure freeze | Hypothesis: 2–3 new primary sources can improve future error analysis or architecture choices without delaying the single real-ML vertical slice. Baseline: existing analogue map is mainly Russian IE/news and banking; no measured MWS transfer. Acceptance: at most 3 actionable primary sources, each classified as PUBLIC DATASET/CODE/INDUSTRIAL LESSON/MARKETING with reported evidence vs unknown, concrete future test and no data/model download, training or new framework. Result: 3 INDUSTRIAL LESSON sources in PRIOR_ART addendum/source index; git diff --check passed; no acquisition or experiment in this documentation task |

При крупных новых изменениях добавлять решение, baseline и acceptance
до изменения; после — command, result и limitation в VERIFICATION/ledger.
