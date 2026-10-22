# Day-1 official adaptation runbook

Times below are cumulative targets after release, not organizer deadlines.
Official materials outrank this runbook; current fact types/qualifiers are
provisional. Preserve the existing pipeline and change only the needed boundaries.

## First 30 minutes — archive and extract the contract

1. Read the complete case specification, dataset README/schema, example
   inputs/submissions, scorer source/config, runtime instructions and licenses.
   Archive originals unchanged under ignored `artifacts/official/`; record
   source URL/file, version/time, section and SHA256. Do not commit private data.
2. Fill the ingestion table below in a local `spec_ingestion.md`. Separate
   CONFIRMED wording from INFERRED engineering choices and UNKNOWN answers.
   Record conflicts and questions for organizers; never resolve them by guessing.
3. Identify the submission deadline/timezone, primary ranking rule and hard
   runtime constraints first. Reclassify affected assumptions in existing notes.

For **each row**, save exact source and exact wording, interpretation and an open
question (or “none” only after resolution). UNKNOWN is an unanswered field,
not an assumed default. Source includes file/URL + version/hash + section.

| Requirement | Exact source | Exact wording | Engineering interpretation | Open question |
| --- | --- | --- | --- | --- |
| INPUT | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| TARGET | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| FACT TYPES | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| ANNOTATION POLICY | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| METRIC | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| SPLIT | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| COMPACTNESS RULE | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| HARDWARE | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| LATENCY RULE | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| EXTERNAL DATA RULE | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| PRETRAINED MODEL RULE | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| INTERNET RULE | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| SUBMISSION FORMAT | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| DEADLINE | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

Annotation policy includes normalization, evidence, spans/relations, missing
labels, negation, temporality and speaker attribution only as actually specified.
Capture RAM/VRAM, throughput, external APIs and artifact limits when provided.

## First 2 hours — adapters, scorer and data audit

1. Implement official `TicketAdapter.to_ticket` and `SubmissionAdapter.convert`
   at the existing boundaries. Preserve raw text, exact IDs, message roles/order,
   timestamps and label presence. Internally `facts=null` differs from `facts=[]`;
   map official missing/empty semantics explicitly. Do not invent labels, fill
   absent qualifiers as annotation truth or force incompatible data into `Fact`.
2. Keep small official-format reference fixtures locally under the data policy.
   Verify load/export round-trips and official ID/schema checks before modeling.
3. Wrap the reviewed local official scorer as `(tickets, predictions) ->`
   `{name, score, provenance, higher_is_better}`; attach with
   `--scorer module:function`. Official conversion belongs inside this boundary.
   The official score becomes primary; provisional metrics remain diagnostic.
4. Manually verify reference cases using official types/values only: exact vs
   normalized values; duplicates; empty/unlabelled cases; partial matches;
   spans/offsets and relations if applicable; macro/micro/class weights; invalid
   predictions; missing/extra IDs; ranking direction. Save input, expected
   score/error, rationale and direct-scorer/wrapper results. Require agreement.
   An unspecified rule stays UNKNOWN; ask instead of inventing expected scores.
5. Run validation and bounded EDA: schema/nulls, label coverage, text/token
   lengths, languages, empty/long/malformed records and annotation conflicts.
   True token lengths need the selected local tokenizer; whitespace is not tokens.
6. Review exact/template/near-duplicate and incident/customer/family links.
   Preserve the supplied official split. If a new split is needed, document
   grouping, seed and label coverage; freeze data/split hashes before comparison.
   A bounded leakage check passing means no detected link, not independence.

## First 4 hours — establish comparable baselines

1. Run empty sanity, applicable deterministic rules and one legal compact
   prompt-only model on the same frozen dev and official scorer. Rules emit only
   official types they can actually identify; current synthetic types do not map
   automatically. Reuse the real runtime probe only if task/runtime rules fit.
2. Save official score, failures, representative raw outputs/predictions and
   10 useful errors. Inspect literal corruption, misses, invalid output and
   task-defined speaker/negation/temporal errors; lexical evidence is not truth.
3. Profile startup/load, warm p50/p95, RAM/VRAM, disk bytes and throughput on
   the target runtime. Unavailable measurements stay null with a reason.
   Save code/data/config/model hashes, seed, hardware and commands in the ledger.

## First 8 hours — choose the next measured experiment

1. Compare at most two serious task-fitting options first: generative LLM;
   encoder/span for literal fixed types; extract-then-classify/link; or rules+ML
   if fusion addresses observed errors. Interfaces already exist; do not build
   every option. Use the same dev/scorer/runtime and quality/resource evidence.
2. Choose one dominant error and one change; record Hypothesis, Change, Expected
   result and Decision rule before running. Prefer official-score gain within
   official resource limits; no invented compactness cutoff or model winner.
3. If training is justified, use [TRAINING.md](TRAINING.md): measured prompt-only
   baseline → real smoke → tiny overfit/generation checks → explicit full stage.
   Full SFT requires all data/scorer/resource gates; external transfer additionally
   needs licensing, organizer permission and an MWS-only comparison. No sweeps.
4. Preserve a known-good version now. Before delivery, replay official adapters,
   scorer and submission from a clean installed package in the allowed runtime;
   verify actual limits, format and deadline. Stop speculative pre-spec work.
