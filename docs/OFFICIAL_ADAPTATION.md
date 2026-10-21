# First hours after case release

1. Read the entire official spec and dataset/evaluator licenses. Record exact
   versions/URLs/hashes. Reclassify every affected assumption in the report;
   lower-priority notes cannot override the official material.
2. Add an explicit `TicketAdapter.to_ticket` and `SubmissionAdapter.convert`.
   Preserve roles/order/timestamps and absent labels. Store official field names
   only at the adapter boundary. Golden official-format fixtures are mandatory.
3. Wrap the official scorer as a local callable `(tickets, predictions) ->
   {name, score, provenance, higher_is_better}`. Adapt canonical facts to official values/schema
   inside that boundary. Compare wrapper outputs to the supplied scorer on
   valid/invalid/empty/duplicate/identifier cases. No semantic normalization
   unless permitted. Attach via CLI `--scorer module:function`.
4. Run bounded read-only EDA and inspect schema/nulls/classes/languages/token
   lengths and conflicting annotations. Core `eda` reports true token lengths
   only when a tokenizer is explicitly supplied through the Python API; default
   whitespace lengths are not model-token estimates.
5. Review observed incident/customer/template/near-duplicate families. Preserve
   the official split if one is provided; do not replace it silently. Use
   grouped components for a new split only with a documented rationale. Core
   split screening is heuristic and bounded; `passed` means no detected
   cross-split link, not proof of independence. Stratification is not currently
   implemented; inspect per-label coverage and adjust with official policy.
6. Pin the dataset and split manifests. Run A0, A and at least one compact B
   with the same evaluator. Download weights only after budget/license/runtime
   checks. Measure a floating-point baseline on the actual target hardware.
7. Run local training smoke, tiny overfit with response-mask/EOS checks and
   generated fact-set checks, then explicitly request a full SFT stage.
   The full stage requires the frozen prompt-only baseline and passed gates.
   `eval_loss` is diagnostic, not the competition objective.
8. Error analysis drives the next single-variable experiment. Semantic wrong
   speaker/obsolete/cause errors need human annotation review; lexical evidence
   checks cannot establish truth. Compare models on quality/resource Pareto
   frontier; no made-up compactness cutoff.
9. Consider constrained decoding, repair/retry, hybrid or quantization only
   where current errors and official rules justify it. Record quality/memory/
   latency deltas; do not assume quantization is faster.
10. Freeze a known-good version before the last hours. Build the allowed
    offline/container artifact, install from lock in a fresh target runtime,
    replay official adapters/scorer, verify submission and size/latency limits.

## Explicit extension points

See `compactfacts.data.adapters`, `compactfacts.evaluation.evaluate`,
`compactfacts.models.Extractor` and typed configs. Python APIs do not require
the official schema to look like our synthetic representation. A reviewed
local plugin is executable code: load only the scorer/adapter you intend.
No official implementation or submission is claimed before these steps pass.
