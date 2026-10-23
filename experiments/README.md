# Experiment ledger

`runs.jsonl` is append-only. Every CLI `run` and every training stage that starts
and writes a manifest records a hypothesis, code revision/dirty flag, package
source hash, data hash, config/model revision/seed, exact command, metrics,
resources or explicit nulls, artifact paths, conclusion and KEEP/REJECT reason.
New runs require four explicit nonempty fields: Hypothesis, Change, Expected result,
Decision rule. Identifiers are unique; existing runs are never replaced. Preflight failures
before an experiment starts return nonzero and do not claim a training run.

`KEEP` for a software baseline or training mechanics gate means retain the
reference artifact. It does not mean improved official task quality.
Synthetic results must never be interpreted as hidden-test predictions.

Large or private artifacts remain ignored under `artifacts/`. Small, explicitly
synthetic preparation results are checked into `experiments/baselines/` for
review. The ledger includes their recorded paths and actual measurements.
`published_artifact` points to the committed review copy. Original artifact
paths, source revision and measured values are preserved as run provenance.
Real support predictions/evidence must stay out of Git and shared logs.

`runtime_probe_20261022/` preserves reviewed synthetic-derived real-model
outputs, measured failed baseline, resource observations and LoRA mechanics.
Its manifest maps immutable ledger IDs to published copies without rewriting
original paths/dirty flags. Weights/checkpoints remain ignored. Memorizing two
training targets and valid JSON syntax do not establish schema/MWS quality.

Freeze a split before model comparison. Use identical data hashes, prompt,
decoding, batch policy, scorer and hardware. Hypotheses change one substantial
variable at a time; retries/fence repair are separately counted. Official
score replaces provisional micro F1 as primary when a reviewed scorer is supplied.

Autoresearch is not active: official data/metric and frozen real validation
are unavailable. The measured runtime probe does not supply an MWS objective.
Before an eventual loop, define runtime budget and fitness
against the official scorer; preserve rejected attempts with safe Git reverts,
not destructive resets of user work.
