# Current priorities

Infrastructure freeze: reuse the existing pipeline. New infrastructure/tests
need a measured runtime defect, an exposed high-risk boundary or official contract.

## Confirmed preparation

- Public repository: [BratishkaDurovaTg/mws-for-mts-hackaton](https://github.com/BratishkaDurovaTg/mws-for-mts-hackaton), branch `work/preparation`.
- Core preparation, saved synthetic baselines and independent wheel replay are
  recorded in `VERIFICATION.md`; CI passed for preparation head `e58f94f`.
  GitHub authorization is resolved. Later changes require their own checks.
- Pinned NLU++ auxiliary slice and exact span integrity machinery exist; they
  establish software behavior, not MWS quality or a trained span extractor.

## Active pre-spec work

| Priority / status | Work | Acceptance / stop rule |
| --- | --- | --- |
| P0 IN PROGRESS | One real compact-model inference through existing HF → parser/schema/evidence → evaluator path | Real local immutable weights/tokenizer; representative raw outputs and failures saved; no final-model claim |
| P0 IN PROGRESS | Real one-model profiling beside empty/rules | Saved startup/load, p50/p95, RAM, disk bytes, tokens/sec and reliable device memory or explicit null; synthetic metrics marked NOT MWS QUALITY |
| P0 IN PROGRESS | One real LoRA smoke; optional cheap tiny overfit | Finite update/gradients, adapter save/reload and real generation; overfit only for mechanics, investigate failure before escalation |
| P0 READY, official execution pending | Day-1 schema/scorer adaptation | `OFFICIAL_ADAPTATION.md`: first30m/2h/4h/8h, exact-source capture, scorer references and frozen validation |
| P1 BOUNDED | Relevant industrial prior art: MTS/MWS, telecom/cloud/support first | At most 3 actionable primary sources this iteration; classify data/code/lesson/marketing; stop when no engineering choice changes |
| P1 PRESERVE OPTIONS | Generative, encoder/span, extract-then-classify/link, rules+ML | Keep comparison plan; build only the task-fitting options after official data/scorer; no parallel framework now |
| P1 AS EXPOSED | Fix real runtime defects | Small measured correction plus targeted regression only where justified |
| P2 OPTIONAL | Small Russian public evaluation slice or diagnostic cases | Only if it answers a concrete gap; license/provenance/gold review; no automatic training mix |
| P2 OPTIONAL | Second runtime model | Only if the first exposes unresolved architecture/runtime uncertainty |

After real load/inference/profiling/LoRA smoke (and optional tiny overfit), stop
generic pre-spec optimization. The runtime probe is not the final MWS model.

## Blocked until official task and validation gates

| Work | Required evidence |
| --- | --- |
| Official adapters/scorer/schema/submission | Original materials, explicit rules and manually checked scorer references |
| Final model/architecture selection; serious prompt tuning | Frozen real dev, official objective, legal candidates and target-runtime measurements |
| Full SFT | Real prompt-only baseline, smoke/tiny-overfit gates, official data/split/scorer and compute budget |
| RuREBus/NEREL/other transfer or multitask training | License + organizer permission + task compatibility; beat MWS-only on the same dev or REJECT |
| Quantization/QLoRA optimization | Working floating-point baseline, permitted target runtime and measured quality/memory/latency delta |
| Constrained decoding/repair tournaments | Observed parse failure bottleneck and official quality/resource benefit |
| Ensembles, distillation, large synthetic generation | Concrete official-task bottleneck and bounded comparative experiment |
| Autoresearch | Frozen real validation, official metric, measured baseline and bounded experiment cost |

Rejected: blind external-data concatenation, unknown-license vendoring, intent
labels treated as fact gold, expanding a guessed MWS ontology, generic scaffolding
or broad model/dataset sweeps. No job or download starts merely from this backlog.
