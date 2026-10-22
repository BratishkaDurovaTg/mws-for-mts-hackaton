# Prior art and falsifiable transfer plan

Checked2026-10-22. All MWS gains are UNKNOWN. No external training, copying
participant code or model selection has occurred.

| Idea | Primary source / reported result | Why it may transfer | Risk | Experiment required | Result / decision |
| --- | --- | --- | --- | --- | --- |
| Mark entity endpoints+pool spans; sample negatives | [RuREBus RE winner R-BERT](https://dialogue-conf.org/media/5102/korzunva-153.pdf):~.44 vs.2066 BERT+NER embeddings |Explicit relation endpoints and exact values |Pair explosion, FP cascade |Extract-then-link vs generation; predicted/oracle entity tracks separately |MWS untested;CANDIDATE |
| Separate NER and RE losses if joint loss harms | [RENERSANS](https://dialogue-conf.org/media/5082/davletovaaplusetal-148.pdf):added NER loss hurt gold-entity RE;final coefficient0 |Direct warning against assumed multitask gains |Foreign ontology/negative transfer |Same budget/split, separate vs joint loss |Prior negative result;do not enable automatically |
| Exact BRAT↔token alignment |Same RENERSANS paper reports~1%test NER tag loss in conversion |Protect IDs/evidence and nested spans |Silent truncation/subtoken/offset shifts |Round-trip all source spans;Unicode/nesting/repeated values tests |Exact-copy bridge tested;no trained-model result |
| Schema-conditioned MRC span questions |[RuNNE MRC](https://dialogue-conf.org/media/5788/rozhkovisplusloukachevitchnv107.pdf),[author code](https://github.com/fulstock/mrc_nested_ner_ru) |Definitions may help rare types |Per-type latency;question bias |Label IDs vs definitions vs selected train examples |Untested;extra components did not consistently help in source |
| Nested span graph / targeted labels |[Sodner](https://dialogue-conf.org/media/5744/abrosimovkiplusmosyaginaag037.pdf):corrected data73.13/56.94;augmented72.39/60.87 general/few-shotF1 |Separate rare-class and overall effects |Extra annotation can lower overall score;enumeration cost |Annotation correction vs augmentation,one variable each |Mixed source result;no automatic data expansion |
| Preserve factuality and anaphora |[FactRuEval organizer report](https://dialogue-conf.org/media/3430/starostinaetal.pdf) |Mentions≠current asserted facts |Foreign factuality policy may differ from MWS |Negation/history/future/conditional/role slices |Diagnostic candidate;no foreign-policy hardcode |
| Retrieval few-shot and calibrated abstention |[Tinkoff dynamic intents](https://habr.com/ru/companies/tbank/articles/696756/) |Low-data/new-class adaptation and negatives |Nearest-neighbor leakage;recall/latency cost |Fixed few-shot vs train-only retrieval;separate calibration |Candidate only if official rules allow |
| Rules propose,model verifies |[T-Bank complaints](https://habr.com/ru/companies/tbank/articles/899130/) |Precise literal candidates can reduce copying errors |Grouping accuracy is not extraction accuracy |Rules/LLM/candidate-verification ablation |No automatic union of facts |

## Track-specific rankings, not a universal winner

| Task/track | Verified reported ordering | Limitation |
| --- | --- | --- |
| [RuREBus official results](https://github.com/dialogue-evaluation/RuREBus#результаты-соревнования),NER |davletov-aa .561;Sdernal .464;ksmith .463 |Gold-RE ordering differs |
| RuREBus gold-entity RE |Sdernal .441;davletov-aa .394;viby .218 |RE winner code not located in bounded search |
| RuREBus end-to-end |ksmith .062 original table |davletov-aa .132 is additional/postdeadline,not original winner |
| [RuNNE report](https://dialogue-conf.org/media/5747/artemovaelplusetal109.pdf),general |Pullenti .811;MSU-RCC .749;SibNN .743 |Saldon .741 is fourth;baseline .674 |
| RuNNE few-shot |Pullenti .710;Saldon .644;MSU-RCC .604 |Baseline .447;winner-specific reproducible repo/ablation unverified |
| NEREL |Original paper MRC79.64;current README MRC84.99,SpERT82.52,Pyramid79.73 |Different snapshots,not verified competition-wide top3 |
| FactRuEval fact track |violet .51;green .33 standard F1;advanced .47/.31 |Only2systems;anonymous colors do not identify architecture |

RuNNE's strongest reported rules system and the multitask counterexample are
reasons to test simpler span/rule approaches seriously. None establishes that
span or generative models win on MWS, or that one corrupts its IDs more often.

## Top three architecture hypotheses

1. Compact generative LLM→strict JSON+evidence: flexible schema/relations;
   risk malformed/hallucinated/corrupted literals and decoder latency.
2. Compact encoder/token/span classifier→source-copy boundary: bounded output,
   exact values; risk nested/long inputs and inferred/normalized fact semantics.
   A base encoder without a trained task head is not a ready extractor.
3. Extract-then-link / span→compact classifier-normalizer: preserves literals,
   adds relation/factuality decisions; risk cascade and extra latency.

Compare only under the official ontology/metric and identical validation/
resource protocol. Start with3–5models, including an encoder; do not sweep
minor LoRA knobs before architecture evidence. No winner selected now.

## Transfer experiment matrix — blocked until real MWS data

Common: frozen group-aware MWS validation; official scorer primary; identical
base/recipe/token budget/seed/hardware, plus total compute reported. Different
data stages never silently share ontologies. If rights or organizer permission
are unknown, do not acquire/train. Auxiliary held-out documents/folds stay out
of auxiliary training. Calibrate on a separate allowed split.

| ID | Hypothesis | Change from EXP-0 | Expected result | Decision rule |
| --- | --- | --- | --- | --- |
| EXP-0 |MWS-only is a strong reference |Base→MWS SFT |Measured official score/resources |KEEP reproducible reference;no external gain claim |
| EXP-1 |Business IE behaviour transfers |Base→RuREBus→MWS |Rare-fact/overall official gain |KEEP only meaningful official gain within agreed cost;otherwiseREJECT |
| EXP-2 |RU boundary/evidence training transfers |Base→NEREL/IE→MWS |Better exact spans/IDs |Same rule;no gain on external benchmark alone qualifies |
| EXP-3 |Schema-conditioned auxiliary mix helps |Separate namespaced IE tasks→MWS |Broader extraction with no ontology bleed |KEEP only official gain;reject negative transfer |
| EXP-4 |Allowed targeted synthetic data fixes dominant errors |MWS-only+audited synthetic cases |Better measured dominant bucket |Only if allowed;KEEP on official gain without resource regression |

Meaningful threshold is set **before runs** from official metric resolution and
validation uncertainty, not invented here. Diagnose precision/recall, exact ID
preservation, validity, rare classes, unsupported evidence vs human-adjudicated
hallucination, latency/memory. If auxiliary training hurts MWS, remove it.
Multi-task loss, retrieval, augmentation and quantization are separate ablations.

## Research completion gate

| Question | Current answer |
| --- | --- |
| Confirmed? |Task direction/stack/48h,source metadata,public NLU++ license,core test evidence |
| Unknown? |Official schema/scorer/data/runtime,transfer gains,model quality |
| Build regardless? |Adapters/evaluator/EDA/tests/profile/ledger;already working |
| Wait for official data? |Label policy,final architecture,training/transfer/scorer decisions |
| Top3architectures? |Generative,span,extract-then-link/hybrid |
| Candidate models? |4controlled compact LLMs+1small RU encoder,metadata only |
| Baseline? |Empty and rules;synthetic F1=0/.591,not model-quality evidence |
| Improvement measurement? |Official frozen validation+quality/resource protocol |
| Highest risk? |Assuming auxiliary/foreign extraction matches MWS semantics |
| Next empirical test? |Pinned public adapter/occurrence-span integrity;then local model baseline after budget/data gates |

Stop broad research. Evidence/history lives here; next work follows BACKLOG.md.
