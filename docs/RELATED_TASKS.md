# Related tasks — bounded evidence map

Checked2026-10-22. Ratings are planning hypotheses, not measured transfer.
T/D/L/C/P=task/domain/language/label-compatibility/transfer-potential,0–5.
L assumes Russian MWS input, which remains UNKNOWN. C=0 means **no official
label-compatibility evidence**, not proven incompatibility. Similar task is
not permission to reuse data or import its ontology.

| Dataset/System | Language | Domain | Task / output | Size | Metric | T/D/L/C/P | Useful for |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| [RuREBus](https://github.com/dialogue-evaluation/RuREBus) | RU | Regional strategic reports | BRAT entities+relations;8entity/11relationtypes |188train parts/30test files |Micro NER/RE/e2e F1 |4/2/5*/0/4 | Business IE transfer; large domain gap to support |
| [NEREL](https://aclanthology.org/2021.ranlp-1.100/) | RU | Wikinews | Nested/discontinuous entities, document relations/events;29/49types |933docs;746/94/93;~56Kentities/~39Krelations |NER/RE P/R/F1 |4/1/5*/0/3 | Literal boundaries, nesting, cross-sentence context |
| [RuNNE](https://dialogue-conf.org/media/5747/artemovaelplusetal109.pdf) | RU | NEREL news | Nested/few-shot spans;same underlying corpus |27,665/5,587/5,826entity mentions after reduction |General/few-shot macro F1 |3/1/5*/0/3 | Rare/nested spans; not independent NEREL data |
| [FactRuEval2016](https://dialogue-conf.org/media/3430/starostinaetal.pdf) | RU | Social/political news | Mentions, attributes, coreference;Ownership/Occupation/Meeting/Deal facts |255docs;122demo/133test |Custom comparator, alternatives/partial credit |4/1/5*/0/3 | Fact-vs-mention and factuality stress tests |
| [NLU++](https://arxiv.org/abs/2204.13021) | EN | Banking/hotels queries | Multi-intent+exact slot spans;62intents/17slots |3,080utterances |Task-specific intent/slot metrics; our slice exact spans |4/3/1/0/4 | Closest small licensed conversational extraction slice |
| [MASSIVE1.1](https://github.com/alexa/massive) |52languages incl.RU/EN | Virtual assistant | Intent+slot annotations |19,521utterances/language |Intent/slot/exact-match metrics |4/2/5*/0/4 | Multilingual slots; support-domain distance |
| [Restaurants-8K/Span-ConveRT](https://github.com/PolyAI-LDN/task-specific-datasets) | EN | Restaurant booking | Conversational span extraction |~8K |Span/slot evaluation |4/2/1/0/3 | Dialogue-conditioned span architecture |
| [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) | EN | Banking support |77intents;no fact/span gold |13,083queries |Intent accuracy |1/3/1/0/2 | Low-data intent methodology, not extraction score |
| [Ubuntu corpus](https://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/) | EN | Technical IRC support |Response-selection dialogues;no supplied fact gold |Large;not fetched |Response ranking |1/5/1/0/2 | Close domain, distant task; no invented fact labels |
| [Tinkoff dynamic intents](https://habr.com/ru/companies/tbank/articles/696756/) | RU examples | Banking support |BERT/KNN+classifier, held-out intents, negatives |Not published |WeightedF0.5,coverage,negative accuracy |2/3/5*/0/3 | Calibration/hard negatives; no open support corpus established |
| [T-Bank complaint analysis](https://habr.com/ru/companies/tbank/articles/899130/) | RU | Bank complaints |Regex product groups→BERT→UMAP→HDBSCAN→topic names |~20K/month stream,not public dataset size |92%sampled regex-group accuracy,not extractionF1 |1/3/5*/0/2 | Topic/error discovery; flattening/cleaning unsuitable for exact IDs |

## Schema/scorer and reuse cautions

- RuREBus entity codes: ACT,BIN,CMP,ECO,INST,MET,QUA,SOC; relation codes:
  NNG,NNT,NPS,FNG,FNT,FPS,PNG,PNT,PPS,GOL,TSK. Foreign labels stay isolated.
  [Organizer paper](https://publications.hse.ru/pubs/share/direct/373988573.pdf)
  documents150-sentence partitioning and annotation/moderation. Guidelines are
  linked in the repo; their PDF body was not retrieved. Scorer allows swapped
  relation endpoints and upstream parser failures can become absent predictions;
  preserve task semantics only in its adapter, keep failures visible here.
- NEREL original2021 baseline and current README are different snapshots:
  MRC NER79.64 vs84.99. Do not combine them into a ranking. RuNNE uses exact
  `(start,end,type)` sets, excluding its3few-shot types from general macro F1.
  [Main scorer](https://github.com/dialogue-evaluation/RuNNE/blob/main/scoring_program/score.py)
  differs from a sibling few-shot script; pin the final task configuration.
- RuREBus/NEREL/RuNNE: inspected repository license metadata null/no root
  LICENSE; annotation reuse permission UNKNOWN. NEREL underlying Wikinews
  CC-BY2.5 does not establish annotation-package rights. FactRuEval has MIT
  software/documentation LICENSE; third-party news rights require separate review.
  No texts from these corpora are vendored or used for training.
- NLU++ dataset CC-BY4.0 is explicit in its repository. Authors describe
  anonymized real queries plus expert-created rare cases, not exclusively real
  support tickets. Pinned small acquisition and exclusion policy: benchmarks/public.
- MASSIVE RU bounded rows request returned not-found; no large archives fetched.
  RU public benchmark coverage is pending, not silently replaced by synthetic RU.

## Industrial lessons and search stop

Tinkoff's2022author account reports3–5seconds for new examples to reach the
index, **not inference latency**. Held-out-intent evaluation, hard negatives
and separate calibration mattered; improving a proxy initially harmed the
production objective. Transfer hypotheses require official MWS validation.
The2025T-Bank article's92% concerns regex grouping on a manual sample whose
size is unspecified. Topic clusters are not gold facts; retain message mappings.

Targeted search covered T-Bank/Tinkoff, Yandex, Sber, MTS/MWS, VK, Alfa,
Raiffeisen, Rostelecom, ODS/Dialogue, DataFusion, AIJourney/AIChallenge and
public dataset catalogs. [Yandex slots](https://yandex.ru/dev/dialogs/alice/doc/ru/nlu)
and [Sber slots](https://developers.sber.ru/docs/ru/va/code/nlp/slot-filling/fields)
are methodology docs, not proof of public training data. No further licensed
close extraction corpus was established within this bounded search; absence is
not asserted. Stop broad research here; next useful evidence is the public
adapter test and, later, official MWS validation.
