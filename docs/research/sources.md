# Sources index

- MWS and library methodology: ../PRE_IMPLEMENTATION_REPORT.md §12.
- Checkpoint cards/config revisions: ../MODEL_CANDIDATES.md and ../../configs/model_candidates.json.
- Analogue datasets/systems: ../RELATED_TASKS.md.
- Winning systems and ideas: ../PRIOR_ART.md.
- Public fixture source/license digests: ../../benchmarks/public/manifest.json.
- D18 telecom/cloud/support update (checked 2026-10-22; three new primary sources;
  classification, limitations and future tests in ../PRIOR_ART.md addendum):
  - [MWS second-line Jira ticket analysis, SRE lead's account](https://habr.com/ru/companies/ru_mts/articles/1051784/)
    — INDUSTRIAL LESSON; informal engineer assessment, not extraction benchmark.
  - [MWS AI subscriber-bot pilot, team's account](https://habr.com/ru/companies/mts_ai/articles/956790/)
    — INDUSTRIAL LESSON; hybrid routing/fallback and delivery/latency tradeoffs.
  - [Microsoft SoftNER author paper, v2, §§IV/VII](https://arxiv.org/html/2101.05961v2)
    — INDUSTRIAL LESSON; datatype-aware extraction, automatically labelled evaluation.
  Public dataset/code release is not established by these inspected sources;
  no data/model acquisition or training occurred in this update.

Budget: original paper + official repository/task scorer per technical question;
stop when another source no longer changes an engineering decision.
