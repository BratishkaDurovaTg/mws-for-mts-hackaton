# Local data only

Raw official/support datasets belong here and are ignored by Git. Preserve raw
files read-only. Store their provenance, hashes and official permission locally.
Before modeling: read the spec, supply an adapter, validate, run EDA/leakage
checks, review annotation policy, freeze a grouped split. Fixtures in
`tests/fixtures/` are synthetic software tests, never official training data.
