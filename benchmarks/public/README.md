# Public auxiliary extraction diagnostic

This is an evaluation-only public-data slice, separate from synthetic contract
fixtures and future MWS validation. It cannot predict MWS leaderboard quality.
Public text and converted tickets remain ignored under `artifacts/`; this directory
contains only acquisition references, attribution and the licence.

## Available slice

PolyAI **NLU++**, English banking, first 10 records of the pinned
`nlupp/data/banking/fold0.json`: 9 literal slot annotations and 4 labelled-empty
records. These counts describe the selected slice, not dataset performance.
The source has no record IDs: our ticket ID contains its path and zero-based array
position. The whole source fold shares one group and must be excluded from any
future auxiliary training when this slice is used for evaluation.

The [NLU++ paper](https://arxiv.org/html/2204.13021v3) describes anonymised real
queries supplemented with expert-created rare cases. Individual origins are not
identified in the released rows. Banking queries provide a closer conversational
slot task than news NER, but their language and domain differ from anticipated
Russian technical support; official MWS languages and schema remain unknown.

Gold preserves source slot names under `nlupp.slot.*`, exact literal strings and
Unicode code-point spans `[start,end)`. Source intents, structured canonical
values and original annotations remain in metadata. No negation, temporality or
relation labels are inferred. Canonical qualifier defaults are transport
placeholders, not annotated semantics. Existing inference prompts exclude both
gold facts and metadata.

**Russian public slice: PENDING.** One bounded request for the official MASSIVE
RU validation viewer rows returned `Not found`. No archive was downloaded.
Technical identifiers, negation, long context, noisy text and relations are not
claimed as covered by this slice. Existing synthetic adversarial tests remain
separate software checks. Source pretraining contamination remains unknown.

## Reproduce

After source changes, rebuild the installed package:

```sh
uv sync --locked --no-editable --reinstall-package mws-compactfacts
uv run --locked --no-editable python scripts/prepare_public_benchmark.py \
  --output artifacts/public_benchmark --count 10
```

This explicitly invoked command downloads only the pinned JSON and licence,
each bounded to 256 KiB, using certificate and hostname verification. It verifies
both expected SHA256 values before conversion. No archives, models, dependencies,
training or remote execution are involved. Existing output artifacts are never
overwritten; use a fresh output directory for a replay. `--count` supports 1–20;
changing it defines a different diagnostic slice.

For a fully offline replay, add verified local source files:

```sh
uv run --locked --no-editable python scripts/prepare_public_benchmark.py \
  --output artifacts/public_benchmark_replay --count 10 \
  --source-file artifacts/public_benchmark/source.json \
  --license-file artifacts/public_benchmark/LICENSE.txt
```

Outputs are `source.json`, `LICENSE.txt`, `tickets.jsonl`, and `manifest.json`.
The manifest records upstream revision/path/hash, individual row locators/hashes,
canonical row/file hashes, attribution and limitations. Outputs contain public
data, not synthetic data, and must remain ignored. The reference manifest here
freezes the default selection; it contains no utterance or gold text.

## Attribution and licence

Publisher: **PolyAI Limited**. Iñigo Casanueva, Ivan Vulić,
Georgios P. Spithourakis and Paweł Budzianowski (2022),
*NLU++: A Multi-Label, Slot-Rich, Generalisable Dataset for Natural Language
Understanding in Task-Oriented Dialogue*.
[Paper](https://arxiv.org/abs/2204.13021),
[pinned repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b).

The upstream README explicitly applies **CC BY 4.0** to its shared datasets;
the complete licence is retained in [LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt).
No dataset-specific copyright-holder notice is supplied in that licence;
no additional notice is invented here. Changes: selection of first rows and
conversion of supplied literal slots into canonical tickets; no relabelling.
This project has no endorsement or affiliation with the dataset authors.
