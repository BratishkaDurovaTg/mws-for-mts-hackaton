# MWS CompactFactExtraction

Воспроизводимая подготовка к кейсу ИННОГЛОБАЛХАК-2026: typed adapters,
независимый evaluator, EDA/leakage checks, baseline inference, profiling,
локальный HF backend и поэтапный LoRA runner.

**Официальные dataset, schema, scorer и hardware constraints ещё не получены.**
Внутренние форматы и метрики provisional. Модель не выбрана и не скачана.
Синтетические software fixtures не являются прогнозом leaderboard quality.
Реальные LLM inference/training ещё не проверены запуском; unit tests их
контрактов используют небольшие doubles.

[Контекст и источники](docs/PRE_IMPLEMENTATION_REPORT.md) ·
[план и решения](docs/IMPLEMENTATION_PLAN.md) ·
[проверки и ограничения](docs/VERIFICATION.md).

## Окружение

Python **3.12.12**, uv **0.10.2**; exact direct/transitive versions и хеши в
`uv.lock`. Core: Pydantic и psutil; dev: pytest/coverage/ruff. PyTorch,
Transformers, PEFT и Accelerate — отдельный opt-in `ml` extra.

```bash
uv sync --locked --no-editable
uv run --locked --no-editable ruff check src tests
uv run --locked --no-editable ruff format --check src tests
uv run --locked --no-editable pytest -q
```

`--no-editable` также обходит локальную особенность macOS: hidden flag на
editable `.pth` мешает Python импорту. После source edits обновите пакет:
`uv sync --locked --no-editable --reinstall-package mws-compactfacts`.
Основной workflow проверяется из установленного пакета и отдельного wheel.

## End-to-end без модели

Команды выполняются из корня checkout. `artifacts/` игнорируется Git.
CLI не заменяет output без `--force` и защищает входные файлы.
Для повтора используйте новый путь либо явно `--force`.

```bash
uv run --locked --no-editable compactfacts validate --input tests/fixtures/synthetic.jsonl
uv run --locked --no-editable compactfacts schema --output artifacts/schema.json
uv run --locked --no-editable compactfacts eda --input tests/fixtures/synthetic.jsonl --output artifacts/eda.json
uv run --locked --no-editable compactfacts split --input tests/fixtures/synthetic.jsonl --output artifacts/split
uv run --locked --no-editable compactfacts infer --input tests/fixtures/synthetic.jsonl --config configs/rules.json --output artifacts/predictions.jsonl
uv run --locked --no-editable compactfacts evaluate --gold tests/fixtures/synthetic.jsonl --predictions artifacts/predictions.jsonl --output artifacts/metrics.json
uv run --locked --no-editable compactfacts benchmark --input tests/fixtures/synthetic.jsonl --config configs/rules.json --repeats 10 --warmup 1 --output artifacts/benchmark.json
uv run --locked --no-editable compactfacts submission --predictions artifacts/predictions.jsonl --output artifacts/canonical-export.jsonl
```

Последний файл — **canonical diagnostic export**, не official submission.
`submission --official` без adapter завершается ошибкой. Нормализация по
умолчанию identity; explicit per-type allowlist можно передать через
`evaluate --normalization path.json`. `403 != 404`, `v1.2.3 != v1.2.4`.

Single/batch Python API:

```python
from compactfacts.data import read_tickets
from compactfacts.inference import predict_batch
from compactfacts.models import create_extractor
from compactfacts.schemas import InferenceConfig

tickets = read_tickets("tests/fixtures/synthetic.jsonl")
engine = create_extractor(InferenceConfig(backend="rules"))
single = engine.predict(tickets[0])
batch = predict_batch(engine, tickets)
```

## Данные, EDA и split

Реальные данные положите локально в `data/` после data-policy check. Raw не
изменяется, файлы игнорируются Git. Формат не угадывается: настройте adapter.
`configs/mapping_example.json` — пример произвольных полей, не схема MWS.

```bash
uv run --locked --no-editable compactfacts adapt --input data/raw.jsonl --mapping configs/mapping_example.json --output artifacts/canonical.jsonl
```

Reader: UTF-8/BOM, strict JSONL/NDJSON; default64MiB/100000records.
Malformed/blank lines, duplicate keys/IDs, non-finite values и превышение
бюджета дают ошибку; строки не пропускаются молча. `facts=null` — unlabelled;
`facts=[]` — размеченное отсутствие фактов.

EDA сохраняет aggregate schema/nulls/labels/facts, robust/classical lengths,
языковые/технические patterns, duplicates и annotation conflicts. Язык —
эвристика. Whitespace lengths — **не** model tokens; true token distribution
требует явного local tokenizer в `analyze` Python API. Near scan bounded,
completeness и лимиты записываются в отчёте. Raw texts/IDs не печатаются.

Split объединяет обнаруженные exact/template/near и явные
incident/customer/family/group связи в connected components. Manifest содержит
seed, hashes, controlled train/dev IDs, audit limits. `leakage_audit.passed`
означает отсутствие обнаруженных cross-split links, не доказанную независимость.
Stratification/semantic duplicate adjudication требуют official-data adaptation.
Если организатор дал split, не заменяйте его автоматически.

## Baselines и experiments

`empty` — sanity reference; `rules` — exact technical mentions с ограниченными
qualifier heuristics. Rules не понимают все исправления, speaker truth,
причины проблем и результаты действий. Evidence audit проверяет substring и
offsets, а не семантическую истинность.

```bash
uv run --locked --no-editable compactfacts run --input tests/fixtures/synthetic.jsonl --config configs/rules.json --output artifacts/rules-run-001 --id rules-run-001 --hypothesis "Exact identifier rules form a reproducible software baseline" --conclusion "Synthetic fixture regression reference only" --decision KEEP --reason "Preserve measured reference"
```

Run сохраняет predictions/metrics/benchmark/config/provenance и append-only
`experiments/runs.jsonl`. [Ledger policy](experiments/README.md).
Малые подготовительные результаты: `experiments/baselines/`.
Сравнивайте configs на одном frozen dev:

```bash
uv run --locked --no-editable compactfacts compare --runs artifacts/model-a/run.json artifacts/model-b/run.json --output artifacts/comparison.json
```

Comparator проверяет data/scorer/hardware/batch/decoding protocol, исключает
technical failures и строит measured Pareto set. `winner=null`: общий порог
компактности без official objective не выдумывается.

Evaluator: raw micro/per-class/macro F1, exact fact-set accuracy,
missed/unmatched prediction rates, validity/failure/duplicate/repair/retry
rates, error buckets. FP не объявляется автоматически hallucination;
semantic hallucinated rate — null. Error buckets могут пересекаться.

Benchmark: fresh process; cold time от worker module entry до первого
prediction (без process spawn), отдельное model loading; warm p50/p95,
serial throughput/goodput, OS process RAM high-water, CUDA allocator peaks,
logical weight/adapter bytes, generated tokens/sec. RAM включает interpreter
и imports; CUDA peaks не включают весь driver memory. Rules tokens и MPS peak
не подменяются оценками: null с причиной. Один cold sample не даёт variance.

## Локальная LLM и PEFT

Сначала [candidate matrix](docs/MODEL_CANDIDATES.md), затем official
license/runtime/budget check. Четыре checkpoint — контролируемый shortlist,
не утверждение об актуальном лучшем варианте. Task quality не измерено.

После выбора окружения и download budget:

```bash
uv sync --locked --no-editable --extra ml
```

Это устанавливает крупные ML-библиотеки; команда при подготовке не выполнялась.
Веса выбранной immutable revision скачайте отдельно в `models/` после
проверки ресурсов. Inference сам ничего не скачивает: local files only,
native code, safetensors; обязательных external APIs нет.
Имя local directory/revision — provenance, не доказательство Hub origin.
Explicit HF run/training hashes local weights для согласованности gates;
это не заменяет проверку official download hashes.

```bash
uv run --locked --no-editable --extra ml compactfacts infer --input artifacts/split/dev.jsonl --config configs/hf_example.json --output artifacts/hf-predictions.jsonl
uv run --locked --no-editable --extra ml compactfacts train --config configs/train_example.json --train artifacts/split/train.jsonl --dev artifacts/split/dev.jsonl --id smoke-001 --hypothesis "Response-only LoRA features support finite gradient updates"
```

Отредактируйте configs под доступные local weights/hardware. Пример config
не означает выбора победителя. [Training runbook](docs/TRAINING.md):
smoke→tiny overfit→explicit full, target masking/EOS, frozen baseline gates.
Full training на synthetic fixtures запрещён. Реальный прирост пока не измерен,
best model/checkpoint не выбран.

## Официальная адаптация

[Runbook](docs/OFFICIAL_ADAPTATION.md): spec→adapters/scorer→EDA→frozen split→
A/B→tiny overfit/SFT→error-driven changes→target-runtime replay.
Official scorer: `--scorer module:function`, становится primary.
Submission: `--official --adapter module:factory`. Это reviewed local executable
plugins; current canonical schema не навязывается официальному формату.

Agent skills установлены отдельно и не являются runtime dependencies:
[аудит8skills/15поисков](docs/SKILLS_AUDIT.md). Attribution методологии EDA/
Transformers, включая Kassis et al. (2026), добавлена в источники отчёта.
