# Local HF inference and gated LoRA training

Статус: реальные MWS quality measurements **pending**. Реальные local inference,
profiling, LoRA smoke, tiny-overfit и reload проверены2026-10-22:
`VERIFICATION.md`, `experiments/runtime_probe_20261022/`. Это synthetic mechanics.
Unit tests с fake tokenizer проверяют masking и gates, не качество LLM.
LoRA recipe в example — стартовая гипотеза, не оптимизированный набор параметров.

## Environment и локальные checkpoints

Core pipeline запускается без ML extra. После выбора hardware установить
явно `uv sync --locked --no-editable --extra ml`. Это отдельная потенциально крупная
установка; подготовка проекта её автоматически не выполняет.
Список точных package versions закреплён в lock. Не добавлены TRL, Datasets,
bitsandbytes или внешние trackers.

В `configs/hf_example.json` указаны относительный local directory, repository
ID и immutable revision. Самостоятельно положить snapshot выбранной revision
в этот каталог после разрешения на download. Нужны native model `config.json`,
tokenizer files/chat template и safetensors (все shards + index при sharding).
Не подменять tokenizer, не принимать remote code. Все loaders используют
`local_files_only=True`, `trust_remote_code=False`, model weights — safetensors.
Отсутствующие файлы/ML extra дают явный error prediction, а не empty success.

Model ID/revision в config — provenance claim. Small local files хешируются;
во время явного training run дополнительно вычисляются SHA256 всех local
weight files, а длительность GB streaming записывается отдельно от training.
Content identity входит в stage fingerprint и сравнение с baseline. Это
локальные hashes, не подтверждение совпадения с official Hub digests. Для
delivery проверить Hub SHA отдельно и сохранить evidence. Arbitrary local
directory не считается доказанной Hub revision. Обычный inference loader
не перечитывает GB weights ради hash на каждом запросе.
Adapter загружается отдельно поверх явно выбранного base.

```bash
uv run --locked --no-editable --extra ml compactfacts infer --input data/tickets.jsonl --config configs/hf_example.json --output artifacts/hf_predictions.jsonl
```

Первый HF run включает lazy loading. Greedy decoding: `do_sample=False`,
`num_beams=1`, explicit non-thinking chat template, fixed seed. Это не обещание
bitwise equality между kernels/hardware. Нет silent truncation: проверяются
input tokens и общий input+generation context budget. `max_retries` 0..2;
каждый retry получает fixed validation hint, attempts и общий generated token
count сохраняются. Exhausted output budget — visible failure, не partial JSON.

## SFT contracts

Inference и SFT используют один `build_messages`: gold/metadata исключены из
prompt, source roles/order/timestamps и исходные Unicode строки сохранены.
Это внутренний provisional extraction contract; после spec изменить adapter/
policy и заново зафиксировать baseline.

`build_features(ticket, tokenizer, max_length)` проверяет **token prefix**:
generation prompt должен быть точным prefix полного assistant target после
реального chat template. Если tokenizer/template меняет границу, run отказывает.
Prompt labels=-100, response и genuine EOS supervised; overflow/no target/
unlabelled ticket отказывают. Padding masked по позиции, поэтому EOS=PAD не
удаляет genuine EOS labels. Текущий runner single-process, без packing.

## Три явных stages

Сначала frozen split + leakage review. Runner сверяет manifest hash, ticket
IDs, content/annotation hashes и `leakage_audit.passed`. Negative screen не
доказывает полного отсутствия неизвестных зависимостей.

```bash
uv run --locked --no-editable --extra ml compactfacts train --config configs/train_example.json --train artifacts/split/train.jsonl --dev artifacts/split/dev.jsonl --id smoke-001 --hypothesis "Проверяем finite updates на tiny subset" --change "Reference contract only" --expected-result "Reproducible valid output" --decision-rule "KEEP only if stated gate passes"
```

1. `smoke`: tiny subset, по умолчанию 2 optimizer steps. Проверяет finite
   losses/updates; сохраняет actual generation validity и exact fact-set checks.
   Smoke output сам по себе не свидетельствует о способности извлекать факты.
2. `tiny_overfit`: новый config с теми же recipe/data/base/seed, новым output
   directory, `stage=tiny_overfit`, `prior_gate_manifest` указывает на passed
   smoke manifest. Требует final loss < initial loss × `minimum_loss_ratio`
   (default0.95) и **actual generated exact fact-set match для всех tiny examples**.
   Parser strict, retry отключён; валидный JSON без правильных фактов gate не проходит.
3. `full`: явный отдельный config/run после passed smoke и tiny manifests с
   тем же fingerprint. Synthetic fixture full run запрещён; data origin должен
   быть честно указан; actual `metadata.synthetic=true` также блокируется.
   `baseline_run_path` указывает на measured prompt-only HF `run.json` с теми
   же frozen canonical dev hash, base/revision, local weight content hashes и
   extraction/decoding contract. Baseline содержит finite primary metric.
   Никакой auto-escalation, обхода gate или автоподбора LR/rank.
   Changing recipe/data/prompt/code/base content требует повторить gates.
   Любой technical generation failure на full dev помечает gate failed.

Ранние stages не загружают adapter друг друга: каждый проверяет тот же fresh
base/recipe. Saved adapter и Trainer checkpoints сохраняются для review;
full quality измеряется на held-out dev, tiny checks — на training subset.
Final adapter не объявляется best: этот runner не выбирает checkpoint по
official metric. До delivery сравнить checkpoints общим evaluator+profiling.
`official_scorer` callable можно передать в `run_training`; его score становится
primary final validation objective, остальные metrics diagnostic.

## Сохранённое evidence

`training_manifest.json`: config, base ID/revision, local hashes/file sizes,
data/split hashes, Git commit/dirty state, seed, precision, quantization=null,
LoRA rank/alpha/dropout/targets, optimizer/scheduler/LR/batch/accumulation,
effective batch, actual epochs/steps, duration, package/hardware environment,
losses, generation gates, validation metrics, adapter bytes и checkpoint status.
Source tickets/targets/generated values в manifest не сохраняются. Output
directory новый/пустой; существующий run не перезаписывается.

Quantization, constrained decoding и autoresearch ожидают measured FP/BF
baseline, official policy и justified experiment; данный runner их не включает.
Перед долгим GPU run нужны понимание стоимости, target runtime и budget.

## Reproduce the one-model runtime probe

The probe configs use MPS/BF16 on the measured Mac; adapt explicitly for other
hardware. This is an explicit approximately988MB download, not an inference side effect:

```python
import json
from pathlib import Path
from huggingface_hub import snapshot_download

c = json.loads(Path("configs/runtime_probe.json").read_text())
snapshot_download(c["model_id"], revision=c["revision"], local_dir=c["model_path"], token=False,
                  allow_patterns=["config.json", "generation_config.json", "tokenizer.json",
                                  "tokenizer_config.json", "vocab.json", "merges.txt",
                                  "model.safetensors", "LICENSE", "README.md"])
```

With the installed ML extra and local weights:

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-editable --extra ml python scripts/inspect_real_generation.py --input experiments/runtime_probe_20261022/diagnostic.jsonl --config configs/runtime_probe.json --output artifacts/runtime-raw-replay.json
```

Original baseline/training commands and four-field hypotheses are preserved in
the ledger and published manifests. Recreate the frozen split at
`artifacts/real-ml/split` from the committed synthetic fixture before using
`runtime_smoke.json` then `runtime_tiny.json`; each stage needs a fresh output
directory. Saved adapters stay ignored. Stop after mechanical gates; do not
promote this memorized adapter as the best MWS checkpoint.
