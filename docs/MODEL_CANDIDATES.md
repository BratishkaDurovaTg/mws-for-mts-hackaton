# Compact model candidates

Проверено 2026-10-21: только public Hub metadata, model cards, configs и
tokenizer configs. Веса не скачивались, inference/training не выполнялись.
Это начальный shortlist для контролируемого сравнения, не выбранная модель,
не полный каталог актуальных моделей и не ограничение размера от MWS.

Runtime probe2026-10-22: select only Qwen2.5-0.5B-Instruct at
`7ae557604adf67be50417f59c2c2f167def9a775` (Apache2.0,988097824weight bytes).
Current bounded check included [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B),
revision `2fc06364715b967f1860aea9cf38778875588b17`:1746942600bytes,
image-text-to-text, multimodal loader. The older native text-only probe
reduces acquisition/runtime cost for a mechanical check; it is not a final
model comparison. Existing locked ML extra installs unchanged on M4/16GiB;
MPS available, CUDA unavailable. Actual quality/compatibility remains to be measured.

Дополнение2026-10-22: пятый кандидат —
[cointegrated/rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2/blob/e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae/README.md),
29.4M, MIT, BERT encoder3layers/hidden312/context2048, F32 checkpoint
117,529,600bytes. Revision/config/digest в registry. Автор называет его
Russian-focused; EN/mixed и extraction quality UNKNOWN. Это BertForPreTraining,
**не готовый NER extractor**: span/token head требует разметки и обучения.
FP16/BF16/PEFT/quantization и latency/RAM для этой задачи не проверены.
Сравнить encoder/span, generative и extract-then-link до выбора final approach;
пять кандидатов — общий бюджет, не причина запускать все без hardware gate.
Машинно-читаемые revisions, размеры файлов и аналитические оценки находятся
в [configs/model_candidates.json](../configs/model_candidates.json).

## Проверенные metadata

| Candidate / pinned card | Card parameters | Serialized BF16 elements | Weight bytes | Architecture | Context: card / config |
| --- | ---: | ---: | ---: | --- | ---: |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/README.md) | 0.49B | 494,032,768 | 988,097,824 | Qwen2, 24 layers, 14Q/2KV | 32,768 / 32,768 |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/README.md) | 1.54B | 1,543,714,304 | 3,087,467,144 | Qwen2, 28 layers, 12Q/2KV | 32,768 / 32,768 |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md) | 0.6B | 751,632,384 | 1,503,300,328 | Qwen3, 28 layers, 16Q/8KV | 32,768 / 40,960 |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/README.md) | 1.7B | 2,031,739,904 | 4,063,515,592 | Qwen3, 28 layers, 16Q/8KV | 32,768 / 40,960 |

Все четыре — dense decoder checkpoints с RoPE/GQA, native Transformers
classes и Apache-2.0. Qwen2.5 варианты instruction-tuned; Qwen3 варианты
post-trained, с переключением thinking/non-thinking. Лицензия checkpoint
не означает, что его использование разрешено будущими правилами MWS.
BF16 storage подтверждён. FP16 conversion/runtime, hardware kernels и
качество после conversion пока не проверены.

Declared tokenizer class у всех `Qwen2Tokenizer`; fast implementation
зависит от runtime. `tokenizer_config.model_max_length=131072` не является
подтверждением model context. Пока используем консервативные 32,768 tokens
из cards; config Qwen3 содержит 40,960. Runtime должен проверять общий
input + generation budget и отказывать при overflow без silent truncation.
Для обоих Qwen2.5 cards дополнительно указывают generation limit 8192.

### Почему Qwen3 metadata больше card parameter count

`tie_word_embeddings=true` в pinned configs. Из serialized elements можно
вычесть одну embedding matrix `151936 × hidden_size`: получаем 596,049,920
для 0.6B и 1,720,574,976 для 1.7B, что согласуется с округлением card.
Это **аналитический вывод о tied storage**, не измерение реально загруженной
модели. Сохраняем обе величины; disk size учитывает все реально опубликованные
weight files. Index/tokenizer/config overhead не входит в weight bytes.

## Claims отдельно от task measurements

| Вопрос | Primary evidence | Что остаётся UNKNOWN |
| --- | --- | --- |
| RU / EN | Qwen2.5 cards перечисляют оба языка среди 29+; [Qwen3 announcement](https://qwenlm.github.io/blog/qwen3/) — среди 119 | Extraction quality RU, EN и mixed RU/EN на support tickets |
| Structured output | Qwen2.5 family cards заявляют улучшение JSON; Qwen3 имеет chat template и tool-output format | Valid JSON rate, exact IDs, grounding, schema adherence и actual F1 |
| Thinking | Qwen3 cards: включён по умолчанию, `enable_thinking=False` — hard switch | Quality/resource delta на extraction; оптимальный decoding |
| PEFT | [LoRA API](https://huggingface.co/docs/peft/en/package_reference/lora) позволяет target linear layers | Candidate-specific injection, masking, tiny-overfit и generation smoke |
| Quantization | [PEFT guide](https://huggingface.co/docs/peft/en/developer_guides/quantization), [bitsandbytes hardware guide](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes) | Разрешение MWS, target runtime, quality/memory/latency delta |

Для Qwen3 extraction baseline явно отключать thinking. Cards запрещают
greedy decoding в **thinking mode** и рекомендуют sampling в non-thinking;
deterministic non-thinking decoding — отдельная проверяемая гипотеза.
Одинаковый prompt/scorer/batch policy не отменяет model-specific chat template.

Quantization исследуется после FP/BF baseline. 4/8-bit training поддерживает
обучение дополнительных параметров; QLoRA требует отдельной подготовки
quantized base. Совместимость CUDA-примера нельзя переносить на macOS/MPS.
Экономия bits не доказывает уменьшение latency. Outlines/GGUF/AWQ/GPTQ или
другой engine не становятся зависимостью только из-за наличия кандидата.

## Аналитические lower bounds, не measured peaks

| Candidate | BF16 unique-weight payload, GiB | BF16 KV cache: batch 1, 2048 cached tokens, MiB |
| --- | ---: | ---: |
| Qwen2.5-0.5B-Instruct | 0.920 | 24 |
| Qwen2.5-1.5B-Instruct | 2.875 | 56 |
| Qwen3-0.6B | 1.110 | 224 |
| Qwen3-1.7B | 3.205 | 224 |

Weight estimate = inferred unique elements × 2 bytes. KV estimate =
`2 × layers × kv_heads × head_dim × cached_tokens × bytes_per_element`.
Не включены activations, allocator, framework, adapter, temporary tensors и
loading duplication. Peak RAM/VRAM, startup, warm p50/p95, throughput,
tokens/sec и adapter bytes пока **не измерены**. Больший Qwen3 KV cache
показывает, почему parameter count не заменяет profiling.

## Следующий model selection gate

Сначала official legality/schema/scorer и frozen validation; затем local
prompt-only benchmarks с одинаковыми данными, prompt semantics, decoding,
hardware и batch policy. Сохранять качество, failures, disk bytes и resource
measurements; выбирать по Pareto frontier, не назначать победителя по card.

Другую family добавлять для измерения независимых tokenizer/training/
architecture failure modes после проверки RU/EN, лицензии и runtime.
Это гипотеза о полезности сравнения, не доказательство превосходства.
Более новые [Qwen3.5 options](https://qwen.ai/blog?id=qwen3.5) также требуют
отдельного metadata/runtime review; текущий shortlist их не ранжирует.
