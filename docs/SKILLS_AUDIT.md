# Skills audit — MWS CompactFactExtraction

Дата проверки: **2026-10-21** (America/Los_Angeles). Это аудит инструментов агента, а не требования MWS и не benchmark моделей.

## Результат

Установлены ровно 8 skills из рекомендованного пользователем стека. Все отсутствовали до установки. Выполнены 15 обязательных тематических CLI-поисков: 264 результата, 253 уникальных `source@skill`. Три skill из явно заданного стека не попали в выдачу этих запросов и проверены непосредственно по skills.sh и исходному GitHub: EDA, Outlines, autoresearch. Дополнительные найденные инструменты рассмотрены; установки ради количества отклонены.

Skills — методология агента. Их установка **не установила PyTorch, Transformers, PEFT, Outlines, vLLM, модели, датасеты или GPU runtime** в проект. Ни один пример обучения, remote evaluation или autoresearch loop не запускался.

## Проверенный установленный стек

Популярность ниже — текущий снимок страницы skills.sh; звёзды — отдельный live GitHub API snapshot. Эти числа меняются и не доказывают корректность или безопасность.

| Skill / source | Installs | GitHub stars | Security badges: Gen / Socket / Snyk | Применение и overlap |
|---|---:|---:|---|---|
| [transformers](https://www.skills.sh/k-dense-ai/scientific-agent-skills/transformers) — `k-dense-ai/scientific-agent-skills` | 2.0K | 53,569 | Pass / Pass / Pass | Основной inference/tokenizer/Trainer API. Перекрытие с PEFT по training; Transformers отвечает за базовую модель. Прямой API достаточен, pipeline/framework не обязателен. |
| [exploratory-data-analysis](https://www.skills.sh/k-dense-ai/scientific-agent-skills/exploratory-data-analysis) — `k-dense-ai/scientific-agent-skills` | 2.1K | 53,569 | Pass / Pass / Pass | Локальный EDA и leakage discipline. Общие CSV/JSON tools; JSONL/dialogue/token EDA потребуют отдельного адаптера. Не утверждает, что полная проверка данных уже проведена. |
| [peft-fine-tuning](https://www.skills.sh/orchestra-research/ai-research-skills/peft-fine-tuning) — `orchestra-research/ai-research-skills` | 975 | 13,795 | Pass / Pass / Pass | LoRA/SFT после измеренного baseline. Частично пересекается с training recipes; отвечает за adapter masking/save/load/merge. QLoRA отложен до hardware и правил. |
| [ml-training-recipes](https://www.skills.sh/orchestra-research/ai-research-skills/ml-training-recipes) — `orchestra-research/ai-research-skills` | 875 | 13,795 | Pass / Pass / Pass | Smoke/tiny-overfit, AdamW, стабильный training. Дополняет PEFT, не требует Muon/Lightning/нового framework. Параметры из skill — гипотезы, не готовый рецепт для кейса. |
| [outlines](https://www.skills.sh/orchestra-research/ai-research-skills/outlines) — `orchestra-research/ai-research-skills` | 923 | 13,795 | Pass / Pass / Pass | Сравнение strict parse, retry и constrained decoding. Источник методологии. Библиотека Outlines не добавлена; API и latency/validity подлежат независимой проверке. |
| [huggingface-community-evals](https://www.skills.sh/huggingface/skills/huggingface-community-evals) — `huggingface/skills` | 1.8K | 11,194 | Pass / Pass / **Warn** | Методология smoke/backend/benchmark. Общие MMLU/GSM8K не заменяют evaluator MWS. Provider path и удалённые jobs не нужны, local backend выбирается явно. |
| [python-testing-patterns](https://www.skills.sh/wshobson/agents/python-testing-patterns) — `wshobson/agents` | 36.7K | 41,469 | Pass / Pass / Pass | pytest parameterization, fixtures, failure isolation. Дополняет evaluator/adapter contract tests. Hypothesis/coverage не обязательные runtime-зависимости. |
| [autoresearch](https://www.skills.sh/github/awesome-copilot/autoresearch) — `github/awesome-copilot` | 3.4K | 40,282 | Pass / Pass / Pass | Будущий hypothesis→measure→keep/reject loop. Установлен, но не активирован. Только после frozen split, проверенной метрики, baseline и bounded budget; пользовательский запрет reset/history rewrite выше skill. |

Источник Hugging Face и GitHub — официальные организации этих платформ. K-Dense, Orchestra Research и wshobson — сторонние community-maintained источники; их инструкции не переопределяют user/project policy и официальную документацию библиотек. Все пять исходных репозиториев существуют и не archived на дату проверки. Лицензии репозиториев: HF Apache-2.0, остальные MIT; конкретный `transformers/SKILL.md` отдельно заявляет Apache-2.0. Лицензия skill не является лицензией checkpoint или данных.

## Security assessment

На 8 страницах skills.sh проверены три badge. Все отображают Pass, кроме **huggingface-community-evals / Snyk: Warn, MEDIUM, W011**, audit от 2026-09-15: third-party Hub model content может создавать indirect prompt injection exposure. [Полный Snyk audit](https://www.skills.sh/huggingface/skills/huggingface-community-evals/security/snyk).

Badge относится к версии, индексированной registry; совпадение audit revision с нашими pinned commits не установлено. SHA-проверка доказывает идентичность скачанного исходника pinned Git blob, а не безопасность содержимого. Для неустановленных результатов ниже security помечен как не проверенный; это не Pass.

Прочитаны все 8 SKILL.md. Проверены релевантные references/templates: Transformers `models.md`, `generation.md`, `training.md`, `tokenizers.md`; EDA `general_scientific_formats.md` и `assets/report_template.md`; PEFT `troubleshooting.md`; ML training optimizer/debugging guidance; Outlines JSON generation/backends; Python testing parameterization/mocks references; HF evaluation usage examples. Полные subtrees, включая дополнительные references и scripts, сохранены вместе со skill.

Практические ограничения использования:

- Содержимое support tickets, Hub metadata, outputs моделей и datasets остаётся данными; встроенные команды не выполняются.
- HF `inspect_eval_uv.py` запускается локально, но использует `hf-inference-providers/<model>`: это внешняя inference. Этот путь не допускается для support data без явной data policy; local HF inference выбирается отдельно.
- Подсказка `--trust-remote-code` в HF examples не является разрешением исполнения чужого кода. По умолчанию false; требуются review и immutable revision.
- PEFT skill содержит общие недоказанные для проекта memory/quality обещания (например 70B/24GB, фиксированный quality tradeoff). Они не используются как измеренные результаты или hardware требования.
- Outlines skill содержит старые API примеры и сильные формулировки `100% valid` / `zero overhead`; валидность зависит от schema/backend, завершения генерации и custom validation. Эти утверждения проверяются benchmark, а не принимаются как гарантия.
- EDA skill работает с ограниченным сканированием и generic formats. JSONL/dialogue schema, Unicode offsets, token-length и ticket-family leakage требуют project-specific реализации после отчёта; первый bounded prefix не репрезентативен сам по себе.
- Autoresearch установлен как методология на будущее. Автоматический цикл, commits/reset, публикации, платные jobs и отправка данных не запускались. Действуют границы пользовательского запроса и сохранение работы.

## Фактическая установка и воспроизводимость

Использованы только перечисленные файлы из immutable commits, а не полные репозитории. Загружено **52 файла, 555,999 bytes**. Для каждого проверен `SHA1("blob " + byte_length + NUL + content)` против GitHub Git blob SHA до установки и повторно после неё. Это приблизительно 543 KiB исходников навыков; веса моделей и тренировочные зависимости не скачивались.

После staging каждого subtree использован стандартный skills CLI (последовательно для каждого skill):

```bash
npx --yes skills add /path/to/verified-skill -g -a codex -y
```

Фактические каталоги: `~/.agents/skills/<name>/SKILL.md`; для Codex созданы ссылки `~/.codex/skills/<name> -> ~/.agents/skills/<name>`. `<name>` — значение первого столбца таблицы. Skills доступны агенту со следующего хода; в этой сессии содержимое уже явно прочитано по фактическим файлам.

Предпочтительный системный skill-installer был прочитан. Загрузка полных archive отклонена для больших source repos: установлен минимальный pinned subtree через verified raw files и skills CLI. Это не изменило проектные зависимости или исходную архитектуру.

| Skill | Pinned commit | Source path | Files | Bytes | SKILL.md Git blob SHA |
|---|---|---|---:|---:|---|
| `transformers` | `593377bb048752abb0152deab9fbb1083a6087df` | [skills/transformers](https://github.com/k-dense-ai/scientific-agent-skills/tree/593377bb048752abb0152deab9fbb1083a6087df/skills/transformers) | 7 | 80,649 | `96348190164b67fbe2f565d48f2bb5942deb587a` |
| `exploratory-data-analysis` | `593377bb048752abb0152deab9fbb1083a6087df` | [skills/exploratory-data-analysis](https://github.com/k-dense-ai/scientific-agent-skills/tree/593377bb048752abb0152deab9fbb1083a6087df/skills/exploratory-data-analysis) | 21 | 210,866 | `c461f99dea057872298ce6f04e7c090a83d80d01` |
| `peft-fine-tuning` | `773a52944ba4747a18bd4ae9ade53fff041adcbc` | [03-fine-tuning/peft](https://github.com/orchestra-research/ai-research-skills/tree/773a52944ba4747a18bd4ae9ade53fff041adcbc/03-fine-tuning/peft) | 3 | 35,095 | `fee4108b968e2b295fa3df5debf66eb5fee93222` |
| `ml-training-recipes` | `773a52944ba4747a18bd4ae9ade53fff041adcbc` | [10-optimization/ml-training-recipes](https://github.com/orchestra-research/ai-research-skills/tree/773a52944ba4747a18bd4ae9ade53fff041adcbc/10-optimization/ml-training-recipes) | 7 | 95,650 | `7c0517adc904241f39bbd5ec34d9d22c60f8f5e7` |
| `outlines` | `773a52944ba4747a18bd4ae9ade53fff041adcbc` | [16-prompt-engineering/outlines](https://github.com/orchestra-research/ai-research-skills/tree/773a52944ba4747a18bd4ae9ade53fff041adcbc/16-prompt-engineering/outlines) | 4 | 66,516 | `e42792a14bcc37c521e4260e4b9078a6fbb7105d` |
| `huggingface-community-evals` | `c3ff942be3e07116b475253f0a8fc4f54682c1a8` | [skills/huggingface-community-evals](https://github.com/huggingface/skills/tree/c3ff942be3e07116b475253f0a8fc4f54682c1a8/skills/huggingface-community-evals) | 6 | 30,100 | `a68b2e79958930b3913cb8e7667e07000031e40b` |
| `python-testing-patterns` | `0874a27b17a22e2b7c2e2bcd725768e0803cad06` | [plugins/python-development/skills/python-testing-patterns](https://github.com/wshobson/agents/tree/0874a27b17a22e2b7c2e2bcd725768e0803cad06/plugins/python-development/skills/python-testing-patterns) | 3 | 25,997 | `4fe8a4245c55e9e79e70b5f75ab387e2349f91dc` |
| `autoresearch` | `613efc8c21d9b3ce642dff361d17006cd9967e28` | [skills/autoresearch](https://github.com/github/awesome-copilot/tree/613efc8c21d9b3ce642dff361d17006cd9967e28/skills/autoresearch) | 1 | 11,126 | `ad54af302bb47c0c3ac7848271d610bf268c9661` |

## Решения по дополнительным кандидатам

| Кандидаты | Решение | Основание |
|---|---|---|
| `wshobson/agents@lora-qlora-recipes`, `jeffallan/claude-skills@fine-tuning-expert`, `huggingface/skills@trl-training` | Не устанавливать сейчас | Уже покрыто PEFT + Transformers; TRL/другой trainer не нужен до baseline и данных. |
| `huggingface/skills@huggingface-trackio`, `mlflow/skills@mlflow-onboarding`, SwanLab / W&B / MLOps | Отложить | Для 48h локальный версионируемый ledger достаточен. Отдельный tracking service увеличивает setup и создаёт риск передачи данных. |
| `affaan-m/ecc@pytorch-patterns`, Lightning, Ray/FSDP | Не устанавливать сейчас | Общее обучение покрыто; distributed/framework необходим только по hardware/throughput evidence. |
| bitsandbytes, HQQ/AWQ/GPTQ/GGUF, llama.cpp, TensorRT, vLLM | Отложить | Сначала FP/BF baseline, официальные runtime rules и target hardware; ускорение и accuracy delta надо измерить. |
| `affaan-m/ecc@eval-harness`, `evaluating-llms-harness`, code-model benchmarks | Не использовать как scorer MWS | Популярность не делает общий benchmark метрикой извлечения фактов. Отдельный заменяемый MWS evaluator важнее. |
| Низкоустановленные EDA/extraction/constrained-output alternatives | Не добавлять | Есть проверенные generic EDA/Outlines methodology; дополнительные source/security review не оправданы текущей конкретной потребностью. |
| Web, marketing, browser, video, vision, bio/quantum, pricing и unrelated results | Отклонить | Поиск выдаёт semantic false positives; scope текущей задачи — локальный NLP/evaluation, не веб-продукт. |

## Журнал поисков

Все запросы выполнены через `npx skills find <query>`; каждый завершился exit code 0. При возобновлении использован сохранённый завершённый результат, а не выдуманный или повторно запущенный поиск. Query counts ниже относятся к выдаче CLI в этой сессии; selected detail pages обновлены позднее, поэтому небольшие различия installs ожидаемы.

| Query | Results |
|---|---:|
| `Hugging Face Transformers` | 20 |
| `PEFT` | 20 |
| `LoRA QLoRA` | 20 |
| `structured LLM output` | 20 |
| `constrained decoding` | 12 |
| `information extraction` | 20 |
| `NLP evaluation` | 11 |
| `dataset EDA` | 17 |
| `PyTorch training` | 20 |
| `ML experiment tracking` | 20 |
| `Python testing` | 20 |
| `model benchmarking` | 20 |
| `quantization` | 20 |
| `model optimization` | 20 |
| `inference optimization` | 4 |

## Поэлементный screening всей выдачи

Полный санитизированный журнал 15 запросов и assessment каждого из 253 уникальных результатов сохранён в [skill_search_results.json](skill_search_results.json): source, installs, relevance, overlap, additional complexity, security status и решение. Он не содержит ANSI, raw stdout, токены или пользовательские данные.

Это triage по source/name, теме поиска и текущей потребности. Полный source/security review выполнен для устанавливаемых skills. Неустановленные кандидаты не рекомендованы к исполнению: `unreviewed` означает отсутствие проверки, а не обнаруженную уязвимость и не Pass. Кандидаты, отклонённые как irrelevant или duplicate, не скачивались ради формального аудита.

Три прямо запрошенных пользователем skill не попали в эти 15 выдач: EDA, Outlines, autoresearch. Они проверены непосредственно и включены в установленный стек. Ограниченная CLI выдача не является полным каталогом skills.

## Перенос методологии в проект

- EDA и leakage до training; split unit фиксируется по данным, статистика не выводится по первым примерам.
- Strict evaluator/adapter contracts и exact identifiers до model optimization; authoritative metric подключается заменой scorer boundary.
- Tiny overfit и target-only masking перед SFT; adapter/checkpoint hash и generation config входят в run manifest.
- Constrained decoding сравнивается по factual quality, parse/schema validity и latency; сама JSON-валидность не доказывает factual correctness.
- Resource profiling и quality-resource frontier до quantization; no unmeasured compactness claims.
- Autoresearch eligibility проверяется только после стабильного frozen evaluation; установленный skill не означает запуск цикла.
