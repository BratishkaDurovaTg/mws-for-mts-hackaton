# MWS CompactFactExtraction: pre-implementation report

Дата проверки: 2026-10-21. Статус: подготовка до выдачи официального case specification.
Этот документ — исследовательская записка проекта, а не требования MWS.

## 1. Current repo state

Read-only проверка выполнена до проектирования и Python-кода:

- Репозиторий пуст; присутствует только `.git`, tracked/untracked файлов нет.
- `git status`: no commits yet on `master`; история отсутствует.
- `origin` указывает на приватный GitHub-репозиторий этого проекта.
- README, AGENTS.md (включая родительские каталоги), source, notebooks, data,
  tests, configs, experiments, requirements, lock, Docker и CI отсутствуют.
- Работающего baseline, placeholder-кода и пользовательских изменений нет.
- Локально доступны macOS arm64, uv 0.10.2 и Python 3.12.12; системный
  Python 3.13.6 не будет неявно определять окружение проекта.
- Технический долг исходного проекта отсутствует; научная неопределённость
  велика. Сохранять или мигрировать существующую архитектуру не требуется.

## 2. Confirmed requirements / CONFIRMED

[Официальная страница организатора](https://globalhack.innopolis.university/),
проверена 2026-10-21: кейс MWS CompactFactExtraction посвящён компактным
моделям извлечения фактов из обращений технической поддержки на основе LLM.
Публично перечислены Python, PyTorch, Hugging Face Transformers/PEFT.
Даты хакатона: 13–16 ноября 2026; самостоятельная работа: 48 часов.
Перечисление стека не определяет версии библиотек и не является полным ТЗ.

Требования пользователя к нашей подготовке: измеримость, воспроизводимость,
заменяемые adapters/scorer, offline inference, evidence audit, тесты, baseline,
EDA/leakage screening, PEFT path, profiling и сохранённый experiment ledger.
Это требования проекта; они не приписываются организаторам.

## 3. Unknown requirements / UNKNOWN

В пределах выполненного публичного поиска официальный case specification,
dataset и evaluator не были получены. Это не доказательство их отсутствия.
Неизвестны:

- dataset fields, fact types, языки и annotation policy;
- train/dev/test, split unit, hidden testing и submission schema;
- official scorer, exact match/F1, нормализация, semantic equivalence;
- правила evidence, negation, speaker attribution и временного состояния;
- parameter/weight-size limits, CPU/GPU, RAM/VRAM, runtime/container;
- latency/throughput constraints и метод измерения компактности;
- quantization, synthetic data, external APIs/models, internet at evaluation;
- pretrained checkpoint rules, permitted licenses и data policy.

С сайта найдены ссылки на [программу](https://photostorage.innopolis.university/index.php/s/WEYQnCWoZwTmGPK)
и [положение](https://photostorage.innopolis.university/index.php/s/Tpc4osJbxzeJ3m3).
Получение содержимого завершилось сетевой ошибкой/timeout. Содержимое не
используется как подтверждение каких-либо технических ограничений.

## 4. Assumptions / INFERRED

| Предположение | Риск | Проверка / действие при опровержении |
| --- | --- | --- |
| Локальный JSONL полезен как внутренний transport | LOW RISK | Формат явно provisional; добавить официальный adapter |
| Fact types должны быть открытым словарём | LOW RISK | Проверить официальный label inventory; конфигурация allowed types |
| Raw values и координаты evidence следует сохранять | LOW RISK | Сверить official normalization; adapters отделены от scorer |
| Зависимые обращения нельзя делить случайно по строкам | MEDIUM RISK | EDA, group/customer/incident/family анализ и frozen manifest |
| Нужен локальный inference без внешних API | MEDIUM RISK | Проверить runtime; сервисы не обязательны |
| RU/EN mix заслуживает отдельного diagnostic slice | MEDIUM RISK | Проверить реальные языки и аннотации |
| Качество следует сравнивать с ресурсами | MEDIUM RISK | Не вводить собственный официальный порог компактности |
| Mention polarity/temporality полезны внутри | MEDIUM RISK | Внутренние qualifiers не навязываются submission schema |
| Кандидаты до 2B могут быть достаточны | HIGH RISK | Только проверяемая гипотеза, не выбранная модель или лимит MWS |
| LoRA улучшит качество относительно prompting | HIGH RISK | Одни данные/split/scorer; результат пока UNKNOWN |

## 5. Data hypothesis

Возможны единичный ticket, несколько сообщений с ролями/временем или уже
структурированные обращения. Возможны списки фактов, slots, spans или
нормализованные значения. До official spec не угадываем одну из этих схем.

Внутренняя provisional representation: Ticket(id, messages, optional facts,
group_id, metadata); Message(id, role, content, optional timestamp);
Fact(type, value, evidence, polarity, temporality, certainty). `facts=null`
означает отсутствие разметки, `facts=[]` — размеченное отсутствие фактов.
Это различие обязательно при оценке и обучении. Значения строковые:
`403`, `v1.2.3`, `bucket-17` не преобразуются в числа/общие слова.
Evidence использует offsets Python Unicode code points, end exclusive,
отдельно в каждом исходном message. Source roles и порядок не стираются.

## 6. Candidate approaches

| Подход | Что проверяем | Ограничение / решение сейчас |
| --- | --- | --- |
| A0: empty predictor | Sanity-check recall и negative examples | Явный нижний baseline |
| A: deterministic rules | Точные IP/UUID/version/code/date/resource mentions | Не считается полноценным semantic extractor |
| B: compact prompt-only LLM | Zero/few-shot, strict JSON, evidence | Реальные веса и качество пока не измерены |
| C: SFT + PEFT | Прирост official quality при заданных ресурсах | После A/B, smoke и tiny-overfit gates |
| Hybrid | Rules помогают LLM с exact identifiers | Не включать merge автоматически: риск FP/duplicate/conflict |

Для structured output сравнить strict parser, явно помеченное fence repair,
bounded retry и constrained JSON/grammar generation. Outlines изучен как
методология; его библиотека не добавляется без эксперимента. Утверждения
skill о нулевой стоимости/100% валидности не переносим на наш runtime:
token limit, custom validators и semantic grounding требуют проверки.

## 7. Evaluation plan

Сначала независимый evaluator с проверяемыми вручную TP/FP/FN примерами.
Official scorer подключается как отдельный callable и становится primary;
до него primary помечен `provisional`.

- Exact raw fact matching — основной внутренний diagnostic.
- Normalized exact — отдельный отчёт, opt-in allowlist по type; по умолчанию
  identity. Никакой semantic equivalence без разрешения official scorer.
- Micro precision/recall/F1, macro/per-type F1, exact fact-set accuracy.
- Missed rate=FN/gold; unmatched prediction rate=FP/predicted — это не
  доказательство hallucination. Grounding audit измеряется отдельно.
- Schema/parse/technical failure rates, duplicate rate; failure не считается
  корректным пустым prediction даже на empty gold.
- Не игнорировать missing/extra/duplicate ticket IDs.
- Error buckets: missing/wrong type/value/qualifier, malformed/technical,
  duplicate, unsupported evidence; speaker/obsolete/long-context ошибки
  требуют ручной adjudication и не выдаются за автоматически доказанные.
- Wall-clock cold startup, warm p50/p95, throughput/goodput, RAM high-water,
  CUDA allocated/reserved peaks при наличии, weight/adapter disk bytes,
  generated tokens/sec при наличии. Unavailable metrics=null с причиной.

Каждый run сохраняет config/data hash, Git revision+dirty state, seed,
environment, command, metrics, artifacts и KEEP/REJECT с основанием.
Синтетические fixtures проверяют software contracts, не leaderboard quality.

## 8. Model selection criteria

До выбора base checkpoint — `docs/MODEL_CANDIDATES.md` и immutable model
revisions в config. Критерии: official legality, лицензия, RU/EN task quality,
exact identifier preservation, context/tokenizer overhead, structured-output
validity, loaded memory и disk size, latency/throughput, PEFT compatibility.
Сравнивать при одинаковом split/prompt/scorer/decoding/hardware/batch policy.
Отделять card claims, аналитическую оценку и реальные measurements.
Parameter count не равен количеству serialized tensors при tied weights.
Победитель до measured validation не назначается; сравнение — Pareto frontier.

## 9. Risks

Leakage между шаблонами/инцидентами/клиентами; малая или несбалансированная
validation; ошибки official annotation; hallucinations и value corruption;
негативные, исторические, предположительные и исправленные упоминания;
speaker confusion; context overflow; invalid JSON; GPU/backend mismatch;
несовместимость library versions; увеличение latency после quantization;
PII в логах и artifacts; фиктивное качество на synthetic fixtures.

Меры: raw immutable data; bounded strict readers; fail visible; exact evidence;
no silent truncation; safe aggregate logs; no real support text to APIs;
local-only model loading; pinned environment; tests/manifest/ledger;
unknowns сохраняются до официальных данных. Near-duplicate screen не является
доказательством полного отсутствия leakage.

## 10. Proposed architecture and review

Рассмотрены (1) monolithic notebook/script, (2) небольшая typed Python package,
(3) отдельные сервисы + orchestration/tracking platform. Выбрана (2): сохраняет
единичную CLI/clean-env воспроизводимость и позволяет заменять официальные
границы без инфраструктурного overhead. Notebook допустим только как view.

| Модуль | Ответственность |
| --- | --- |
| schemas | Strict canonical data, prediction/error envelopes, JSON schema |
| data | Bounded JSONL IO, configurable adapters, EDA, leakage, frozen split |
| preprocessing | Сохранение raw текста и message structure; no implicit rewriting |
| models | Общий extractor contract; empty/rules/local HF backends |
| inference | Predict single/batch, parser, validation, evidence audit |
| evaluation | Provisional exact scorer, error analysis, official scorer extension |
| postprocessing | Только explicit allowlisted normalization; no identifier edits |
| training | Chat-template SFT masking/collation, LoRA, smoke/overfit/full gates |
| utils | Hashes, private atomic output, provenance, profiling, ledger |
| CLI/configs | Reproducible commands and typed JSON configs; no hidden state |

Не добавляем заранее Outlines, vLLM, bitsandbytes, MLflow, W&B, Datasets,
FastAPI или контейнер GPU без требований. Core environment небольшой;
PyTorch/Transformers/PEFT/Accelerate — отдельный opt-in extra.

## 11. Pre-registered implementation hypotheses and acceptance

| ID | Гипотеза | Baseline / проверка | Acceptance criterion |
| --- | --- | --- | --- |
| H1 | Canonical boundary не привязывает pipeline к input fields | Два разных raw mappings одного ticket | Одинаковая canonical запись, round-trip и failure tests |
| H2 | Evaluator сохраняет exact semantics | Hand-computed TP/FP/FN, empty/failure/duplicates | Все analytic cases, `403 != 404`, версии/IDs не сливаются |
| H3 | Split удерживает обнаруженные families вместе | Connected components exact/template/group/near duplicates | Zero discovered cross-split overlaps, repeatable manifest |
| H4 | Rules дают проверяемый non-empty baseline | A0 vs A на hand-labelled synthetic fixtures | Saved metrics + known failures; без quality claims |
| H5 | SFT labels supervise только response | Fake tokenizer boundary/EOS/padding cases | Prompt/pad=-100, genuine EOS сохранён, no empty target/truncation |
| H6 | Offline inference предсказуемо отказывает при плохом setup | Missing model, malformed JSON, overlong input | Typed technical failure, никогда silent empty success |
| H7 | Profiling честно отделяет measured/unavailable | Fresh process + repeated warm requests | Positive timing, RAM + environment, null reasons for missing GPU/tokens |
| H8 | Repair/retry не маскирует parser failures | Fixed parser corpus strict vs fence-repair | Separate repaired/failure counters, bounded retry tests |

До следующей фазы требуется concrete plan. После реализации каждая строка
будет сверена с тестами и сохранёнными artifacts; намерение не равно результату.

## 12. Methodology sources

- [Transformers generation](https://huggingface.co/docs/transformers/en/main_classes/text_generation)
  и [chat templates](https://huggingface.co/docs/transformers/en/chat_templating).
- [PEFT LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora).
- [uv lock/sync](https://docs.astral.sh/uv/concepts/projects/sync/).
- [pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html).
- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
  [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065).
  Author list/current record verified 2026-10-21; contribution: EDA and
  Transformers workflow methodology. Skills are reviewed guidance, not an
  authority above official library documentation or MWS rules.
