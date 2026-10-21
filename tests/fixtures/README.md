# Synthetic contract fixtures

`synthetic.jsonl` содержит 20 вручную сформулированных и явно размеченных
искусственных обращений. Это **внутренний provisional формат**, а не MWS dataset
или предположение об официальной schema. Каждый record помечен
`metadata.synthetic=true`, содержит slice names и policy version.

Тексты и gold labels заданы независимо от regex/rules extractor. Offset helper
при подготовке только вычислил положение явно выбранных цитат; правила
извлечения не использовались для получения labels. В fixtures присутствуют
`service`, `action`, `action_result` и `cause`, которые простой идентификаторный
baseline может не поддерживать. Его несовершенство здесь ожидаемо.

Политика разметки для тестов:

- `facts=[]` — размечено отсутствие фактов; `facts=null` означает unlabelled и
  не встречается в этом gold fixture.
- Указываются точные строки значений, без изменения ID, версии, URL или регистра.
- `evidence.start/end` — Python Unicode character offsets внутри конкретного
  message, полуоткрытый интервал `[start,end)`, не UTF-8 bytes.
- Fact types открыты. Explicit отрицания, исправления, прошлое/настоящее и
  гипотезы представлены отдельными qualifiers; неуказанная temporality остаётся
  `unknown`, без догадки по позиции сообщения.
- В вопросе оператора код не считается утверждением пользователя. Evidence
  сохраняет message ID и тем самым роль источника.
- «Не помог» — утверждённое значение результата действия, а не успешное решение.
- Пример с ключом сохраняет старую гипотезу и её текущее отрицание. Это локальная
  тестовая annotation policy; реальная политика будет определяться MWS.

Набор покрывает empty/no-facts, несколько фактов, error codes, версии, UUID,
документационный IP, resource IDs с различной пунктуацией, дату, URL с
зарезервированным `.invalid`, RU/EN mix, опечатку, emoji, длинный диалог,
отрицание, исправление `403 -> 404`, historical/current state, speaker separation,
попытку действия и опровергнутую причину.

Есть 18 group IDs. `syn-http`, `syn-http-copy` и `syn-http-near` принадлежат одной
`http-template` family намеренно: один exact duplicate и одна вариация. Их нельзя
разносить между train/dev. Набор проверяет plumbing и обнаруженные overlaps, а
не доказывает достаточность автоматического semantic deduplication.

Любые score/latency на этих 20 записях — результаты synthetic mechanics smoke
checks. Они **не оценивают качество на реальных support tickets и не предсказывают
leaderboard performance**. Здесь нет реальных клиентских данных или секретов.
