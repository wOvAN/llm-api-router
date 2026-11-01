# Конвертер диалектов API (chat ↔ messages ↔ responses)

> Статус: дизайн согласован в диалоге, реализации нет.
> Объём — **вариант A**: перевод тела и стрима между диалектами. Никакого хранилища ответов в
> прокси (`previous_response_id` не материализуем), никакого `chat ↔ completions`, ничего audio/embeddings.
> Конверсия включается, когда бэкенд **достаточно** хорошо говорит на диалекте клиента, а не когда строки
> в `api_types` не совпали.
> Потерянное поле = 400 по умолчанию; молча выбрасывать разрешает только `settings.convert.drop_params`
> (позиция LiteLLM: `UnsupportedParamsError` с именем поля, дроп — opt-in).

## Цель

Сейчас роутер — прозрачный ретранслятор: путь клиента определяет протокол
(`apiTypeFromPath`: `/messages` → anthropic, иначе openai), протокол определяет только **URL**
бэкенда (`Server.GetURLForAPIType`) и форму heartbeat-кадра. Тело запроса и ответ никто не
переводит. Следствие: попытка, у которой бэкенд не говорит на диалекте клиента, уезжает на
апстрим с чужим путём и чужим телом, апстрим отвечает 404/422, а 4xx/5xx по существующему
правилу **не считаются ошибкой** → ни retry, ни fallback. Клиент получает сырой 404 бэкенда,
правило считается здоровым, в метриках — успешный fallback-проход. Это дыра, и она уже
существует; конвертер — способ её закрыть, а не только способ «дать Claude Code локальный
llama.cpp через OpenAI-эндпоинт».

Второе следствие, более тонкое: у локальных движков поддержка диалектов **градированная**
(см. таблицу ниже). «Бэкенд отдаёт `/v1/responses`» не значит «Responses-запрос клиента
пройдёт»: у vLLM `store` молча игнорируется, и следующий запрос цепочки падает уже на другом
движке. Значит триггер конверсии — не имя протокола, а **способность бэкенда обслужить
конкретный запрос**.

## Не делаем (сознательно)

- **Никакого стора ответов в прокси.** `previous_response_id` материализуется только там, где
  его реализовал автор бэкенда (vLLM с `VLLM_ENABLE_RESPONSES_API_STORE`, SGLang с включённым
  response store) — тогда он едет как есть. Прокси-стор воспроизвёл бы худший из отказов,
  которые мы сами себе перечислили: `200` с тихо усечённой историей, плюс требует общего
  состояния между репликами (`docs/ha-replicas.md`: всё состояние процесса per-process), TTL,
  вытеснения и семантики дерева. Это отдельная подсистема и отдельный spec.
- **Никакого `chat ↔ completions`** (legacy text completions). `/v1/completions` хочет сырой
  prompt; chat-шаблон живёт в tokenizer'е бэкенда, у прокси его нет. Строгий 400.
- **Никакого mid-stream fallback / регенерации** — как класса, в том числе между диалектами.
  LiteLLM это делает: до первого чанка — повтор с неизменным запросом, после — чат-путь уходит
  на fallback с нуля (`router.py:2426`), Responses-путь дописывает `developer`-инструкцию
  «Continue exactly where it stopped» + partial-ассистент (`router.py:2606`), usage складывается
  из двух попыток. Цена — склейка двух потоков в один SSE-ответ, буферизация сгенерированного в
  прокси, разбитый prefix cache и (по собственному признанию автора, `router.py:2620-2624`)
  «providers without prefill semantics ... **may regenerate**», то есть клиенту дублированный
  текст посреди предложения. Между диалектами добавился бы второй слой: вторая половина потока в
  другой событийной схеме. Наше правило остаётся: mid-stream → `*proxy.MidStreamError` в
  клиентском диалекте, сервер unhealthy, fallback невозможен (заголовки ушли).
- **Никакого прямого `messages ↔ responses`.** Только через хаб (chat), два спока.
- Ничего audio / embeddings / `systemone` / `realtime`: нет парного диалекта.
- Никакого авто-ремёпа `tools`-имён «в обе стороны с коллизиями» на v1: коллизия = 400.
- Никакого стора «второй копии тела» в трассе (см. «Наблюдаемость»).

## Словарь

Диалекты (`convert.Dialect`), которые конвертер различает:

| Диалект | Путь клиента | Форма тела |
|---|---|---|
| `chat` | `/v1/chat/completions` | OpenAI Chat Completions |
| `responses` | `/v1/responses` | OpenAI Responses |
| `messages` | `/v1/messages` | Anthropic Messages |
| `completions` | `/v1/completions` | legacy, конверсии не подлежит |

**Хаб — `chat`** (как в LiteLLM). Spoke = пара `(диалект → хаб)` и `(хаб → диалект)`;
N² переходов никто не хранит. Канон = структуры `chat`, они же — единственная точка, где
маппинги проверяются тестами.

## Что умеют бэкенды (проверено по `.info`, не по именам роутов)

| | `/v1/responses` | `/v1/messages` | стрим Responses | state |
|---|---|---|---|---|
| llama.cpp | шим над chat: `tools/server/server-chat.cpp:32` (`input`→messages, `instructions`→system, `tools[].type`→`function`) | есть (`server.cpp:190`) | created/in_progress/output_item.*/content_part.*/output_text.*/function_call_arguments.*/completed; нет `failed`, `incomplete`, `reasoning_summary_*`, hosted-тулзы | **400** `'previous_response_id'` (`server-chat.cpp:36`) |
| vLLM | полный (`entrypoints/openai/responses/`) | есть (`entrypoints/anthropic/api_router.py`) | самый богатый: +`web_search_call.*`, `mcp_call*`, `code_interpreter_call*`, `custom_tool_call_input.*`, `failed`, `incomplete` | `store` **молча игнорируется** без env-флага (`responses/serving.py:162-172`), стор — dict без вытеснения (`FIXME: memory leak`, `:183`) |
| SGLang | есть (`srt/entrypoints/openai/serving_responses.py`) | есть (`http_server.py:2150`) | есть `reasoning_summary_text.*`/`reasoning_summary_part.*` | под флагом store; иначе `_response_store_disabled_error` (`:339`); в коде `TODO(v): ensure this is propagated` (`protocol.py:2395`) |
| audio.cpp | нет (чата нет) | нет | — | — |
| Strata | есть (`/v1/responses`, `/v1/messages`, `count_tokens`) | есть | не аудитировано | — |
| облако OpenAI | полный | нет | полный | полный |
| облако Anthropic | нет | есть | — | — |

Вывод, определяющий приоритет рёбер: `responses → chat` нужно не ради имени пути, а ради
state-несовместимости, которой **нет ни у одного локального движка по умолчанию**. Ребро
`messages ↔ chat` — самое частое по быту (Claude Code против OpenAI-only бэкенда; OpenAI-клиент
против облака Anthropic).

Модель способности нельзя вывести из `/v1/models`: полноту эндпоинта там не объявляет никто
(`supported_endpoints` в этом клоне vLLM отсутствует, `server-models.cpp:2168` у llama.cpp даёт
`id`/`owned_by`/`created_at`).

## Модель способности

`api_types` **не перегружаем** — сегодня это ключ выбора URL (`GetURLForAPIType`), а не
объявление способности; существующие конфиги объявляют в нём единицы, и «переводить по
отсутствию строки» молча сломало бы каждому, у кого llama.cpp описан как `["openai"]` при
работающем `/v1/messages`.

Способность живёт в новом поле сервера:

```json
"servers": {
  "llama": {
    "url": "http://host:8080/v1",
    "api_types": ["openai"],
    "protocols": [
      { "path": "/v1/chat/completions" },
      { "path": "/v1/messages" },
      { "path": "/v1/responses", "state": false }
    ]
  }
}
```

- Пусто (`protocols` не задан) = **совместимость**: считаем, что бэкенд говорит на том, что
  объявлено в `api_types`, плюс (для локальных движков) результаты зонда. Поведение
  существующих конфигов не меняется ни на байт.
- Решение о конверсии на **каждую попытку** (у fallback-сервера и jev-кандидата свои
  способности): `plan(clientDialect, path, body, server.protocols)`.
- `rule.convert` / `fallback.convert` / `jev.candidates[].convert`: `"auto"` (по
  способности) | `"off"` (никогда, чистый passthrough) | `"messages>chat"` (принудительно,
  для бэкендов, которые зонд не распознаёт).
- Per-request предохранитель `X-Router-Convert: off` (в домашнем стиле `X-Router-Jev: off`).
  LiteLLM такие переключатели возит в теле запроса; мы не можем: тело запроса конвертером и
  так переписывается, заголовок — единственное, что доходит до решения о конверсии.

Настройки:

```json
"settings": {
  "convert": {
    "max_tokens_default": 4096,
    "drop_params": false,
    "probe": false,
    "anthropic_version": "2023-06-01"
  }
}
```

Главный инструмент наблюдения — **сухой прогон**, `POST /admin/api/convert/preview` с телом
`{path, body, server_id}`: прогоняет `plan()` + `convert.Request` и отдаёт, что уехало бы на
бэкенд (путь, тело, список потерь), **не трогая бэкенд**. Это ровно LiteLLM
`POST /utils/transform_request` (`proxy_server.py:12109` → `return_raw_request`) и
`GET /utils/supported_openai_params` (`:12081`). Дёшево до неприличия: `convert.Request` — чистая
функция, эндпоинт — её обёртка.

Зонд способностей (`config/capabilities.go` рядом с `CapabilitiesCache`, тот же кеш) —
**выключен по умолчанию** (`probe: false`) и не входит в фазу 1. `POST {}` различает «нет роута»
(404) и «есть роут» (400 с текстом про поле: llama.cpp `'input' is required`), а
`{"previous_response_id":"x"}` выдаёт 400 с текстом про это поле ⇒ поддержка есть, `state: false`.
Почему это не дефолт: LiteLLM не зондирует ничего (grep по `probe` — только Prisma-watchdog), он
**объявляет** способность кодом (`get_supported_openai_params`) и показывает её через preview.
Активный зонд — это настоящий запрос на бэкенд: токены, rate-limit-бюджет, у audio/TTS-эндпоинтов
— реальный синтез. Preview закрывает ту же потребность (понять, что сломается) без побочного
эффекта, а `protocols` на сервере закрывает потребность «я знаю, что умеет мой билд».

## Точка врезки

**Запрос — как можно раньше, ответ — как можно позже.** Всё остальное в цепочке продолжает
видеть нативную форму апстрима, и `proxy/usage.go`, `modelRewriteWriter`, `loopDetector` и
трасса не меняются ни на байт.

- `router.Handle`: диалект клиента считается один раз из пути. `plan()` вызывается **внутри
  цикла попыток**, рядом с `rewriteAttemptBody` (`router/router.go:294`): результат —
  `(upstreamPath, convertedBody)`. `serverURL` (`:323`) берётся по **upstream**-диалекту, не
  по клиентскому.
- `EnsureStreamUsage` (`:245`) сегодня привязан к клиентскому пути и лежит вне цикла
  попыток. Инъекция `stream_options.include_usage` бессмысленна для Anthropic-апстрима и
  обязательна для chat-апстрима ⇒ **переезжает внутрь попытки** и решается по
  upstream-диалекту. `stripStream` получает то же происхождение.
- `ping` (`:258`) остаётся по **клиентскому** диалекту: heartbeat читает клиент.
  `chat`/`responses` → SSE-комментарий, `messages` → `event: ping`.
- Ответ: `convert.Stream` в цепочке **после** `sseRelay` (там уже целые `\n\n`-кадры) и
  **до** `silenceWriter`-пары. Контракт: вход — один целый нативный кадр, выход — 0..N целых
  кадров клиента. Нативные байты до конвертера уже прошли `metricsWriter` (usage/тайминги) и
  `modelRewriteWriter` (имя модели правится в нативной форме — клиент увидит своё имя в своей
  форме).
- Mid-stream: `sendMidStreamError` и `loopDetector.sendLoopError` обязаны писать кадр в
  **клиентском** диалекте (`chat` → `data:`-чанк с `finish_reason:"error"`; `messages` →
  `event: error`; `responses` → `event: error` + `response.failed`). Существующая
  таблица «OpenAI vs Anthropic» расширяется третьей веткой.
- Ошибки апстрима (4xx/5xx, которые мы форвардим как есть) после конверсии становятся
  нечитаемыми SDK клиента (у Anthropic `{"type":"error","error":{...}}`, у OpenAI
  `{"error":{"message",...}}`). Значит **тело ошибки тоже переводится** — узкий переводчик по
  трём формам, без разбора смысла.

## Маппинг полей: `messages` ↔ `chat`

Эталон для сверки — LiteLLM `llms/anthropic/experimental_pass_through/adapters/transformation.py`
(1584 стр.) и `.../adapters/streaming_iterator.py` (1102), плюс llama.cpp-шим как живая
спецификация того, что отвергать громко.

| Канон (`chat`) | `messages` | `responses` | Судьба |
|---|---|---|---|
| system-сообщение | top-level `system` (string\|blocks) | `instructions` | lossless; несколько system → конкатенация в один |
| `max_tokens` (опц.) | `max_tokens` **обязателен** | `max_output_tokens` | chat-клиент без лимита → `settings.convert.max_tokens_default` (4096). У нас нет таблицы возможностей по моделям (у LiteLLM это costs map: `get_max_tokens(model)`, и лишь при промахе `DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS`), так что 4096 — основной источник, а не последняя защита: переопределение на правиле `rule.convert_max_tokens` обязательный спутник фичи |
| `stop` | `stop_sequences` | — | в `responses` терять некуда → поле не переводится, ребро `chat↔responses` его теряет |
| `tools[].function.parameters` | `tools[].input_schema` | `tools[].function` | lossless; имена `/[a-zA-Z0-9_-]{1,64}/` |
| `tool_calls[].{id,name,arguments}` / role `tool` + `tool_call_id` | `tool_use`(id,name,input) / `tool_result`(tool_use_id) | `function_call`(call_id,name,arguments) / `function_call_output` | lossless, `id` ↔ `call_id` |
| `parallel_tool_calls` | `tool_choice.disable_parallel_tool_use` | `parallel_tool_calls` | lossless |
| `reasoning_effort` / `reasoning{effort,summary}` | `thinking{type,budget_tokens}`, блоки `thinking`/`redacted_thinking` с **подписью** | `reasoning{effort,summary}` | **lossy**: effort↔budget_tokens — грубая сетка; подпись Anthropic не воспроизводится ⇒ запрос с `thinking`/`reasoning` на конвертируемом пути = 400 по умолчанию (просить оператора включить `drop_params`, тогда `thinking=off` в notes). Эмулировать подпись запрещено |
| `response_format` (`json_schema`) | `output_format` (+`additionalProperties:false`) | `text.format` | lossless с правкой схемы |
| `image_url{url: data:...}` | `source{type:base64,media_type,data}` | `input_image` | lossless; Anthropic `document`/PDF-блок в chat не помещается → 400 |
| `usage` (`prompt_tokens`/`completion_tokens`, `prompt_tokens_details.cached_tokens`, `completion_tokens_details.reasoning_tokens`) | `input_tokens`/`output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens` | `input_tokens`/`output_tokens` + details | метеринг не трогаем: читается с нативной стороны |
| `cache_control` / `prompt_cache_key` | `cache_control{ephemeral}` | `prompt_cache_key` | **не переносится**; конверсия рвёт prefix cache бэкенда — видно в колонке `Cache hit` |
| `stop_reason`: `stop`/`length`/`tool_calls`/`content_filter` | `end_turn`/`max_tokens`/`tool_use`/`stop_sequence`/`refusal` | `status`: completed/incomplete/failed | таблица; `refusal`/`content_filter` теряются, пишутся в notes |
| — | `anthropic-version`, `anthropic-beta` | — | при синтетическом `messages`-запросе `anthropic-version` инжектится (дефолт `2023-06-01`); beta-фичи → notes, не дропаем молча |
| `metadata.user_id` | `metadata.user_id` | `prompt_cache_key`/`safety_identifier` | jev-sticky уже читает обе ветки (`router/jev.go:545`), менять не нужно |

Политика потерь (позиция LiteLLM, `anthropic/chat/transformation.py:424-432`): поле, у которого нет
представления в целевом диалекте, по умолчанию **отказывается** — 400 с именем поля и текстом «set
`settings.convert.drop_params` to drop unsupported params», запрос на апстрим не уходит. Молча
выбрасывать разрешает только `settings.convert.drop_params: true` (глобально или
`rule.convert_drop_params`), и тогда каждое выброшенное поле уходит в `ConvertNotes`. Причина, по
которой дефолт — отказ: тихий дроп `stop`/`tools`/`thinking` даёт клиенту правдоподобный ответ,
полученный не из того запроса, а это ровно тот класс отказов, ради которого мы отказались от
стора ответов.

Отдельный класс — то, что в целевом диалекте не выражается **никак**: Anthropic `document`/PDF в
chat; hosted-тулзы (`web_search`, `file_search`, `code_interpreter`, `mcp`) на бэкенде без них;
коллизия имён тулзов после усечения до 64 байт; `n>1`; `audio`-блоки. Это 400 всегда, `drop_params`
их не включает.

Политика отказа касается **запроса**. Ответ маппится по таблице всегда и всегда молча: отказывать
нельзя, ответ уже у клиента, поэтому несопоставимое имя (`refusal`, `content_filter`) уходит в
`ConvertNotes`, а не в ошибку.

## Маппинг полей: `responses` → `chat`

Одно направление, и оно stateless: `input` (string \| items) → `messages`, `instructions` →
system, `function_call`/`function_call_output` → assistant `tool_calls` / role `tool`,
`max_output_tokens` → `max_tokens`, `text.format` → `response_format`,
`reasoning{effort}` → `reasoning_effort`, `include`/`truncation`/`background`/`store` → notes
(прокси их не исполняет), `previous_response_id` → 400 с текстом «backend has no response
store; send full `input`» (текст лучше чужого `store-disabled`: он объясняет клиенту, что
делать). Обратное направление (`chat` → `responses`) — тот же код с эмиттером кадров, фаза 4.

## Стрим: канонический событийный IR

Конвертер стрима — **stateful на запрос**, не чистая функция: у Anthropic есть lifecycle
контент-блоков (`content_block_start`/`delta`/`stop` с индексом), у Responses —
`sequence_number`, `output_item`/`content_part`-вложенность и терминальный
`response.completed` со всем объектом ответа. У chat-чанка индекса блока нет. Поэтому:

- `convert.Stream` — объект на запрос, с `Write(frame []byte) []Frame` и `Finish() []Frame`.
- Внутри: нормализация нативных кадров в канонические дельты
  (`text`, `reasoning`, `tool_call{name,args_delta,index}`, `usage`, `finish`, `error`), затем
  эмиссия в диалект клиента. Один IR, два эмита, N парсеров — spoke-пара.
- `Finish()` обязан добить хвост (открытый блок/`content_block_stop`, `message_delta` с
  usage, `response.output_item.done` + `response.completed`), как это уже делают
  `modelRewriteWriter.Finish()` и `sseRelay.finish()`.
- Heartbeats и wait-signals в конвертированный стрим идут **клиентским** кадром и мимо IR
  (они пишутся ниже конвертера, в `silenceWriter`), поэтому не участвуют в склейке.
- Лимиты: конвертер не буферит тело целиком; состояние — индексы блоков, буфер аргументов
  тулз (усекается), накопленный usage. Тот же порядок, что у 256KB-буфера `metricsWriter`.

## Отказы

Три разных исхода, не смешивать:

1. **Нечего переводить** (audio/embeddings/systemone/`completions`) → 400, поле в тексте.
2. **Бэкенд не может** (`store`-нуждающийся Responses на движке без стора) → 400 с
   подсказкой, **до** отправки на апстрим. Ничего не отправлено → fallback не считается
   неудачей попыток: роутер пробует следующую попытку, и только когда попытки кончились,
   отдаёт наш 400 (а не 502 «all backends failed»).
3. **Конвертер сломался mid-stream** (кадр, который он не собрал, при уже отправленных
   заголовках) → `*proxy.MidStreamError` в клиентском диалекте; сервер помечается unhealthy
   тем же путём, что сейчас. Молча «пропустить кадр» запрещено: клиент получит обрезанный
   ответ без ошибки.

## Наблюдаемость

- `RequestMetric.UpstreamEndpoint` — путь, уехавший на апстрим; `RequestMetric.Convert` —
  `messages>chat` (пусто = passthrough); `RequestMetric.ConvertNotes` — список того, что
  потеряно (`thinking=off`, `stop_dropped`, `store_ignored`), ≤ 512B.
- `X-Router-Convert` (+ `-Notes`) в ответ; `X-Router-Status`/`-Server` не меняются.
- `class:*`-агрегация остаётся по **клиентскому** пути: «что прислал клиент» — тот вопрос,
  на который таблица Request Class отвечает. Пятая семья ключей не вводится; в Recent Requests
  конверсия — пилюля рядом с классом (в tooltip — notes), как уже сделано для jev и audio.
- Трасса: запрос сохраняется **как прислал клиент** (после красной цензуры), ответ — как
  прислал бэкенд (нативная форма: `metricsWriter` стоит до конвертера). Второй копии
  «что уехало» нет (2×512KB на запись); вместо неё в записи — `convert`/`upstream_endpoint`.
  Для отладки самого конвертера этого хватает: `POST /admin/api/trace` + SDK-тесты ниже.

## Тесты

- **Юнит на маппинг**: золотые векторы `вход → выход` на спок, табличные. Источники векторов:
  поведение llama.cpp-шима, LiteLLM-адаптеры, типы `.info/apis/openai-go` и
  `.info/apis/anthropic-sdk-go`.
- **Юнит на стрим**: последовательность нативных кадров → последовательность кадров клиента,
  включая «кадр разрезан пополам» (у нас уже есть `TestStreamProxyRewritesModelSplitAcrossReads`
  как образец), пустой дельта-чанк, терминальный usage-кадр, mid-stream обрыв.
- **E2E в `router`**: hijacked `httptest`-бэкенд, говорящий в другом диалекте; проверка
  кадров, дошедших до клиента. Требование к скорости: только живые `httptest`-серверы,
  никаких `localhost:1` (`CLAUDE.md`: цена долгого suite — таймауты dial).
- **`sdktest/`** — главный слой: официальные `openai-go`/`anthropic-sdk-go` как клиенты.
  Именно он доказывает, что сгенерированные кадры парсятся реальными SDK, а не нашим
  парсером. Добавить кейсы «Anthropic SDK → chat-бэкенд», «OpenAI SDK → anthropic-бэкенд».
- Имена: `TestConvert*`, `TestMessagesToChat*`, `TestChatToMessages*`, `TestResponsesToChat*`,
  `TestHandleConvert*`.
- Регресс дыры, ради которой всё началось: `TestHandleProtocolMismatchConvertsFallback` —
  primary без способности, fallback с ней → 200, `X-Router-Convert` выставлен, метрика
  записана; и `TestHandleProtocolMismatchNoConvert` → 400 нашей формулировки, не 404 бэкенда.
- Политика потерь: неподдерживаемое поле → 400, в тексте имя поля, hijacked-бэкенд не получил
  ничего; с `drop_params: true` → запрос ушёл, поле перечислено в `ConvertNotes`.
- `TestConvertPreview`: сухой прогон через `/admin/api/convert/preview` возвращает путь, тело и
  потери; отдельный счётчик подтверждает, что на бэкенд не ушло ни одного запроса.

## Фазы

1. **Каркас без перевода.** Модель способности (`protocols`), `plan()`, `UpstreamEndpoint`/
   `Convert`/notes, `X-Router-Convert` (+ `off`-предохранитель), `settings.convert` со всеми
   четырьмя полями, `/admin/api/convert/preview`, громкие 400 вместо 404 бэкенда,
   `EnsureStreamUsage` переезжает в попытку. Польза: дыра «молча форвардим чужой 404» закрыта,
   точка врезки существует, оператор сухим прогоном видит, что сломается, наблюдение показывает
   конверсии (а их пока ноль).
2. **`messages → chat`**: JSON, затем SSE. Это то, ради чего фичу просят (Claude Code против
   OpenAI-only бэкенда).
3. **`chat → messages`**: обратное направление того же спока, здесь же `max_tokens_default` и
   `rule.convert_max_tokens` (направление, где Anthropic требует поле, которого у клиента может
   не быть).
4. **`responses → chat`** (stateless, `previous_response_id` → 400), `chat → responses` — по
   требованию.

## Риски

- Горячий путь: декод/код JSON на кадр. Passthrough (без конверсии) не должен платить ничего
  — одна nil-проверка, как `TraceTap`.
- Конверсия рвёт prompt cache бэкенда (нет `cache_control`/`prompt_cache_key` в другой
  форме) → деградация `Cache hit` должна быть видна, иначе её припишут модели.
- `thinking` на конвертированном запросе ломает многоходовые tool-циклы (подпись Anthropic
  невоспроизводима) — поэтому по умолчанию отказ, а эмуляция не предусмотрена даже при
  `drop_params`: дроп должен быть решением оператора, принятым после preview.
- Строгие клиенты (opencode) валидируют каждое событие: эмиту только события своего диалекта,
  ничего «поверх».
- `config.cloneRule`/`Store.AddServer`/`UpdateServer` перечисляют поля — новое поле
  (`Protocols`, `Rule.Convert`, `FallbackEntry.Convert`, `JevCandidate.Convert`) обязано
  появиться там же, иначе админские правки его молча стирают (известная ловушка CLAUDE.md).
- `config.json` не перезаписывается (правило из CLAUDE.md), новая схема описывается в
  `example.config.json`.

## Решения, принятые в диалоге

1. **`max_tokens` — дефолтом, не отказом**: `settings.convert.max_tokens_default` (4096) +
   `rule.convert_max_tokens`. У LiteLLM нормальный источник — costs map по модели
   (`get_max_tokens(model)`), а `DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS` там лишь сеть последней
   защиты (`constants.py:412`); у нас такой таблицы нет, поэтому 4096 несёт нагрузку, которую у
   LiteLLM несёт таблица. Отсюда обязательный сосед фичи — переопределение на правиле.
2. **Способность объявляется, а не зондируется**: `probe: false` по умолчанию, зонд вне фазы 1.
   Инструмент наблюдения — `POST /admin/api/convert/preview` (сухой прогон, аналог
   `/utils/transform_request`).
3. **`X-Router-Convert: off` берём**, вопреки LiteLLM (там per-request переключатели едут в
   теле запроса): тело запроса конвертером переписывается, заголовок — единственное, что
   доживает до решения о конверсии.
4. **Потери = 400 по умолчанию**, дроп только по `drop_params` (LiteLLM-дефолт; в первой
   редакции этого spec'а было наоборот).
5. **Mid-stream fallback/регенерация — вне объёма** (см. «Не делаем»).
