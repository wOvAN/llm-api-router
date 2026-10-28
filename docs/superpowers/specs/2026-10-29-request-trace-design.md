# Request Trace (on-demand full-text наблюдение и отладка)

> Статус: дизайн согласован в диалоге, реализации нет.
> Триггер — тумблер в GUI/API; текст живёт только в окне трассы; классификация — эвристика в роутере,
> работает всегда; стрим — сырые фреймы + склейка, в GUI компактно с разворачиванием.

## Цель

Сейчас про роутере видно **агрегат**: кольцо последних 100 запросов (`metrics.Store`) с токенами,
таймингами, `error_body` ≤ 4KB и jev-решением. Для наблюдения этого хватает, для отладки — нет:
не видно, **что именно** уехало на бэкенд (какой промпт, сколько tools, stream или нет) и **что
именно** вернулось (склейка стрима, finish_reason, имена SSE-событий). Отладка клиента
(Claude Code / opencode против локального llama.cpp) сейчас превращается в «угадай по токен-счётчику».

Нужен вторый плоскость: **wire-capture по требованию** — включил окно, получил полные тела запросов и
ответов по конкретным моделям/серверам, посмотрел в GUI, выключил, в RAM ничего не осталось лишнего.

## Не делаем (сознательно)

- Никакого персиста на диск (ни JSONL, ни ротации) — трасса живёт только в памяти и умирает с процессом.
- Никакой LLM-классификации запросов (решает эвристика; decision-модель и так тратится на роутинг).
- Никакого per-request триггера заголовком — только окно из админки (плюс дефолты из конфига).
- Никакого DevTools/EventStream-фронтенда, OTel, Prometheus-серий по трассе.
- Таймлайн-полоса с драг-фильтром и зумом (идея из harness-визуализаций) — отложена, не v1.

## Триггер и окно

`Settings.Trace` (дефолты из `config.json`, все нулевые = выключено):

```json
"settings": {
  "trace": { "enabled": false, "seconds": 300, "models": [], "servers": [] }
}
```

Живое состояние окна живёт в `trace.Store` (`enabled`, `expiresAt`, `models`, `servers`) и
**не пишется в `config.json`**: админский тумблер — одноразовое действие, а не настройка, и
перезапуск не должен оставлять включённый сбор промптов. `config.json`-блок задаёт только
начальное состояние при старте. `seconds: 0` = пока не выключишь вручную.

Решение «трассировать ли этот запрос» принимается один раз в начале запроса в `router.Handle`:
окно активно **и** (`models` пусто || incoming model ∈ `models`) **и** (`servers` пусто ||
первичный сервер ∈ `servers`). Решение не меняется mid-stream: запрос, начатый в окне, дотрассировывается
до конца, даже если окно истекло в середине генерации.

## Сбор (кран на существующем writer-цепочке)

Никакого второго RoundTripper и повторного чтения ответа: `metricsWriter` уже видит каждый байт
ответа (`Write` в relay-цепочке) и держит хвост 256KB ради usage-экстракции.

- `router.Handle`: если трасса взята, собирается `*trace.Capture` (ID, начало, метаданные, тело запроса
  после красной цензуры) и кладётся в `StreamProxyOptions`. `nil` = hot path не трогается.
- `proxy.StreamProxy`: `capture != nil` → writer-цепочка пишет каждый chunk в capture (с тем же
  усечением), а не только в хвостовой буфер. SSE-фреймы сохраняются **как шли**, побайтно: имена
  `event:`/`data:` видны, «что реально пришло кусками» видно.
- Бинарный ответ (не JSON и не `text/event-stream`: TTS WAV, PCM) в тело не кладётся —
  `<audio 4.3MB sha256:ab12…>` + `ResponseSize`. Мегаразмерные тела не буферизуются сверх лимита.
- `X-Router-Trace-Id: <id>` пишется в ответ **только** когда запись реально взята в кольцо.

Лимиты: 512KB на запрос и 512KB на ответ, усечение — префикс + маркер
`…[truncated N bytes]` + флаг `truncated: true` (префикс-и-флаг, как в harness; ничего не
«выбирается» из полей). Кольцо — 100 записей, как recent requests: `trace.New(100)` в `main.go`.
Худший RAM-случай ≈ 100 МБ, типичный — единицы МБ (промпты > 512KB редки, бинарники наружу не идут).

## Красная цензура (бинарники → заглушки)

`trace.RedactRequest(body, contentType)`:

- JSON: любой `data:image/...;base64,` URL и любое значение поля `image_url`/`source.data`,
  похожее на base64 → `<image 1.2MB sha256:ab12…>`.
- multipart: аудио-парти → `<audio 4.3MB sha256:ab12…>`; границы, заголовки парти и поле `model`
  остаются на месте (заимствованный скан из `proxy/multipart.go`: байтовый проход, форму не парсить).
- Текст промптов и ответов не режется: решение пользователя — режем только бинарники.
- Заголовок `Authorization` в трассу не попадает никогда (он не поле запроса и не сохраняется).

Хеши — `sha256` первых 16 байт, достаточно чтобы опознать тот же ли это файл/картинка.

## Классификация (всегда, 0 токенов)

`trace.Classify(path, method, body) ([]byte-скан, форму не парсить)`:

- `Class` — одна метка: `chat`, `responses`, `messages`, `embeddings`, `completions`, `speech`,
  `transcriptions`, `voices`, `tts`, `systemone`, `models`, `other`. Ядро совпадает с
  `router.apiEndpointFromPath`, так что метрики, трасса и GUI говорят одними словами.
- `Features []string` — вторичные признаки, ровно те, что считаются сканом: `stream`,
  `tools:<n>`, `tool_result`, `images:<n>`, `system:<chars>`, `reasoning`, `audio-upload`,
  `json:<bytes>`, `fallback`, `retry:<n>`.

`Class` ездит в `RequestMetric` (`class`), `Features` — в `RequestMetric.Features` (`features`,
omitempty). В `metrics.Store` появляется ключ `class:<метка>` рядом с существующими
`model` / `target:<server>:<model>` / `server:<id>`: механика агрегации и рендер таблиц GUI
переиспользуются без изменений, добавляется таблица «Summary by Request Class».

Заимствования из harness (дешёвые, в существующих таблицах):

- Кэш-хит считается суммой токенов по группе (`cached / (prompt + cached + cache_creation)`), а не
  средним по запросам, и отдельное состояние `cache miss` (0 закэшированных при ненулевом входе) —
  не прочерк.
- «Почему числа нет» подписывается в ячейке (`usage not reported`), а не оставляется пустой.

## Склейка стрима (компакт + разворачивание)

`trace.Digest(rawBody, contentType)` считает один раз при записи в кольцо:

- `Text` — склеенные дельты: OpenAI chat (`choices[].delta.content`, `reasoning_content`),
  OpenAI Responses (`response.output_text.delta`), Anthropic (`content_block_delta.text`,
  `thinking_delta`), audio.cpp (`transcript.text.delta`; для `speech.audio.delta` — только счётчик).
- `Events map[string]int` — гистограмма событий (`content_block_delta:412`, `message_stop:1`).
- `FinishReason` — из finish-события/chunk.

GUI показывает компактный вид (Text + гистограмма + finish), полный сырой поток — по разворачиванию.
Сырые фреймы при этом хранятся всегда (в пределах лимита): склейка — производная, а не замена.

## API

В стиле существующего `admin/handler.go`:

| Path | Метод | Что делает |
|------|-------|-----------|
| `/admin/api/trace` | GET | `{enabled, expires_at, models, servers, entries:[…]}` — список компактных записей (ID, ts, model, target, server, endpoint, class, features, status, latency, ttft, размеры, truncated, jev_choice) |
| `/admin/api/trace?id=` | GET | Одна запись целиком: `request`, `digest{text,events,finish_reason}`, `response` (сырое), `truncated` |
| `/admin/api/trace` | POST | `{enabled, seconds, models[], servers[]}` — включить/выключить окно, задать фильтр; отвечает текущее состояние |
| `/admin/api/trace/reset` | POST | Очистить кольцо (живое окно не трогает) |

## GUI

В существующем `admin/static/index.html`, теми же идиомами (модалки, теги, tooltips):

- Кнопка **Trace** в шапке: выключена / включена + обратный отсчёт окна.
- Панель трассы: фильтр по моделям/серверам, `seconds`, `Clear`, таблица записей (compact-колонки),
  клик по строке — раскрытие: **Request** (pretty JSON, длинные строки свёрнуты), **Response**
  (склейка + гистограмма событий), **Raw** (сырые байты/фреймы целиком).
- Recent Requests: пилюля класса; колонка кэш-хита — по сумме токенов, с явным `cache miss`.

## Ошибки и границы

- Сбор — best effort: любая ошибка редации/скана не должна влиять на запрос. Запись пишется с тем,
  что удалось собрать, плюс пометка; паники нет нигде на этом пути.
- Окно протухает лениво (проверка при записи/чтении), кольцо — overwrite по индексу, как recent
  requests; `reset` обнуляет кольцо, счётчик активных запросов не трогает.
- Отключённая трасса = одна ветка `if capture != nil` на пути прокси; поведение relay (per-event flush,
  heartbeats, loopDetector, frame buffering) не меняется.
- Конфиг: `Settings.Trace` добавляется в существующий `domain.Settings`. `RequestMetric` нигде не
  deep-копируется, так что новые поля туда добавляются свободно; deep-copy с перечислением полей
  (`config.cloneRule`, `Store.AddServer`) — это правила и серверы, их трасса не касается.

## Тесты

- `trace`: `TestClassify` (эндпоинты + признаки, multipart-audio, tools/images/system),
  `TestRedactRequest` (data-URL, anthropic `source.data`, multipart-заглушка, текст не тронут),
  `TestDigest` (OpenAI-стрим, Anthropic-стрим, Responses, audio SSE, JSON-ответ),
  `TestStoreRingAndWindow` (кольцо 100, overwrite, истечение окна, reset).
- `router`: трасса выключена → кольцо пустое; включена → запись с телом запроса и сырым ответом,
  `X-Router-Trace-Id` проставлен; фильтр по модели/серверу; стрим → digest с текстом и гистограммой.
- `admin`: GET/POST/reset `/admin/api/trace`, `?id=` отдаёт тело.
- `metrics`: `class:*` в сводках, кэш-хит по сумме токенов + `cache miss`.

## Файлы

`trace/store.go`, `trace/capture.go` (+ тесты) — новые.
`router/router.go` (решение о трассе, digest, record), `proxy/proxy.go` (capture в opt-ах, write-кран,
бинарный ответ → заглушка, `X-Router-Trace-Id`), `domain/metric.go` (`Class`, `Features`),
`domain/config.go` + `config/store.go` (`Settings.Trace`), `metrics/metrics.go` (`class:*`),
`admin/handler.go` (эндпоинты), `admin/static/index.html` (панель), `example.config.json`,
`CLAUDE.md` + `AGENTS.md`.
