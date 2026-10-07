# Jev Router — инструкция по настройке

Jev Router — режим автоматического выбора модели для каждого запроса. Правило с
`"router": "jev"` перед отправкой запроса спрашивает decision-модель (TypeSafe
System One, эндпоинт `/v1/systemone`) «какая модель из пула справится дешевле
всего», и отправляет запрос на выбранный бэкенд. Остальной пул остаётся цепочкой
fallback: при сетевой ошибке запрос идёт на следующую модель, сначала на более
сильные, потом на слабые.

Связанные заметки по дизайну: [jev-router.md](jev-router.md).

## Как это работает

```
клиент (model: "auto")
   │
   ▼
router: спрашивает decision-сервер  POST <jev.server_id>/v1/systemone
   │   questions: model (choice по пулу) + 3 score-вопроса (объяснимость)
   │              + security (noul, только при security: true)
   ▼
attempts = [выбранный] + более сильные по tier ↑ + более слабые ↓
   │
   ▼
retry (num_retries) → fallback по пулу → метрики + X-Router-Jev-* заголовки
```

Политика fail-open: если решение получить не удалось (сервер недоступен, таймаут,
мусорный ответ, выбор вне пула), запрос уходит на `default_tier` — роутинг никогда
не блокирует запрос. Единственное исключение — security-гейт (см. ниже).

## Шаг 1. Decision-сервер

Нужен бэкенд, отвечающий на `POST /v1/systemone`. Варианты:

**A. Локальная decision-модель (llama.cpp).** Нативные decision-модели
(laya / clef / openjev) llama.cpp отдаёт через `/v1/systemone` автоматически —
отдельных флагов не нужно:

```bash
llama-server -m /models/laya.gguf --alias laya --port 8081
# проверка:
curl -s http://127.0.0.1:8081/v1/models | grep -o decisions   # должен быть в output_modalities
```

**B. Hosted TypeSafe API.** Эндпоинт совместим с `/v1/systemone`, поэтому в
`url` сервера достаточно базового адреса API, а ключ — в `api_key` (роутер
шлёт его и в `Authorization: Bearer`, и в `x-api-key`).

Decision-сервер — обычный `Server` в конфиге, ничем не отличается от
генеративного бэкенда (могут использоваться `api_key`, `proxy`,
`openai_url`/`anthropic_url`).

## Шаг 2. Правило в config.json

Минимальное рабочее правило (полный пример — в `example.config.json`):

```jsonc
{
  "incoming_models": ["auto"],          // клиент шлёт model: "auto"
  "router": "jev",
  "jev": {
    "server_id": "jev-decision",        // decision-сервер из "servers"
    "model": "laya",                    // имя decision-модели (пусто = бэкенд решает)
    "default_tier": 1,                  // тир по умолчанию (fail-open, low-confidence)
    "min_confidence": 0.3,              // порог уверенности (0 = 0.3)
    "timeout_ms": 1500,                 // бюджет решения (0 = 1500)
    "state_max_chars": 8000,            // обрезка текста запроса для решения (0 = 8000)
    "cache_ttl": 60,                    // кэш одинаковых решений, сек (0 = выкл)
    "security": false,                  // Noul-гейт вредоносности (см. ниже)
    "question": "",                     // override вопроса choice (пусто = дефолтный)
    "candidates": [                     // пул: tier 0 = слабее/дешевле
      { "server_id": "local-llama",  "target_model": "claude-haiku-4-5",  "tier": 0,
        "description": "small and fast; simple questions, formatting, short edits",
        "context_window": 200000 },
      { "server_id": "fallback",     "target_model": "gpt-4o-mini",        "tier": 1,
        "description": "mid tier; everyday coding, summaries" },
      { "server_id": "primary-openai","target_model": "o3-mini",           "tier": 2,
        "description": "strong reasoning; multi-file refactors, tricky bugs" }
    ]
  },
  "enabled": true
}
```

Поля `jev`:

| Поле | По умолчанию | Смысл |
|------|--------------|-------|
| `server_id` | — | decision-бэкенд (POST `/v1/systemone`) |
| `model` | бэкенд решает | имя decision-модели в запросе |
| `question` | встроенный | текст choice-вопроса; правьте, если пул не про «код» |
| `candidates` | — | пул; `description` читает decision-модель, пишите по делу |
| `tier` | 0 | порядок пула: 0 = слабее/дешевле; политика работает по тирам |
| `target_model` | `server_id` | что подставить в `model` для этого бэкенда |
| `context_window` | — | для `context_window` правила в `/v1/models` (min по пулу) |
| `enabled` | true | выключенные кандидаты исключаются из пула |
| `min_confidence` | 0.3 | ниже — см. «Политика» |
| `default_tier` | 0 | якорь политики: fail-open и границы даунгрейда |
| `timeout_ms` | 1500 | таймаут решения; решение добавляет ~300–1000 мс к запросу |
| `state_max_chars` | 8000 | максимум текста запроса, который видит decision-модель |
| `cache_ttl` | 0 (выкл) | кэш решений для одинаковых запросов |
| `security` | false | Noul-гейт: блокировать вредоносные запросы |
| `images` | false | передавать картинки последнего user-сообщения decision-модели |

Замечания:

- `server_id`/`target_model`/`fallbacks` самого правила у jev-правила не
  используются — пул задаётся в `jev.candidates`.
- Идентификатор кандидата для decision-модели = `target_model` (пусто →
  `server_id`); при коллизиях добавляется `@server_id`.
- Правило участвует в retry/fallback/health/rate-limit/quota как обычное:
  `num_retries` работает по выбранному серверу, `context_window` правила —
  переопределяет min по пулу.

## Шаг 3. GUI

Правила → Add/Edit Rule → **Router: Jev**. Появляются: decision-сервер,
decision-модель, Min Confidence, Default Tier, Cache TTL, чекбокс Security gate,
вопрос и строки кандидатов (сервер, target model, описание, tier, context
window, enabled). В таблице правил jev-правило показано тирами-тегами и
`jev → decision-сервер`; в Recent Requests — колонка Jev (выбор + уверенность,
в тултипе причина/латентность/токены/скоры).

## Шаг 4. Проверка

```bash
curl -s http://localhost:8080/v1/models | jq '.data[] | select(.id=="auto")'
# → { "id":"auto", "router":"jev", "models":["claude-haiku-4-5","gpt-4o-mini","o3-mini"], "context_window":200000 }

curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" -D - -o /dev/null \
  -d '{"model":"auto","messages":[{"role":"user","content":"rename var x to y"}]}' | grep -i x-router-jev
# X-Router-Jev-Choice: claude-haiku-4-5
# X-Router-Jev-Confidence: 0.82
# X-Router-Jev-Ms: 341
# X-Router-Jev-Reason: jev
```

Ответ содержит реальную ответившую модель (не `auto`) — на jev-маршруте модель
не переписывается обратно, клиент видит, кто ответил.

## Политика выбора (reason-ы)

Порядок: решение → confidence-правила → context-правило → порядок обхода.

| `X-Router-Jev-Reason` | Что произошло | Куда уйдёт запрос |
|---|---|---|
| `jev` | уверенный ответ (≥ min_confidence) | выбранный кандидат |
| `cache` | решение взято из кэша | как в закэшированном решении |
| `unavailable` | decision-сервер недоступен/таймаут/мусор | `default_tier` |
| `unknown-choice` | ответ вне пула | `default_tier` |
| `low-confidence-no-downgrade` | уверенность ниже порога, ответ слабее дефолта | `default_tier` (не даунгрейдим) |
| `low-confidence-capped` | уверенность ниже порога, ответ сильнее дефолта+1 | `default_tier+1` (капаем) |
| `context-no-downgrade` | даунгрейд при контексте > 20k токенов | `default_tier` (беречь prompt cache) |
| `sticky` | tool-вершина агентной петли (без user-текста) — переиспользует последнее решение этой беседы | запиненный кандидат, решения нет |
| `off` | заголовок `X-Router-Jev: off`, `count_tokens`, или tool-вершина без известного решения беседы | `default_tier`, решения нет |
| `blocked` | security-гейт (см. ниже) | HTTP 400, ничего не отправлено |
| `no-pool` | все кандидаты выключены | HTTP 500 |

Практика: `default_tier` — ваша «безопасная» модель (обычно средняя). Дешёвые
модели включайте в tier ниже, сильные — выше; при низких скорингах/латентности
решений поднимайте `min_confidence` (агрессивнее сидим на default_tier) или
ставьте `cache_ttl`.

## Security gate (`security: true`)

К тому же вызову добавляется Noul-вопрос «насколько этот запрос вредоносен».
Ответ ≥ **0.95** → HTTP 400 `request blocked by security gate`, на бэкенды
ничего не уходит, в метрике `jev_reason: "blocked"`. Порог намеренно
экстремальный — ложные блокировки дороже пропущенных; гейт — грубый фильтр
намерения, а не модерация. Решения гейта тоже кэшируются (тот же запрос →
то же решение). Обход для доверенных клиентов: `X-Router-Jev: off`.

## Кэш решений (`cache_ttl`)

Ключ = decision-модель + вопрос + состав пула + текст запроса + картинки. Совпал — решение
переиспользуется (reason `cache`, нулевой расход токенов решения). Полезен при
веерных одинаковых запросах (батчи, ретраи клиента). Для чата с уникальными
репликами толку нет — оставляйте 0.

## Sticky-решение агентной петли

Запрос без user-текста (tool-вершина агентной петли: `/v1/responses` с
инкрементальным `input`, чистый `tool_result`) не получает решения заново —
роутер переиспользует последнее **настоящее** решение этой беседы (reason
`sticky`, нулевой расход токенов решения), а без него падает на
`default_tier` (reason `off`). Беседа определяется по `prompt_cache_key`
(Responses API), затем `metadata.user_id.session_id` (Claude Code на
messages), затем hash первого user-сообщения; ключ скоупится к входящей модели
правила. TTL запина — 30 минут, размер — 4096 бесед (свип при вставке).
Это поведение reference-роутеров: плагин Jev пинит tier беседы, LiteLLM пинит
сессию. Заголовок `X-Router-Jev: off` пин обходит.

## Картинки в решении (`images: true`)

По умолчанию decision-модель видит только текст последнего user-сообщения. С
`images: true` роутер достаёт картинки этого сообщения (OpenAI `image_url` и
Anthropic `source.base64`, data URL, максимум 8) и шлёт их в `images` запроса
`/v1/systemone`. Нужно, когда задача визуальная («что на скриншоте») и
decision-модель vision-совместима (llama.cpp: Clef/OpenJev/PPLX-decider;
обычные laya/kev картинки отвергнут 400). Дорого: изображения добавляют сотни
токенов к каждому решению. Картинки входят в ключ кэша решений.

## Стоимость и латентность

- +1 round-trip на каждый jev-запрос (локально ~300 мс warm, ~1 с cold;
  таймаут 1500 мс, дальше fail-open).
- Токены решения платные; видны в колонке Jev (токены) и `jev_tokens`.
- Смена модели ломает prompt cache бэкенда — поэтому даунгрейд запрещён выше
  ~20k токенов контекста.
- Стриминг не затрагивается: решение принимается до отправки запроса.

## Пресет: opus / sonnet / haiku (от сильной к быстрой)

Готовое трёхуровневое правило: haiku (tier 0, быстрая/дешёвая) → sonnet
(tier 1, дефолт) → opus (tier 2, сильная). Замените `server_id` на свои
серверы и `target_model` на реальные слаги бэкендов; `jev.server_id` —
decision-сервер из шага 1.

```json
{
  "incoming_models": ["auto"],
  "router": "jev",
  "jev": {
    "server_id": "jev-decision",
    "model": "laya",
    "min_confidence": 0.3,
    "default_tier": 1,
    "timeout_ms": 1500,
    "candidates": [
      {
        "server_id": "srv-haiku",
        "target_model": "claude-haiku-4-5-20251001",
        "tier": 0,
        "description": "fast and cheap; simple questions, rephrasing, formatting, short edits, one obvious command, factual lookups"
      },
      {
        "server_id": "srv-sonnet",
        "target_model": "claude-sonnet-4-6",
        "tier": 1,
        "description": "everyday engineering; implement a specified feature, tests, summaries, refactors with a clear shape, understood local bugs"
      },
      {
        "server_id": "srv-opus",
        "target_model": "claude-opus-4-6",
        "tier": 2,
        "description": "strongest reasoning; unknown-cause debugging, cross-module design, security, auth, concurrency, migrations, unusually large tasks"
      }
    ]
  },
  "enabled": true
}
```

Что даёт эта раскладка:

- `default_tier: 1` — sonnet якорь: без решения / при низкой уверенности /
  при `X-Router-Jev: off` запросы идут на него.
- Низкая уверенность: даунгрейд до haiku запрещён, апгрейд разрешён только до
  opus (это `default_tier+1`), т.е. sonnet ↔ opus остаются, haiku — только при
  уверенном ответе.
- Даунгрейд sonnet → haiku отключается выше ~20k токенов контекста
  (`context-no-downgrade`), беречь prompt cache.
- Fallback при сетевых ошибках: выбранная + остальные вверх (opus раньше
  sonnet/haiku).
- `description` — главный рычаг качества: decision-модель выбирает по ним,
  правьте под свои формулировки задач.

## Отладка

| Симптом | Куда смотреть |
|---|---|
| Всё время `unavailable` | жив ли decision-сервер: `curl <url>/v1/systemone`; лог `jev decision ... failed` |
| Модель выбирает неадекватно | улучшите `description` кандидатов; для не-кодовых пулов замените `question` |
| Слишком агрессивный даунгрейд | поднимите `min_confidence`, `default_tier` |
| Долгие ответы | `X-Router-Jev-Ms` в ответах / колонка Jev; уменьшите `state_max_chars`, поднимите `timeout_ms` |
| Нужна фиксированная модель на запрос | `X-Router-Jev: off` (тогда `default_tier`) |

## Пример: полный конфиг

См. `example.config.json` — сервер `local-llama` как decision-бэкенд
(`model: "laya"`), трёхуровневый пул, `cache_ttl: 60`, `security: true`.
