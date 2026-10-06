# JEV Router (авто-роутинг по решению decision-модели)

> Статус: **фазы 1–2 реализованы** (domain, `proxy/jev.go`, `router/jev.go`, метрики, GUI,
> example-конфиг, тесты, docs). Фаза 2: Noul security-гейт (`security: true`, порог 0.95 → 400),
> 3 score-вопроса (`RequestMetric.JevScores`), кэш решений (`cache_ttl`), context_window = min по
> пулу (кандидатам добавлен `context_window`), «не даунгрейдить при контексте > 20k»
> (`context-no-downgrade`). Отклонения от плана: метрики — без изменений в
> `metrics/prometheus.go` (решение живёт в `RequestMetric`-полях, Prometheus-серии для
> jev не заводились), `JevReason` добавлен к списку полей, `question` — override промпта.

Прототип: OpenRouter `model: "typesafe/jev-router"` + `plugins:[{id:"jev-router", models[], allowed_models[], excluded_models[]}]`.
Jev (System One, TypeSafe) читает диалог, оценивает тип/сложность задачи и «насколько сильнее модель
поможет» → роутер берёт **самый дешёвый кандидат из пула, который соответствует порогу**. Фактическая
модель — в ответном `model`. Списки пула: exact slug / dated revision / wildcard (`anthropic/*`) /
alias (`~author/family-latest`), до 1024 паттернов; include-список, не совпавший ни с чем, игнорируется
(`list_fallback:"models_ignored"`), exclusions применяются всегда, списки могут срезать тир
(`list_tier_cap`) и убрать советника (`max_fallback:"deep"`), пустой пул → `404`.
Наблюдаемость: `openrouter_metadata.pipeline` с `resolved_models` в порядке fallback.

Wire-контракт (`.info/jev`): `POST /v1/systemone`
`{state, model, questions:{name:{type:"noul"|"choice"|"score", instructions, criteria}}}` →
`{model, answers:{name:{type, noul | choice+probabilities+confidence | score+legend+probabilities}}, usage}`.
Лимиты: 255 опций choice, 10 уровней score, ~64k токенов. Роутер уже **проксирует** этот эндпоинт
(правило `systemone`); фаза 1 делает роутер его **клиентом**.

## Фаза 1 — ядро (один Choice по пулу, fail-open)

1. `domain/rule.go`: `RoutingRule.Router string` (`""` = static, `"jev"`) + `*JevRouter`:
   ```go
   type JevRouter struct {
       ServerID      string         `json:"server_id"`       // decision-бэкенд (/v1/systemone)
       Model         string         `json:"model"`           // "jev-latest" / "laya"
       Candidates    []JevCandidate `json:"candidates"`      // пул
       MinConfidence float64        `json:"min_confidence"`  // default 0.3
       DefaultTier   int            `json:"default_tier"`    // Jev недоступен/низкая уверенность
       TimeoutMs     int            `json:"timeout_ms"`      // default 1500
       StateMaxChars int            `json:"state_max_chars"` // default 8000
   }
   type JevCandidate struct{ ServerID, TargetModel, Description string; Tier int; Enabled *bool }
   ```
2. `proxy/jev.go` (new): клиент SystemOne — тело запроса, `http.Client` с `TransportFor(srv.ProxyURL())`
   (per-server proxy уже есть), заголовки `Authorization` + `x-api-key` (как в proxy), таймаут, разбор
   `answers.<name>.{choice,confidence,probabilities}`; переиспользовать `getField`/`floatToFloat64`.
3. `router/jev.go` (new): `state` = `{request: <последний user-текст>, session:{current_model,
   context_tokens}, environment:{available_models: id пула}}` (форма CLI-роутера), извлечение через
   `json.Unmarshal` (только для jev-правил); критерии Choice = `description` кандидата;
   policy: unknown choice / ошибка / таймаут → `DefaultTier`; `confidence < min_confidence` → запрет
   даунгрейда и кап по текущему тиру; выбор → `attempts = [выбранный] + остальные по tier asc`,
   вся существующая retry/fallback/health/quota-механика не трогается.
4. `router/router.go`: хук после `GetRuleByModel` (до построения attempts); `responseModel = targetModel`
   (клиент видит ответившую модель — поведение уже есть для fallback); заголовки
   `X-Router-Jev-Choice`, `-Confidence`, `-Ms`; пропуск по `X-Router-Jev: off`.
5. Метрики: `RequestMetric` + `JevChoice/JevConfidence/JevLatencyMs/JevModel`, колонки в Recent
   Requests, `metrics/metrics.go` + `prometheus.go`.
6. GUI `admin/static/index.html`: в диалоге правила select `Router` (static/jev), jev-сервер,
   decision-модель, `min_confidence`, `default_tier`, редактор кандидатов по образцу `addFallbackRow()`.
7. `example.config.json`: пример jev-правила; тесты `proxy/jev_test.go` (wire-формат на httptest),
   `router/jev_test.go` (state, выбор тира, low-confidence, отказ Jev, стриминг);
   `CLAUDE.md` + `AGENTS.md`.

## Фаза 2

`security` Noul-гейт (block → 400, пороги 0.95), 3 `score`-вопроса для объяснимости + отчёт как у
`jev-explain`, кэш решений (hash state, TTL), `context_window` в `/v1/models` = min по пулу,
правило «не даунгрейдить при контексте > 20k» (prompt cache).

## Фаза 3

Glob-списки `models`/`excluded_models` вместо явного пула + `list_tier_cap`, инъекция
`reasoning_effort`/`thinking` по решению, двухстадийный отбор для больших пулов (coarse Top-K 8 →
final Choice), images в state (OpenJev vision), cost/savings (нужны цены на `Server`).

## Цена

+1 round-trip на запрос (281 ms локально, 1500 ms таймаут + 1 retry как в CLI), токены решения
платные, смена тира ломает prompt cache. Стриминг не трогается — решение до отправки.

## Находка попутно

`.info/jev/jev-router/src/explain.mjs:30` читает `answers.model_tier.choice`, а весь остальной код
пишет `answers.model.choice` — отчёт «Recommended tier» на живой модели даст `UNKNOWN`; моки в тестах
повторяют ту же опечатку.
