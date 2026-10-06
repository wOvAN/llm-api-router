# Jev Router: vision-роутинг (картинки → vision-модель)

> Статус: план, не реализовано. Зависимости: фаза 1–2 jev-роутера
> (`docs/jev-router.md`), извлечение картинок `jevRequestImages` (уже есть,
> для `images: true`).

## Проблема

Запрос с картинками («что на скриншоте») в jev-пуле может попасть на слепую
модель — decision-модель про способности кандидатов ничего не знает, а бэкенд
без vision отвергнет запрос 4xx. Нужна ось «кандидат видит картинки»: пул перед
решением сужается до vision-способных моделей.

Способности берём **автоматически из бэкенда, где он их отдаёт, и руками — где
нет** (приоритет: явный флаг > автоопределение > неизвестно).

## Источники способностей (проверено по `.info/`)

| Бэкенд | Поле в `/v1/models` | Надёжность |
|---|---|---|
| llama.cpp | `architecture.input_modalities` ∋ `"image"` — ставится только при реальном `--mmproj` (`server-models.cpp:567-587`); там же `output_modalities: ["decisions"]` у decision-моделей | да |
| vLLM (наша версия) | карточка = `id/max_model_len/root/permission` — модальностей нет (в новых версиях есть `supported_modalities`) | нет |
| Strata | карточки как llama.cpp (`/v1/models` llama.cpp-совместим) | да |
| Anthropic API | `id/display_name/created_at` — нет | нет |

## Дизайн

### 1. Domain: `JevCandidate.Vision *bool`

```go
// Vision: nil = неизвестно (решает автоопределение), true = видит картинки,
// false = слепа (явно; из пула image-запроса исключается).
Vision *bool `json:"vision,omitempty"`
```

- `config.cloneRule` копирует struct целиком — `*bool` копируется как указатель;
  deep-copy слайса кандидатов уже есть, указатель на bool разделять безопасно
  (значения никто не мутирует) — оставить как есть.
- GUI шлёт `true/false/absent`; absent = nil.

### 2. Router: pre-filter пула (`router/jev.go`)

В `jevAttempts`, после `pool := j.EnabledCandidates()`:

```go
hasImages := len(jevRequestImages(body)) > 0   // извлекаем всегда, дёшево
if hasImages {
    if vis := filterVision(pool); len(vis) > 0 {
        pool = vis                              // режем явных слепых
    }                                        // ни одного vision-кандидата — fail-open: весь пул
}
```

- `filterVision`: оставляет `Vision == nil || *Vision == true` (nil щадим —
  автоопределение могло ещё не успеть).
- Пусто после фильтра → пул не сужаем (fail-open, как везде; бэкенд сам
  вернёт 4xx, клиент увидит).
- `ids`, `criteria`, `jevTierIndex`, attempts, low-confidence-политика —
  работают по отфильтрованному пулу без изменений (якорь `default_tier`
  берётся из выживших тиров, `nearest above/below` уже умеет).
- Кэш-ключ картинок уже не содержит различия «текст vs текст+картинки»?
  содержит — `images` в ключе с `b9db671`. Ничего не добавляем.
- Причина в метрике не меняется (фильтр — не исход решения); для
  наблюдаемости — `log.Debugf("jev: pool filtered to %d vision candidates")`.

### 3. Автоопределение (ленивое, с кэшем)

Новый файл `config/capabilities.go` (рядом с health/ratelimit — тот же паттерн
фонового/ленивого знания о серверах):

```go
// CapabilitiesCache — кэш «видит ли сервер модель картинки» по данным /v1/models.
type CapabilitiesCache struct{ ... }
func (c *CapabilitiesCache) Vision(server *domain.Server, model string) (vision bool, known bool)
func (c *CapabilitiesCache) Probe(server *domain.Server)   // GET <url>/v1/models, TTL 10min
```

- Ключ `serverID + "\x00" + targetModel`; значение `vision *bool, fetched time.Time`.
- `known == false` (бэкенд не отдаёт модальности / сервер недоступен / TTL истёк
  и пробник не удалось) → роутер трактует как nil (щадим).
- Вызов из `jevAttempts` **только при hasImages** (горячий путь без картинок
  не трогает): для каждого кандидата с `Vision == nil` — кэш-хит? иначе
  однократный синхронный `Probe` с коротким таймаутом (2s) на сервер,
  отрицательный ответ тоже кэшируется (vLLM/Anthropic не будем долбить).
- Парсинг: `data[].id == targetModel (или alias)` →
  `architecture.input_modalities ∋ "image"`; vLLM-поле `supported_modalities`
  читаем заодно (новые версии). Переиспользовать разбор из `admin`
  (server models fetch) — вынести общий хелпер в `proxy` (там уже есть
  транспорт/авторизация `TransportFor`).
- Пробник ходит тем же путём, что health-checker (тот же транспорт, api_key).

### 4. GUI (`admin/static/index.html`)

- Строка кандидата: чекбокс `vision` → три состояния недостижимы чекбоксом,
  поэтому select: `auto (—) / vision / blind` (""/true/false).
- Автозаполнение при выборе модели в строке: `fillModelSelect` уже тянет
  `/admin/api/servers/:id/models` — прокинуть карточки наружу (он их и так
  получает): если у выбранной модели `architecture.input_modalities ∋ image`
  → выставить select в `vision` (только если стоит `auto`, руками поставленное
  не перетирать).
- Preset-кнопку не трогаем (vision-статусы — свойство бэкенда, не шаблона).

### 5. Конфиг (пример)

`example.config.json`: у jev-правила у одного кандидата `"vision": true`,
остальные без поля (= auto).

## Тесты

- `router/jev_test.go`:
  - `TestJevAttemptsVisionFilter` — картинка в запросе, пул haiku(vision:false)/
    opus(vision:true) → attempts начинаются с opus, слепой только после
    (в fallback-цепочке остаётся? нет: фильтр до решения, слепой вне пула —
      ожидаем пул из vision-кандидатов);
  - картинка + ни один кандидат не помечен и кэш не знает → пул без изменений
    (fail-open);
  - без картинки → фильтр не срабатывает (слепой в пуле);
  - `Vision: nil` + кэш сказал `true` → остаётся в пуле; кэш сказал `false` →
    удалён (нужен тестовый хук: in-process httptest-бэкенд /v1/models с
    `architecture.input_modalities`).
- `config/capabilities_test.go` — парсинг карточек (llama.cpp-форма,
  vLLM-форма, отсутствие поля → known=false), TTL-протухание, кэш отрицательного.
- `proxy` — общий хелпер фетча /v1/models (если выделяется).

## Документация

- `docs/jev-router-guide.md`: секция «Vision-роутинг» (флаг, авто, приоритет
  явного флага, fail-open без vision-кандидатов), строка таблицы полей.
- `CLAUDE.md`/`AGENTS.md`: jev-буллет + `Vision *bool`, pre-filter,
  `CapabilitiesCache`.

## Порядок работ

1. domain `Vision *bool` + pre-filter + тесты (ручной режим работает уже тут).
2. `CapabilitiesCache` + парсинг карточек + подключение в `jevAttempts` + тесты.
3. GUI: select auto/vision/blind + автозаполнение из лукапа.
4. example-конфиг, гайд, CLAUDE/AGENTS.

Оценка: 1 — ~40 строк, 2 — ~120, 3 — ~30, 4 — доки.

## Открытые вопросы

- Порождать ли `context-no-downgrade`-подобную защиту при смене модели из-за
  фильтра? (image-запросы редки, prompt cache на них и так тёплый только у
  выбранной модели — по умолчанию нет.)
- Видео (`input_modalities: video`, llama.cpp «videos not supported» в
  systemone) — не трогаем до реального запроса.
