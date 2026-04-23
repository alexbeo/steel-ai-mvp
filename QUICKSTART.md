# QUICKSTART

## Запуск за 5 минут

```bash
# 1. Install
pip install -r requirements.txt

# 2. Smoke test (~1-2 минуты, проверяет всё)
PYTHONPATH=. python scripts/smoke_test.py

# 3. Web UI (Streamlit)
PYTHONPATH=. streamlit run app/frontend/app.py
# Открыть http://localhost:8501

# Или через Docker (одна команда)
docker-compose up --build
```

## Что попробовать в UI

### Tab «🤖 Обучение модели»
1. Выбрать target property (по умолчанию `yield_strength_mpa`)
2. Поставить Optuna trials = 20 (для скорости) или 50+ для качества
3. Нажать «Обучить модель»
4. Подождать 1-3 минуты
5. Посмотреть на отчёт Critic — он обязательно поймает calibration issue (это by design)
6. Посмотреть Feature importance chart

### Tab «🎯 Дизайн сплава»
1. После обучения — вернуться сюда
2. Задать target: σт 485-580 МПа
3. CEV max = 0.43, Pcm max = 0.22
4. Нажать «Запустить дизайн» (10-60 секунд в зависимости от population size)
5. Посмотреть топ-5 кандидатов:
   - Каждый раскрывается по клику
   - Видно химию, режим, прогноз с CI
   - Валидационный статус с emoji (✅⚠️❌)

### Tab «📊 Прогноз»
1. Ввести конкретный состав (например, 09Г2С: C=0.10, Mn=1.5, Si=0.5)
2. Получить прогноз с uncertainty
3. Проверить OOD flag

### Tab «📚 История»
1. Посмотреть все решения проекта в Decision Log
2. Фильтровать по фазе
3. Раскрыть любое — увидеть reasoning, альтернативы, outcome

## Troubleshooting

**Ошибка `ModuleNotFoundError: No module named 'app'`**
→ Запускать с `PYTHONPATH=.` или использовать docker-compose

**Ошибка `No models found` в UI**
→ Сначала обучить модель через Tab «Обучение»

**Streamlit не открывается**
→ Проверить, что порт 8501 свободен. Или запустить на другом: 
`streamlit run app/frontend/app.py --server.port 8502`

**Обучение слишком медленное**
→ Уменьшить Optuna trials до 10-20 для dev, 50-100 для production

## Что дальше

1. **Под ваш датасет:** замените synthetic generator в `data_curator.py` на парсер вашего формата (см. маппинг в `skills/steel-data-ingestion/references/mapping_template.md`)
2. **Под ваш класс сталей:** обновите `PIPE_HSLA_FEATURE_SET` в `feature_eng.py` и physical bounds в `data_curator.py`
3. **Tenant-specific validation:** добавьте client config в `validator.py` (доступные элементы, достижимая чистота S/P, тип АКОС)
4. **Real NIMS data:** замените `generate_synthetic_hsla_dataset` на загрузку NIMS MatNavi через их API

## Структура проекта

```
steel-ai-mvp/
├── agents/              # System prompts для LLM-агентов
├── app/
│   ├── backend/         # Главный код: engine + 6 executive agents
│   └── frontend/        # Streamlit UI
├── data/                # NIMS/synthetic parquet (gitignored)
├── decision_log/        # SQLite memory проекта
├── models/              # Обученные модели (gitignored)
├── pattern_library/     # Anti-patterns checklist для Critic
├── reports/             # Сгенерированные HTML-отчёты
├── sales/               # Pitch deck, pilot proposal templates
└── scripts/             # CLI: run_pipeline.py, smoke_test.py
```
