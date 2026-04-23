# Steel AI MVP — HSLA Pipeline Steels

Рабочий пакет для построения демо-версии AI-платформы для дизайна трубных HSLA-сталей за 6-8 недель.

## Ключевое решение архитектуры

Вместо «Orchestrator + 7 executive агентов + 6 скиллов», как в большом плане, здесь — **упрощённая система**, нацеленная на демо:

```
┌──────────────────────────────────────────────────────────┐
│  Streamlit UI (для демо) / FastAPI (для API-пилотов)     │
└──────────────────────────────────────────────────────────┘
                          ↕
┌──────────────────────────────────────────────────────────┐
│  Orchestrator (Claude Sonnet) с Decision Log + Memory    │
│   └─ вызывает агентов, сохраняет все решения             │
└──────────────────────────────────────────────────────────┘
                          ↕
┌──────────────────────────────────────────────────────────┐
│  6 executive-агентов (Claude Sonnet каждый):             │
│  DataCurator → FeatureEng → Trainer → InvDesigner        │
│               → Validator → Reporter                     │
│  + Critic (проверяет каждого)                            │
└──────────────────────────────────────────────────────────┘
                          ↕
┌──────────────────────────────────────────────────────────┐
│  Skills (детерминированный Python-код, без LLM):         │
│  data_acquisition · preprocessing · feature_engineering  │
│  model_training · inverse_design · validation · reporting│
└──────────────────────────────────────────────────────────┘
                          ↕
┌──────────────────────────────────────────────────────────┐
│  Pattern Library (анти-паттерны, проверяются Critic)     │
│  Decision Log (история решений проекта, structured JSON) │
│  Data (NIMS-датасет + кэш)                               │
└──────────────────────────────────────────────────────────┘
```

## Почему это работает на вашем профиле

Вы — fullstack-разработчик. Наша автоматизация держится на трёх столпах:

**1. Pattern Library заменяет tacit knowledge частично.**
Я закодирую в структурированный checklist те 50-80 анти-паттернов, про которые senior-инженер знает из опыта: data leakage, calibration drift, distribution shift, physical inconsistency, target contamination и т.д. Critic-агент проверяет каждую итерацию против этого checklist.

Это не полная замена senior-инженеру, но ловит 60-70% типовых ошибок. Для MVP и первых демо — достаточно.

**2. Decision Log заменяет институциональную память.**
Каждое техническое решение Orchestrator записывает в structured JSON: что решил, почему, с какими альтернативами. Через 3 месяца вы можете восстановить reasoning любого решения. Это компенсирует ограничение context window LLM.

**3. Human-in-the-loop checkpoints на критических моментах.**
В коде явно отмечены места, где автоматизация опасна — и именно там система останавливается и просит вашего решения. Вы — человек в цикле, но цикл короткий.

## План 6-8 недель

### Неделя 1-2: Data pipeline + baseline
- Скачивание и парсинг NIMS HSLA dataset
- Preprocessing pipeline
- Feature engineering (CEV, Pcm, CEN + ratios)
- Baseline XGBoost с правильной валидацией
- **Checkpoint 1:** убедиться, что R² > 0.75 на held-out set

### Неделя 3-4: Forward model + uncertainty
- Hyperparameter tuning через Optuna
- Quantile regression для uncertainty
- Calibration check, out-of-distribution detection
- SHAP для explainability
- **Checkpoint 2:** прогнозы физически разумны на 10 референсных компонентах

### Неделя 5-6: Inverse design + validator
- NSGA-II optimization pipeline
- Validator с физ.ограничениями (Pattern Library)
- Orchestrator соединяет всё в pipeline
- **Checkpoint 3:** для запроса "K60 + Сэкв ≤ 0.43" система возвращает ≥ 5 валидных кандидатов

### Неделя 7-8: UI + demo
- Streamlit-интерфейс для демо
- Decision log visualization
- Sales materials (pitch deck, pilot proposal)
- Docker Compose для локального запуска
- **Checkpoint 4:** полный end-to-end запуск за 1 команду

## Что дальше (после 8 недель)

Этот MVP — **демо, а не продукт**. После 8 недель у вас есть:
- Работающая система для демонстрации инвесторам
- Pitch deck и sales materials
- Доказательство работоспособности подхода
- Основа для масштабирования до пилота

Для превращения в пилот-ready продукт понадобится:
- Data ingestion под конкретного клиента (MES integration)
- Multi-tenancy, auth, biling
- Production-grade deployment
- Полный audit log для compliance

Это — следующая фаза, 3-6 месяцев после MVP.

## Как использовать этот пакет

```bash
# 1. Clone и setup
git clone <your-repo>
cd steel-ai-mvp
pip install -r requirements.txt

# 2. Скачать NIMS данные (через скилл)
python scripts/bootstrap.py --step data

# 3. Запустить pipeline end-to-end
python scripts/run_pipeline.py --target yield_strength

# 4. Поднять UI для демо
streamlit run app/frontend/app.py

# 5. Запустить полный стек локально
docker-compose up
```

## Структура

```
steel-ai-mvp/
├── README.md                    ← этот файл
├── requirements.txt
├── docker-compose.yml
├── agents/                      ← системные промпты агентов
│   ├── orchestrator/
│   ├── data_curator/
│   ├── feature_eng/
│   ├── model_trainer/
│   ├── inverse_designer/
│   ├── validator/
│   ├── reporter/
│   └── critic/                  ← проверяет других агентов
├── skills/                      ← детерминированный Python-код
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── model_training/
│   ├── inverse_design/
│   ├── validation/
│   └── reporting/
├── pattern_library/             ← checklist анти-паттернов
│   ├── data_issues/
│   ├── model_issues/
│   └── production_issues/
├── decision_log/                ← структурированная память проекта
├── app/
│   ├── backend/                 ← FastAPI
│   ├── frontend/                ← Streamlit
│   └── tests/
├── data/                        ← NIMS datasets (gitignored)
├── scripts/                     ← запускаемые скрипты
├── sales/                       ← pitch deck, pilot proposal
└── docs/                        ← документация
```

## Бюджет проекта

- **Claude API:** ~€200-500 на все 8 недель при активной разработке с агентами
- **Compute:** локальный, CPU достаточно для MVP
- **Данные:** NIMS бесплатны
- **Ваше время:** 30-40 часов в неделю фокус-работы
