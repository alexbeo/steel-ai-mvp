---
title: AI integration roadmap — 5 направлений
date: 2026-04-25
status: in-progress
---

## Контекст

После reframing (см. `2026-04-25_project_purpose_reframe.md`) ключевая цель проекта зафиксирована как «AI находит паттерны, неочевидные для человека». Текущий pipeline — это classical ML без AI-driven discovery components. Нужно расширять.

## Текущий «AI-уровень» системы — честная позиция

| Компонент | Тип | «AI-depth» |
|---|---|---|
| XGBoost composition→property | Supervised ML, gradient boosting | Низкий — predict только в рамках задачи |
| NSGA-II inverse design | Эволюционный алгоритм | Нулевой ML |
| GaussianMixture OOD | Density estimation | Нулевой ML в современном смысле |
| Pattern Library | Hand-coded Python rules | Нулевой ML |
| LLM-Critic (опц., через ANTHROPIC_API_KEY) | Claude Sonnet наблюдатель | Низкий — комментирует готовое, не генерирует insight |

То есть «AI» в системе ограничен XGBoost'ом — и тот находит только предписанные паттерны.

## 5 направлений для развития AI-capabilities

Все 5 направлений согласованы с пользователем для последовательной реализации. Эффект каждого проверяется отдельно перед переходом к следующему.

### A2 — Automated hypothesis generation [next, in-progress]

**Что:** LLM получает обученную модель + статистику датасета + feature importance + sample predictions, формулирует **testable hypotheses** уровня:
> «Модель сильно опирается на normalizing_temp при низком C — возможно, в этом диапазоне работает другой механизм закалки. Предлагаю эксперимент: фиксировать C=0.2, варьировать normalizing_temp от 850 до 950 °C с шагом 10.»

Каждая hypothesis структурирована: statement / rationale / proposed_experiment / expected_outcome.

**Почему первая:**
- Самый wow-effect-per-неделю.
- Никаких архитектурных изменений; LLM-Critic infrastructure уже есть в `app/backend/critic_llm.py`, можно расширить или скопировать pattern.
- Применимо к любой обученной модели — class-agnostic.
- Verifiable: на Agrawal NIMS модели LLM должен предложить ≥3 hypotheses, ≥1 из которых нетривиальная (не просто «больше углерода → выше fatigue»).

**Estimated effort:** 1 неделя.

### A1 — LLM-driven feature discovery [next after A2]

**Что:** LLM получает датасет + текущий feature_set + описание задачи. Предлагает 5-10 новых физически-осмысленных features (interaction terms, ratios, log-transforms, threshold-binarizations). Тестируем какие реально подняли R².

**Почему вторая:**
- Зависит от той же LLM-инфраструктуры что A2.
- Реальный numerical uplift в R² — proof of value.
- Class-agnostic.

**Verifiable:** на Agrawal NIMS feature discovery должен предложить ≥3 features, тестовый прогон должен показать R² не хуже исходного, желательно лучше на 0.005+.

**Estimated effort:** 1 неделя.

### B1 — Symbolic regression [3rd]

**Что:** Использовать PySR / SymbolicRegressor для извлечения **аналитических формул** из данных. Не «model predicts 550 МПа», а «yield ≈ 380 + 1200·C + 80·√Mn − 0.4·tempering_T». Эти формулы металлург может проверить против Hall-Petch / Orowan / классической physical metallurgy.

**Почему третья:**
- Особенно ценно для academic пользователя (см. profile).
- Требует тяжёлой dependency (PySR — Julia backend), интеграция нетривиальна.
- Найденные формулы могут раскрыть **новые** эмпирические законы.

**Verifiable:** на Agrawal должна найтись формула с R² ≥ 0.85 на test, осмысленная для металлурга (не «random combination of variables»).

**Estimated effort:** 2-3 недели.

### B2 — Active learning loop [4th]

**Что:** Bayesian optimization. Система предлагает **следующий эксперимент с максимальной information gain**: «composition X, cooling rate Y, ожидаемое снижение модельной uncertainty 35 %».

**Почему четвёртая:**
- Самое реальное business-value для R&D engineer (профиль 1).
- Требует UI-flow «эксперимент → запись результата → re-train» — больше работы чем previous.
- Зависит от calibrated uncertainty, которое мы только что довели до 0.89-0.92 (conformal).

**Verifiable:** на Agrawal с искусственно retained 50-record subset, active learning должен достигать R² ≥ 0.95 за меньшее число итераций чем random sampling.

**Estimated effort:** 3-4 недели.

### A3 — Anomaly explanation [5th]

**Что:** Когда OOD-детектор флагает рецепт, LLM объясняет почему: «эта композиция отличается от обучающих по mn/c ratio = 5.2, у обучающих ratio в диапазоне 2-4. Высокий Mn без пропорционального C обычно даёт austenite-retention; модель к этому не готова».

**Почему пятая (последняя):**
- Polish-level, маленький wow.
- Зависит от LLM-инфраструктуры (та же что A2/A1).
- Простой, можно сделать быстро после остального.

**Verifiable:** на 5 искусственно сгенерированных OOD compositions LLM должен дать содержательное объяснение (не «out of distribution»).

**Estimated effort:** 0.5 недели.

## Общий timeline (грубая оценка)

| Неделя | Направление | Deliverable |
|---|---|---|
| 1 | A2 | Hypothesis generator + UI + verification на Agrawal |
| 2 | A1 | Feature discovery + verification |
| 3-4 | B1 | Symbolic regression |
| 5-7 | B2 | Active learning loop |
| 7.5 | A3 | Anomaly explanation |
| 8 | — | Consolidation, write-up of findings |

Любой пункт может расшириться при сложностях; verification gate между ними даёт возможность остановиться или повернуть.

## Verification model (для каждого пункта)

После реализации каждого направления — три проверки:

1. **Технический smoke test** — feature не ломает существующий pipeline (smoke_test.py + unit tests pass).
2. **Empirical value test** — есть ли реальный numerical uplift / non-trivial output (специфично для каждого пункта, см. описание).
3. **Commit + rollback point** — атомарный commit, можно reverter точечно если AI-feature окажется bullshit.

## Fallback / pivot triggers

Если на каком-то пункте оказывается, что:
- LLM генерирует только тривиальные / wrong hypotheses (A2)
- Найденные features не двигают R² (A1)
- Symbolic regression застревает на random combinations (B1)
- Active learning не побеждает random sampling (B2)

— останавливаемся, документируем negative result, обсуждаем с пользователем что делать дальше.

Negative result сам по себе тоже value — пишем в этом же `docs/discussions/` как ретроспективу.

## Decision points для пользователя по ходу реализации

- После A2 verification: довольны ли качеством hypotheses? → Идём в A1 / меняем порядок / останавливаемся.
- После A1 verification: реален ли uplift R²? → Аналогично.
- После B1: формулы интерпретируемы? → Аналогично.
- После B2: active learning работает? → Аналогично.
- После A3: общий polish финиширует MVP-фазу.

## Source

Эта roadmap — результат discussion 2026-04-25 (assistant Claude Opus 4.7 + alex), после reframing проекта с sales-tool на MVP с фокусом на AI-discovery.
