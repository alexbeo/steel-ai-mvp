# Pattern Library — Anti-patterns в ML для металлургии

Это главный артефакт системы. Здесь собраны типовые ошибки ML-инженерии в применении к металлургии сталей. Critic-агент проверяет каждое решение против этого checklist.

**Зачем это нужно:** вы не senior ML-инженер и не senior металлург. Этот документ кодирует tacit knowledge, чтобы система могла проверять свою же работу против накопленного опыта отрасли.

**Как использовать:** Critic-агент в своём системном промпте имеет ссылку на этот файл. На каждой итерации (после каждого значимого действия) Critic проверяет результат против релевантных паттернов. При подозрении — эскалирует к вам через human-in-the-loop checkpoint.

**Как расширять:** каждый раз, когда вы сами находите ошибку (или клиент указывает) — добавляйте сюда. Через 6-12 месяцев проект накопит сотни таких паттернов, и система станет заметно умнее.

---

## Таксономия

- **data_issues/** — проблемы с данными: аномалии, утечки, сдвиги, качество
- **model_issues/** — проблемы с моделями: overfitting, калибровка, некорректная валидация
- **production_issues/** — проблемы в продакшене: drift, latency, безопасность

---

## Главные 20 паттернов для MVP

### D01: Target leakage через derived feature

**Проблема:** В feature set случайно попадает фича, вычисленная из target (прямо или косвенно).

**Пример в металлургии:** В обучающем наборе есть колонка "грейд стали" (например, "09Г2С"). Грейд присваивается **после** испытаний, на основе свойств. Использовать грейд как фичу — classic target leakage.

**Как детектировать:** 
- R² на train = R² на val (идеально совпадает) — подозрительно
- Feature importance сконцентрировано на одной фиче > 70%
- При удалении "подозрительной" фичи — метрика падает катастрофически

**Проверка Critic:**
```
Q: Есть ли в feature set колонки, которые заполняются после получения target?
Q: Есть ли постобработка плавки (тип термообработки, классификация), которая зависит от свойств?
```

**Что делать:** удалить, переобучить, сравнить метрики.

---

### D02: Duplicated heats через варианты написания

**Проблема:** Плавка "Я-47832" записана трижды: "Я-47832", "Я 47832", "YA-47832" — и пересекается между train и test.

**Как детектировать:**
- Fuzzy matching по heat_id с порогом 0.9
- Подозрительно близкие составы (L2 < 0.01) с разными ID

**Проверка Critic:**
```
Q: Запущен ли fuzzy dedup перед split?
Q: Проверена ли пересекаемость train/test по composition similarity, а не только по ID?
```

**Что делать:** объединить дубликаты или полностью исключить из test set.

---

### D03: Распределение target на test сильно отличается от train

**Проблема:** На train средний σт = 520 МПа, стандартное отклонение = 40. На test средний = 580, σ = 30. Модель хорошо работает на train, но на test систематически занижает прогноз.

**Причина:** time-based split, и за последний год завод стал производить более прочные марки. Или лаборатория изменила методику. Или clientside filter исключил часть старых плавок.

**Как детектировать:**
- Разница средних target > 1σ между train и test
- KS-test p-value < 0.05 для распределения target

**Проверка Critic:**
```
Q: Совпадают ли распределения target на train и test (визуально и по статистике)?
Q: Есть ли в данных временной тренд по target?
```

**Что делать:** если тренд легитимный (реально меняется production) — делать sliding window retraining. Если это артефакт — найти причину и исправить.

---

### D04: Unit chaos в данных

**Проблема:** 70% записей σт в МПа, 30% в psi. Или температуры местами в °C, местами в К. Модель обучается на смеси и выдаёт мусор.

**Как детектировать:**
- Бимодальное распределение в колонке с ожидаемо непрерывным target
- Outliers сконцентрированы в один субдиапазон (не случайно)

**Проверка Critic:**
```
Q: Проведена ли unit canonicalization с логом каждой конверсии?
Q: Бимодальны ли распределения главных колонок?
Q: Есть ли "orphan" unit indicators в raw data?
```

**Что делать:** написать unit detector (см. skills/preprocessing/canonicalize.py), с явным flag на каждой конверсии.

---

### D05: Train-serve skew через feature computation

**Проблема:** В training pipeline feature CEV вычисляется через формулу IIW. В inference pipeline — через формулу Yurioka. Модель в production ошибается, и никто не понимает почему.

**Как детектировать:**
- Integration test: прогон одной и той же строки через train pipeline и через inference → сравнение features

**Проверка Critic:**
```
Q: Используется ли один и тот же feature computation module в train и inference?
Q: Есть ли версия feature set, зафиксированная в model artifact?
```

**Что делать:** всегда одна функция compute_features(), вызываемая и в training, и в inference.

---

### D06: Random split вместо time-based в temporal data

**Проблема:** В металлургии данные — временная последовательность. Рандомный split создаёт data leakage через correlated heats из одной кампании.

**Как детектировать:**
- R² на random split > R² на time-based split на 0.1+
- Плавки одного дня/смены оказываются в train и test одновременно

**Проверка Critic:**
```
Q: Использован ли time-based split (последние X% по дате — в test)?
Q: Если groups (campaigns) — использован ли GroupKFold для CV?
```

**Что делать:** всегда time-based split + GroupKFold.

---

### D07: Физически невозможные значения в раздатке

**Проблема:** В данных встречается %C = 3.45 (это уже чугун) или σт = 50 МПа (это медь, а не сталь). Опечатки оператора.

**Как детектировать:**
- Hard filter по physical bounds (см. skills/preprocessing/physical_bounds.py)
- Распределение выбросов: один знак отличается от нормы (0.345 → 3.45)

**Проверка Critic:**
```
Q: Прогнан ли physical bounds check с таблицей допустимых диапазонов?
Q: Сохранены ли rejected rows с причиной для аудита?
```

---

### M01: Overfitting на маленьком датасете

**Проблема:** 500 записей, XGBoost с max_depth=10, 2000 деревьев. R² train = 0.99, val = 0.45.

**Как детектировать:**
- |R² train - R² val| > 0.15

**Проверка Critic:**
```
Q: Соответствует ли сложность модели размеру данных? (N < 1000 → max_depth ≤ 5)
Q: Используется ли early_stopping_rounds?
Q: Применена ли регуляризация (reg_alpha, reg_lambda)?
```

**Что делать:** уменьшить max_depth, добавить reg_lambda, early stopping, уменьшить n_estimators.

---

### M02: Overconfident uncertainty

**Проблема:** Модель выдаёт 90% CI. На hold-out test только 60% точек попадает в CI. Это значит модель врёт об уверенности.

**Как детектировать:**
- Coverage test: в 90% CI должны попадать 85-95% точек
- Calibration plot: systematic deviation от diagonal

**Проверка Critic:**
```
Q: Проведён ли coverage test?
Q: Использована ли conformal prediction или quantile regression на hold-out?
Q: Откалибрована ли uncertainty на validation set?
```

**Что делать:** conformal prediction поверх baseline модели, либо quantile regression с calibration на отдельном set.

---

### M03: Неправильная метрика для задачи

**Проблема:** Обучение оптимизирует RMSE. Клиенту важно не ошибаться в нижнюю сторону по прочности (недооценка = опасность). RMSE симметрична, клиенту нужна asymmetric loss.

**Проверка Critic:**
```
Q: Согласована ли метрика обучения с бизнес-задачей?
Q: Если ошибки асимметричны — использован ли weighted или quantile loss?
Q: Показан ли residual plot для проверки систематического смещения?
```

**Что делать:** для safety-critical — quantile loss 0.1 (pessimistic prediction) в дополнение к mean.

---

### M04: Единичная модель без uncertainty

**Проблема:** Модель выдаёт одно число без CI. Инженер использует как точное, делает опытную плавку, результат не совпадает, теряет доверие.

**Проверка Critic:**
```
Q: Возвращает ли модель prediction interval, а не только point estimate?
Q: Показан ли CI в отчёте и в UI?
```

**Что делать:** quantile regression пара (q05, q95) всегда.

---

### M05: Feature importance без физической интерпретации

**Проблема:** Модель показывает, что для прочности HSLA важнее всего %Cu. Физически — это странно. Но модель так думает, и ей верят.

**Как детектировать:**
- Сравнить top-10 feature importance с металлургическим ожиданием для класса
- Если несоответствие — искать data leakage, spurious correlation, выборочное смещение

**Проверка Critic:**
```
Q: Соответствует ли top-10 feature importance металлургической интуиции для класса?
Q: Если нет — проведён ли causal analysis (permutation importance, partial dependence)?
```

**Что делать:** добавить в Critic металлургическую onboarding-onto: для HSLA ожидаемы C, Mn, Nb, Ti, процессные параметры. Нетипичные top-фичи — red flag.

---

### M06: OOD (out-of-distribution) не детектируется

**Проблема:** Клиент вводит состав с %Cr = 8% (а в training было max 1%). Модель уверенно предсказывает, что σт = 650 МПа. Это фантазия — модель экстраполирует.

**Проверка Critic:**
```
Q: Есть ли OOD detector перед inference?
Q: Какой порог Mahalanobis distance / density estimation для OOD flag?
Q: Возвращается ли warning при OOD input?
```

**Что делать:** fit Gaussian Mixture на training composition space, flag всё, что отстоит > 3σ от clusters.

---

### M07: Validation strategy без groups

**Проблема:** KFold split просто по строкам. Но плавки одной кампании (один день, один ковш) дают похожие свойства. Они оказываются в разных фолдах, и CV-score становится оптимистично завышенным.

**Проверка Critic:**
```
Q: Есть ли группирующая переменная (campaign, heat_group, month)?
Q: Используется ли GroupKFold/StratifiedGroupKFold?
```

**Что делать:** всегда GroupKFold по месяцу или campaign.

---

### I01: Inverse design без OOD check

**Проблема:** NSGA-II находит "оптимальный" состав, но он вне training domain. Модель выдаёт красивый прогноз, но это вера в экстраполяцию.

**Проверка Critic:**
```
Q: Заданы ли variable bounds в NSGA-II в пределах training distribution?
Q: Проверен ли каждый Pareto candidate на OOD до возврата пользователю?
```

**Что делать:** bounds = [min, max] по training ± 10% max. OOD flag при превышении.

---

### I02: Objectives с конфликтующими единицами

**Проблема:** Objective 1 = distance_to_target (в МПа), Objective 2 = alloying_cost (в ₽/т). Числа на разных порядках. NSGA-II оптимизирует в основном по cost, игнорирует prediction.

**Проверка Critic:**
```
Q: Нормализованы ли objectives перед подачей в NSGA-II?
Q: Документирован ли выбор весов в Decision Log?
```

**Что делать:** MinMaxScaler на каждый objective перед оптимизацией.

---

### I03: Constraint violations не возвращаются корректно

**Проблема:** Constraint g(x) = CEV - 0.43. Если CEV = 0.50, g = 0.07. Если CEV = 0.60, g = 0.17. Но в pymoo constraint нужно нормализовать — иначе один constraint подавляет другие.

**Проверка Critic:**
```
Q: Нормализованы ли constraints в pymoo (|g| < 1 для feasible → infeasible range)?
Q: Проверена ли Constraint Dominance на тестовом запуске?
```

---

### V01: Validator не знает tenant capabilities

**Проблема:** Система рекомендует состав с 0.008% N. Технологически это недостижимо на АКОС клиента (минимум 15 ppm = 0.0015%). Валидатор пропускает, клиент делает плавку — и не получает прогнозируемую структуру.

**Проверка Critic:**
```
Q: Включены ли tenant-specific constraints (purity limits, available elements) в Validator?
Q: Если tenant неизвестен (демо) — явно показано, что это generic validation?
```

**Что делать:** для MVP — generic validation, но с явным disclaimer в UI. Для пилота — tenant config обязателен.

---

### P01: Нет мониторинга distribution drift

**Проблема:** Модель в production 3 месяца, потом клиент сменил поставщика лома. Распределение S% сдвинулось. Модель начинает систематически ошибаться, но никто не видит.

**Проверка Critic:** (для production, не MVP)
```
Q: Есть ли weekly PSI check на ключевых features?
Q: Есть ли алерт при PSI > 0.2?
```

**Для MVP:** документировать как известное ограничение, план на следующую фазу.

---

### P02: Latency > 1 сек на inference

**Проблема:** Ensemble из 5 моделей + NSGA-II 300 поколений = 15 секунд на запрос. В UI пользователь видит spinner, закрывает вкладку.

**Проверка Critic:**
```
Q: Измерена ли p95 latency на inference?
Q: Если > 1 сек — есть ли caching / async processing / progress bar?
```

**Что делать для MVP:** Streamlit показывает progress, это нормально для демо.

---

## Как Critic использует эту библиотеку

```python
# Pseudocode
class CriticAgent:
    def review(self, phase: str, artifact: dict, context: dict) -> CriticReport:
        relevant_patterns = self.load_patterns(phase)
        warnings = []
        for pattern in relevant_patterns:
            if pattern.applies(artifact, context):
                check_result = pattern.run_check(artifact, context)
                if check_result.failed:
                    warnings.append({
                        "pattern_id": pattern.id,
                        "severity": pattern.severity,
                        "message": check_result.message,
                        "suggestion": pattern.suggestion,
                    })
        return CriticReport(warnings=warnings, overall_verdict=...)
```

## Roadmap расширения

- **Неделя 1-4:** 20 паттернов выше, покрывают 60% типовых ошибок в MVP.
- **Неделя 5-8:** добавить ещё 10-15 на основе того, что найдёте при работе.
- **Месяц 3-6 (пилот):** расширить до 50-80 паттернов, включая tenant-specific.
- **Год 1-2:** 100-200 паттернов. В этот момент Critic реально начинает ловить 70-80% ошибок.

Это живой документ. Растёт с проектом.
