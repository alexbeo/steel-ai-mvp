---
title: A2 hypothesis generator — verification на Agrawal модели
date: 2026-04-25
status: decided
verdict: PASSED
---

## Контекст

A2 (automated hypothesis generation) — первое из 5 направлений AI integration roadmap. Verification gate из roadmap:

> на Agrawal NIMS модели LLM должен предложить ≥3 hypotheses, ≥1 из которых нетривиальная (не просто «больше углерода → выше fatigue»).

## Параметры запуска

- **Модель:** `fatigue_fatigue_strength_xgb_20260424_233914` (Agrawal NIMS, R² 0.978, MAE 16.6 МПа, conformal-corrected coverage 0.92)
- **Provider:** Claude Sonnet 4.6 через `app/backend/hypothesis_generator.py`
- **Latency:** 46.6 секунд
- **Tokens:** 3042 input + 2400 output. Cache miss (первый запуск); следующий запуск с тем же system prompt ~90 % input cost экономии.
- **Cost:** ≈ $0.045 за запрос (Sonnet 4.6 pricing $3/$15 per Mtok)
- **Полный лог:** Decision Log запись `author=hypothesis_generator`, tag `hypothesis`

## Полученные 5 гипотез (краткая таблица)

| # | Novelty | Statement (одной фразой) | Predictable? |
|---|---|---|---|
| 1 | MEDIUM | Carburizing temp saturates fatigue benefit at ~900 °C | Да, классический механизм grain coarsening |
| 2 | HIGH | Narrow normalizing-temp optimum 880-900 °C; importance 0.41 на 105°C-диапазоне → steep non-linear | **Нетривиально** |
| 3 | HIGH | Cr × cooling-rate interaction: Cr substitutes for quench severity in hardenability | **Нетривиально** (interaction effect) |
| 4 | MEDIUM | Bulk C surprisingly low importance — surface carburizing dominates | **Counter-intuitive** vs стандартной школы |
| 5 | HIGH | Asymmetric conformal CI signals sparse high-strength tail; Mo as hidden gating variable | **Самая сильная** (mixed reasoning: uncertainty + features + data) |

## Анализ качества

**По critère verification gate (≥3 hypotheses, ≥1 нетривиальная):**
- Сгенерировано 5 (требовалось ≥3) ✓
- Нетривиальных (HIGH novelty + grounded в specific artifact data): **3 из 5** (#2, #3, #5) ✓✓✓

**Что особенно ценно:**

1. **Hypothesis #5** — Claude связал три разнородные сигнала: asymmetric conformal CI (signal от calibration), low Mo importance (signal от feature_importance), tail of target distribution (signal от dataset stats). Это **мульти-сигнальный reasoning**, который человек не сделал бы при беглом просмотре tabular data — нужно было бы открыть отдельно несколько графиков и сопоставить. Конкретный proposed experiment с ожиданием «narrowing of the conformal CI width» — testable.

2. **Hypothesis #3** — interaction effect (Cr × cooling-rate) в tabular data принципиально невидим без либо ML-модели, либо ручного построения 2D heatmap'ов. Claude корректно вывел его из importance-ranking + физического смысла Cr как hardenability promoter.

3. **Hypothesis #4** — самое полезное для металлурга-академика. Claim «bulk C неважен в карбюризованных сталях» противоречит textbook intuition но согласуется с физикой carburizing (поверхностный профиль решает усталость, не bulk). Это hypothesis уровня small paper.

**Что в hypothesis-output можно улучшить (для будущих итераций prompt'а):**

- Hypothesis #2 цитирует «r2_train–r2_test gap (0.9966 vs. 0.9775) hints at mild overfitting». Это правда, но gap небольшой и для small-N датасета (290 train) ожидаем. Claude мог бы быть calibrated жёстче — этот sub-claim скорее distractor чем insight.
- Все 5 гипотез работают в основном с processing variables. Composition-side (особенно interaction между Mn-S, P-S clustering, Ni-Cr substitution в hardenability) почти не затронут. Возможно нужен hint в prompt'е чтобы LLM искал **разные** angles.
- Proposed experiments отлично структурированы (fix + sweep), но не учитывают cost эксперимента и время. Для R&D engineer это полезно добавить.

Ничто из этого не блокер. Текущее качество достаточно для production.

## Вердикт

**A2 verification: PASSED.**

Качество гипотез **превосходит** verification gate. Hypothesis generator готов к продакшн-использованию. UI integration (A2.3) имеет смысл — есть что показывать.

## Следующие шаги (по roadmap)

1. **A2.3** — Streamlit UI вкладка «Гипотезы»: кнопка «сгенерировать», отображение list, кнопки accept/reject (для будущей RLHF, см. C-уровень roadmap).
2. **A2.4** уже частично сделано через этот документ.
3. После A2.3 → переходим к **A1** (LLM-driven feature discovery) согласно roadmap.

## Operational notes

- Стоимость одного запроса hypothesis generator: ≈ $0.045. Допустимо для interactive use.
- Latency 46s — слишком долго для real-time UI. В Streamlit нужен async pattern с прогресс-баром или background thread.
- Decision Log правильно фиксирует все запросы. Можно потом анализировать: какие hypotheses пользователь принял/отклонил, какие приводили к экспериментам, какой % HIGH novelty оказывался реально валидным.

## Replicate

```bash
export $(grep -v '^#' .env | xargs)
PYTHONPATH=. .venv/bin/python scripts/generate_hypotheses_for_model.py
```

Output идентичен с фиксированным random_seed на стороне модели. Claude отвечает каждый раз чуть иначе (sampling), но содержательно повторяемо.
