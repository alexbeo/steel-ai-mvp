---
title: Target users — R&D engineer + Materials scientist
date: 2026-04-25
status: decided
---

## Контекст

Параллельно с переориентировкой на MVP-mode (см. `2026-04-25_project_purpose_reframe.md`) уточнён профиль конечного пользователя.

## Профили

### 1. R&D engineer в металлургии (industrial R&D)

Контекст:
- Работает на стальном заводе или в исследовательском центре корпорации (типа Voestalpine Research, Salzgitter Mannesmann Forschung, ArcelorMittal Global R&D).
- Имеет доступ к **реальным заводским данным** — плавки, состав, термообработка, испытания.
- Бюджет на эксперимент ограничен: каждая плавка = тысячи евро, каждое испытание = часы лабораторного времени.
- Знает domain глубоко, не нуждается в объяснении что такое CEV или TMCP.
- Цель использования системы: **сократить число экспериментов**, прийти к target свойствам быстрее.

Ожидания от AI-feature:
- Активное предложение «следующего эксперимента» (Bayesian / active learning).
- Объяснимость каждого предложения (не просто «попробуйте X», а «попробуйте X потому что ...»).
- Уверенность в predictions с честной uncertainty (conformal calibration уже закрыли).
- Интеграция с собственными данными — не «подайте мне ваши плавки», а «вот мой CSV из MES, обучайтесь».

### 2. Materials scientist в академии (academic research)

Контекст:
- Университет / госинститут (TU Berlin, MPI Eisenforschung, ИМЕТ РАН и подобные).
- **Меньшие датасеты** — чаще десятки-сотни записей, не тысячи.
- Цель — публикуемое научное открытие, не ROI.
- Сильнее интерес к interpretability и replicability чем к production-ergonomics.
- Конкурирует с другими ML-методами в литературе, нужны метрики уровня Agrawal 2014 R²~0.98.

Ожидания от AI-feature:
- **Symbolic regression** — извлечение аналитических формул, которые можно сравнить с Hall-Petch / Orowan / классической physical metallurgy.
- **Hypothesis generation** — testable predictions, которые можно вынести в follow-up paper.
- **Causal reasoning** — корреляция vs причинность, defensible в peer review.
- Воспроизводимость — фиксированные seed, артефакты, scriptable workflow.

## Общие требования двух профилей

Оба профиля нуждаются в:
- Прозрачности что внутри модели (feature importance, training_ranges, OOD-границы).
- Возможности использовать **свои данные**, а не demo-синтетику.
- Корректной uncertainty — никакого overconfident prediction.
- Decision log как audit trail (для R&D — для compliance, для academic — для methods section).

## Что это меняет в продукте

1. **UI-flow «загрузить свой CSV → обучить»** должен быть first-class. Сейчас он реализован частично через train-вкладку, но требует ручного создания SteelClassProfile. Нужен **upload + auto-detect schema + training**.

2. **AI-features должны работать с произвольным steel-class**, не только с предзагруженными pipe_hsla / en10083_qt / fatigue_carbon_steel. Hypothesis generator, symbolic regression, active learning — все должны быть class-agnostic.

3. **Документация и UI на двух языках**: русский (pis-russian-author) + английский (для academic + EU R&D пользователей). Российских колаборантов меньше чем международных, поэтому английский — primary, русский — secondary.

4. **Sales pitch для Voestalpine CPO становится не первоочередной задачей.** Если он позже будет нужен — переписываем под уточнённого пользователя (R&D Head, не Einkauf CPO — это другая роль).

## Не входит в этот profile

- Production engineers / operators (тех кто реально делает плавку) — другие требования (интеграция с MES, real-time, compliance с конкретными стандартами).
- Sales / procurement — мы не пытаемся показать ROI закупкам.
- Студенты / любители — не наша аудитория, MVP не для них.

## Решение

Все следующие AI-features (A1-B2 из roadmap) проектируются под этих двух пользователей. Конкретные различия (R&D хочет active learning, academic хочет symbolic regression) учитываются при выборе порядка реализации.
