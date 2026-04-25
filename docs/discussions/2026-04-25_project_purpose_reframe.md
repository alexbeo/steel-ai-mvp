---
title: Project purpose reframe — MVP, не sales-tool для Voestalpine
date: 2026-04-25
status: decided
---

## Контекст

К этому моменту в репо накопилось 4 pitch-deck-а для Voestalpine (DE_EDITORIAL, DE_AEROSPACE, EINKAUF_DE.html, EINKAUF_DE.md), несколько scout-отчётов, серия data-spike коммитов под HSLA pipeline steel. Это создавало впечатление, что проект — сейлз-инструмент для конкретного клиента (CPO Voestalpine), и все технические решения должны приниматься через призму этого pitch'а.

При обсуждении следующих шагов разработки выяснилось, что это **неверная рамка**.

## Реальная позиция

Пользователь явно зафиксировал:

1. **Это MVP**, не sales-product. Цель не «продать Voestalpine», а проверить работоспособность концепта.
2. **HSLA это абстракция / демонстрация**. Конкретный класс стали не имеет принципиального значения. Pipe HSLA был выбран как один из понятных индустриальных кейсов, не как product focus.
3. **Voestalpine pitch — артефакт промежуточного этапа**, не центральная цель. Pitch-decks остаются в репо как артефакт, но не диктуют roadmap.

## Главное ожидание от продукта

> Возможность интеграции технологий ИИ в приложение для получения лучшего результата из возможных, **используя способность ИИ находить и видеть паттерны, которые не очевидны для человека**.

То есть фокус — *AI как discovery tool*, не как калькулятор. Текущий pipeline (XGBoost + NSGA-II + GMM + hand-coded patterns) — это classical ML с safety-rails, в нём «AI» в современном смысле почти нет. Эту сторону нужно развивать.

## Что эта переориентировка меняет

| Было | Стало |
|---|---|
| Принимать решения через «как это смотрится в pitch» | Принимать решения через «даёт ли это unique AI-driven value» |
| Приоритет — HSLA-specific accuracy | Приоритет — generalizable AI capabilities, work с любым steel |
| Customer = Voestalpine CPO | Customer = R&D engineer + Materials scientist (см. отдельное обсуждение) |
| Метрика успеха = «pitch принят» | Метрика успеха = «AI нашёл что-то неочевидное и металлург согласился что это полезно» |
| Backlog содержит «обновить pitch-bullet» | Backlog содержит «развить AI-capabilities, pitch-сlaims обновим post-hoc» |

## Что НЕ меняется

- Архитектура pipeline (data → train → critic → inverse → validate → report) остаётся.
- Pattern Library + Decision Log + LLM-Critic — полезные части, развиваем дальше.
- Conformal calibration, multi-class support, deoxidation calculator — всё legitimate technical infra.
- Pitch-файлы остаются в `sales/` как артефакт; их **не обновляем под изменившийся scope** до появления конкретного customer ask.

## Решение

Прекратить рассматривать «обновление pitch-bullet» как next step. Зафиксировать AI-discovery direction как core продукт. План реализации — в `2026-04-25_ai_integration_roadmap.md`.

## Open questions

- Когда / как pitch-материалы обновляются под новый scope? Скорее всего — после первого реализованного AI-feature, который реально работает на Agrawal данных. Тогда pitch перепишется естественно вокруг этого.
- Сейлз / academic outreach остаётся за пользователем. Я не предлагаю outreach-инициатив, пока не попросят.
