# Steel AI — Pitch Deck Outline

10-слайдовый pitch deck для инвесторов и pilot-клиентов. 
Заполнять можно в Google Slides / Pitch.com / Canva.

---

## Slide 1: Cover

**Steel AI**
AI-driven alloy design for steel manufacturers

Subtitle: *10× faster development of new steel grades with uncertainty-aware ML*

Logo · Year · Your name · Contact

---

## Slide 2: The Problem

**Разработка новой марки стали = 2-5 лет и €5-20M**

- Традиционный цикл trial-and-error: 50-200 лабораторных плавок
- Каждая плавка 20-50 кг = €10-50k и 2-4 недели
- Промышленная квалификация — ещё 12-18 месяцев
- Результат часто не попадает в ТЗ с первого раза

**Almost nobody uses modern AI properly в этой индустрии:**
- Citrine фокусируется на химии, CPG и пластике
- QuesTek специализируется на аэрокосе и спецсплавах
- Intellegens — general ML-tool, не вертикальный steel-продукт

**Steel industry-specific AI tool is missing.**

---

## Slide 3: The Solution

**Steel AI — вертикальная AI-платформа для стального сектора**

Три ключевые функции:

1. **Forward prediction.** От химии и режима обработки — к свойствам (σт, σв, δ, KCV) 
   с uncertainty bars. R² > 0.85 на реальных данных.

2. **Inverse design.** От желаемых свойств — к оптимальному составу через multi-objective 
   optimization (NSGA-II), с учётом стоимости легирования и технологических ограничений.

3. **Physics-informed validation.** Каждый кандидат проверяется на свариваемость (CEV/Pcm),
   горячую деформируемость, OOD, соответствие марочнику ГОСТ/ASTM/DIN.

**Результат:** 50-200 плавок trial-and-error → 5-15 целенаправленных плавок.

---

## Slide 4: Why Now

- **Foundation models созрели.** LLM-агенты делают автоматизацию пайплайнов реальностью в 2026.
- **Декарбонизация стали** требует новых марок (водородная DRI, EAF, scrap-based). 
  Существующие справочники не помогают.
- **Carbon Border Adjustment Mechanism (CBAM)** в ЕС с 2026 делает оптимизацию легирования 
  критичной для экономики заводов.
- **Dataset-access норма отрасли сдвигается** — крупные заводы готовы экспериментировать 
  с внешними AI-вендорами, чего не было 5 лет назад.

---

## Slide 5: Product Demo

[Вставить скриншот Streamlit UI с найденным кандидатом]

**User flow:**
1. Металлург задаёт ТЗ: σт ≥ 485 МПа, KCV-60 ≥ 50 Дж/см², CEV ≤ 0.43
2. Платформа за 2-5 минут возвращает 5 оптимальных составов
3. Для каждого: прогноз ± CI, SHAP explanation, ближайшая марка, cost
4. Металлург выбирает 1-2 для опытной плавки
5. После плавки — фидбэк в систему, модель переобучается (active learning)

**Demo link:** [ваш URL]

---

## Slide 6: Technical Approach

**Гибрид ML + physics + LLM-orchestration:**

- **Prediction:** XGBoost + quantile regression для uncertainty. Physics-informed features 
  (CEV, Pcm, CEN, Hollomon-Jaffe, VEC).
- **Inverse design:** NSGA-II multi-objective, с learning-based surrogate.
- **OOD detection:** Gaussian Mixture на composition space.
- **Agents:** Claude Sonnet координирует 6 специализированных агентов.
- **Pattern Library:** структурированная база из 100+ анти-паттернов для автоматической 
  self-review.
- **Decision Log:** persistent memory проекта через SQLite.

**Data efficiency:** работает на 500-5000 плавок (размер одного завода).

---

## Slide 7: Market

**TAM:** €400-800M глобальный R&D-software budget в сталеварении.

**SAM:** €100-200M — заводы, активно инвестирующие в digital.

**SOM (3 года):** 10-15 pilot customers × €50-150k/год = €0.5-2M ARR.

**Scaling (5 лет):** 30-50 enterprise × €150-500k/год = €5-25M ARR.

**Logos, на которые целимся:**

Тier 1: Voestalpine, SSAB, Salzgitter, Dillinger, ArcelorMittal Europe
Тier 2: Tenaris, Vallourec, Liberty Steel, US Steel  
Тier 3: Региональные заводы (POSCO, Tata, JSW, HBIS)

---

## Slide 8: Business Model

**SaaS подписка + Success fee:**

- Starter: €30-60k/год — 1 класс сталей, 1 engineer user, email support
- Pro: €100-200k/год — 3 класса, 10 users, pilot design services  
- Enterprise: €300-500k/год — unlimited, on-premise, custom integrations

**Optional success fee:** 10-15% от документированной economy от легирования или 
ускорения development cycle.

**Unit economics:** CAC €30-80k, LTV €500k-2M, payback 12-24 месяца.

---

## Slide 9: Team

**Founder & CEO — [Your Name]**
- Fullstack-разработчик с 3+ годами в AI/LLM
- Опыт управления metal trading компанией (понимание индустрии)
- [Ваши дополнительные regalia]

**Co-founder & Domain Expert — [German Partner Name]**
- [X]+ лет в европейской чёрной металлургии  
- Широкая сеть контактов в R&D: Voestalpine, Salzgitter, SMS Group
- [Предыдущие достижения]

**Advisors (планируется):**
- Металлургический профессор из TU или Cambridge
- Бывший CTO крупного заводского R&D

**Роли, которые мы нанимаем:**
- Senior ML Engineer (co-founder/CTO-track, 10-15% equity)

---

## Slide 10: Ask

**Раунд:** €1.5-2.5M Seed

**Использование средств:**
- 50% — hiring (ML Engineer, Data Engineer, BD)
- 25% — pilot implementations (3 pilots за 18 месяцев)
- 15% — infrastructure + compute
- 10% — legal, travel, go-to-market

**Milestones (18 месяцев):**
- ✓ MVP готов (месяц 0)
- Pilot #1 signed — месяц 3
- Pilot #1 live — месяц 9
- 3 paying customers — месяц 15
- €500k ARR — месяц 18 → Series A

**Target investors:** Speedinvest, HTGF, EIT InnoEnergy, Breakthrough Energy Europe, 
Earlybird, industrial corporate VCs.

---

## Appendix (slides 11-15)

- Competitor analysis (Citrine, QuesTek, Intellegens, BIOVIA)
- Financial model (5-year projection)
- Technical architecture detail
- Pilot case study
- Science advisors bio
