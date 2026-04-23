# Pilot Proposal Template

Шаблон предложения для первого платного пилота. Заполняется под конкретного клиента после discovery-звонков.

---

## 1. Executive Summary

**Client:** [Название завода]
**Pilot Duration:** 3-6 months
**Investment:** €[50-150k]
**Expected Value:** [€X-Y savings / Z% faster development cycle]

---

## 2. Problem Statement

Из discovery-звонков с [Иван Иванович, R&D Director] выявлены следующие боли:

- [Например: "Разработка новой марки K60-Arctic для Ямал-Европа занимает 18 месяцев и 50+ опытных плавок"]
- [Например: "Current approach даёт 30% первопроходный brak rate из-за KCV-60"]
- [Например: "Оптимизация легирования ведётся по handbook-формулам 1990-х"]

**Quantified pain:** [€X в год из-за переделок / Y недель ожидания результатов / Z% rejected plans]

---

## 3. Proposed Solution

### Scope

В pilot включено:
1. **Data onboarding** — импорт ваших плавок за 3-5 лет ([N] записей) в нашу платформу
2. **Custom model training** — модель для прогноза [σт, σв, KCV-60] под ваш класс [K60]
3. **Inverse design workflow** — по 10-15 новых ТЗ от ваших инженеров
4. **Training session** — 2-3 дня для ваших металлургов
5. **Active learning loop** — 3-5 опытных плавок по нашим рекомендациям в период pilot

Не входит в pilot:
- Интеграция с вашим MES/ERP (отдельная фаза)
- On-premise deployment (по умолчанию cloud EU)
- Custom feature development

### Success Criteria

Pilot считается successful при достижении 2 из 3:

1. **Accuracy:** R² модели ≥ 0.82 на hold-out plавках клиента
2. **Actionability:** минимум 2 рекомендации из 10-15 перешли в опытную плавку
3. **Time savings:** documented 3-5× ускорение на explored composition space

---

## 4. Technical Approach

### Phase 1: Discovery & Data (weeks 1-4)
- Access к плавкам (NDA, secure S3 / on-premise ingestion)
- Data cleaning pipeline под ваш MES-формат
- Baseline model на подмножестве

### Phase 2: Full model + Validation (weeks 5-10)
- Full training on всех плавках
- Calibration, OOD detection, SHAP
- Review с вашими металлургами — корректировка Feature Set

### Phase 3: Production usage (weeks 11-20)
- 10-15 ТЗ от ваших инженеров через UI
- Выплавка 3-5 кандидатов
- Feedback loop, переобучение модели
- Final report с ROI analysis

### Phase 4: Decision point (week 21-24)
- Результаты pilot оформляются в документ
- Joint business review
- Decision: continuation в full contract / extension / conclusion

---

## 5. Team & Responsibilities

### Steel AI team:
- [Your name], Tech Lead — ML integration, model development
- [Partner name], Domain Expert — metallurgical consultation, review
- [Potential ML engineer hire], Data Engineer — data pipeline

### Client team (requested):
- Project Sponsor — VP R&D или equivalent
- Technical Owner — 1 ведущий металлург, 10-15h/week
- Data Access Owner — IT специалист для data extraction
- Trial Heats Coordinator — доступ к лабораторной плавильне

### Cadence:
- Weekly 30-min sync calls
- Monthly steering review (60 min, с Project Sponsor)
- Quarterly business review (90 min)

---

## 6. Investment & Timing

### Pricing structure:

**Option A — Fixed fee:** €120,000
- Payment schedule: 40% at kickoff, 30% after Phase 2, 30% after Phase 4
- Includes: все перечисленное выше
- Excludes: VAT, трэвел (billed at cost)

**Option B — Success-based:** €60,000 base + €80,000 success fee
- Base payable monthly over 6 months
- Success fee payable only if ≥2 из 3 criteria met

**Option C — Partnership:** €30,000 + 25% equity stake in alloy IP developed
- Applicable only если pilot ведёт к proprietary new grade
- Subject to separate IP agreement

### Additional costs (client-side):
- Opытные плавки: 3-5 × €15-30k = €45-150k (по тарифу вашего RTL)
- Testing: стандартные испытания, у вас и так бюджетируются
- Team time: ~10-15h/week × 4 месяца = ~180-280 часов engineering

---

## 7. Data & IP Terms

### Data:
- Ваши данные остаются вашими. Мы имеем лицензию на обучение модели **только в 
  вашем tenant**. Модели не кросс-используются между клиентами.
- Raw data хранится в encrypted form, EU region.
- По окончанию pilot вы получаете trained model artifact (weights + config) — 
  не только inference doc.

### IP:
- Методология Steel AI (алгоритмы, агенты, Pattern Library) — наша.
- Сgenerated составы кандидатов — ваши (вы не платите royalty за марки, 
  проектированные платформой).
- Joint publications / case studies — только с вашего письменного согласия.

### Termination:
- Любая сторона может прервать pilot с 30-дневным уведомлением.
- Pro-rata refund за неиспользованные phases при early termination не по вашей fault.

---

## 8. Why Us (Differentiation)

**vs Citrine Informatics:** Мы specialize в стали, они — в химии и CPG. Our Pattern 
Library содержит 100+ металлургически-специфичных anti-patterns, которых у них нет.

**vs QuesTek ICMD:** Они сильны в аэрокосе/AM, но HSLA и конструкционные стали — не 
core. Мы дешевле в 3-5× и fast-turnaround.

**vs Intellegens Alchemite:** General-purpose ML-tool. Требует data scientist клиента. 
Наше — вертикальная платформа, с UI для металлургов без data science background.

**vs internal R&D software:** Ваш team занят основной работой. Мы — turnkey 
дополнительный capacity без найма 5 ML-инженеров.

---

## 9. Next Steps

1. **This week:** Review этого документа, comments назад
2. **Week 2:** 90-min technical deep-dive (ваши ML/IT people + наш team)
3. **Week 3:** NDA подписан, data sample передан для feasibility
4. **Week 4:** Feasibility report, final SOW
5. **Week 5-6:** Contract signing, kickoff

**Primary contact:**
[Your name]
[email] | [phone]
Steel AI · Belgrade, Serbia
