---
title: Steel AI MVP
emoji: 🔥
colorFrom: red
colorTo: indigo
sdk: streamlit
sdk_version: 1.56.0
app_file: app/frontend/app.py
pinned: false
license: mit
short_description: Подбор состава стали + раскисление + диагностика рецептов через глубокую нейронную сеть
---

# Steel AI MVP

Прототип системы поддержки решений в металлургии: подбор химического состава
стали, раскисление жидкой стали алюминием, прогноз механических свойств,
анализ аномалий — с использованием глубоких нейронных сетей и классических
термодинамических моделей.

## Что внутри

8 функциональных вкладок Streamlit:

- **🎯 Дизайн сплава** — multi-objective NSGA-II под целевые свойства + минимум стоимости ферросплавов
- **🤖 Обучение модели** — XGBoost + quantile regression + conformal calibration
- **📊 Прогноз** — predict с 90% CI + автоматический PhD-разбор аномалии
- **🔥 Раскисление** — 3 термодинамические модели + полный operator protocol с PhD-рецензией
- **💡 Гипотезы** — глубокая нейронная сеть формулирует testable research hypotheses, второй агент-критик делает peer review
- **🧪 Подбор рецепта** — designer + adversarial critic с двойной evidence base (artifact-data + classical metallurgical mechanism)
- **🔭 Следующие эксперименты** — cost-weighted Expected Improvement для planning experimental queue
- **📚 История** — audit trail всех решений (Decision Log)

## Как запустить

```bash
pip install -r requirements.txt
PYTHONPATH=. streamlit run app/frontend/app.py
```

Для функций с глубокой нейронной сетью (PhD-критик, hypothesis generator,
recipe designer, anomaly explainer, deoxidation advisor) необходим
`ANTHROPIC_API_KEY` в `.env` или в Streamlit Cloud / HF Spaces secrets.

## Архитектурные особенности

- **Все 25 признаков fatigue-модели обучены на 437 реальных peer-reviewed
  записях** (Agrawal NIMS 2014, DOI 10.1186/2193-9772-3-8). HSLA / Q&T —
  пока synthetic generators для демо.
- **Conformal-corrected 90% CI** для всех прогнозов через split-conformal
  калибровку (Romano et al. 2019).
- **Adversarial peer review** — все генеративные функции (рецепты, гипотезы,
  раскисление) проходят независимую PhD-рецензию с построчным fact-check
  доказательной базы.
- **Cost-aware optimization** — ferroalloy pricing snapshot (EUR), audit
  trail цен в Decision Log per run.

## Лицензия

MIT.
