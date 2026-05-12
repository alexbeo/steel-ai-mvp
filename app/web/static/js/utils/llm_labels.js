// Mapping LLM tool_use schema keys → human-readable Russian labels.
//
// Источник правды — JSON tool_use schemas в `prompts/*.md` (gitignored,
// см. prompts/README.md). Каждый раздел этого файла соответствует
// одному LLM-модулю (advisor/critic пара). Ключи здесь — это
// `attack_vectors[].id`, enum-значения и имена top-level полей,
// которые могут попасть в UI как plain string.
//
// Free-text technical jargon (XGBoost, SHAP, NSGA-II, OOD, Murakami,
// Hall-Petch, Pcm, Pickering, Andrews Ms, Larson-Miller, p-value,
// power analysis, conformal correction) НЕ переводится — это
// стандартная ML/DOE/metallurgy терминология.
//
// Для defensive-rendering используется `labelFor(key, dict)`: если
// ключ отсутствует — возвращается human-readable fallback с replace
// `_` → пробел. Так UI не падает на новые ID, добавленные в промпт
// до обновления этого словаря.

// ──────────────────── Hypothesis critic (view: hypotheses.js) ────────────────────

/**
 * Attack-vector IDs из prompts/hypothesis_critic.md →
 * adversarial_mindset.attack_vectors[].id. Sonnet возвращает строки
 * `weaknesses` в формате "id: text" (см. скриншот пользователя
 * 2026-05-09); функция `splitKeyedLine` парсит prefix и подставляет
 * перевод.
 */
export const HYPOTHESIS_CRITIC_LABELS = {
  // attack_vectors из adversarial_mindset
  logical_leap: 'Логический скачок',
  confounders_in_experiment: 'Смешивающие факторы в эксперименте',
  power_problems: 'Проблемы статистической мощности',
  inflated_novelty: 'Завышенная новизна',
  fictional_economics: 'Сомнительная экономика',
  ignored_alternative_explanations: 'Игнорированные альтернативные объяснения',
  scope_creep_ood_risk: 'Расширение scope / OOD-риск',
  misuse_of_uncertainty: 'Некорректное использование неопределённости',
  doe_design_violations: 'Нарушения DOE-дизайна',
};

// ──────────────────── Recipe critic (view: recipe.js) ────────────────────

/**
 * Attack-vector IDs из prompts/recipe_critic.md → adversarial_mindset[].id.
 */
export const RECIPE_CRITIC_LABELS = {
  evidence_integrity: 'Целостность доказательной базы',
  ml_prediction_vs_expected_outcome: 'ML-прогноз vs ожидание автора',
  ood_risk: 'OOD-риск',
  cost_saving_fact_check: 'Проверка экономии',
  novelty_inflation: 'Завышение новизны',
  ignored_confounders: 'Игнорированные смешивающие факторы',
  mechanism_inversions: 'Перепутанное направление механизма',
  baseline_ambiguity: 'Неоднозначность baseline',
  weldability_blind_spot: 'Слепое пятно по свариваемости',
  misuse_of_uncertainty: 'Некорректное использование неопределённости',
};

// ──────────────────── Recipe designer (view: recipe.js) ────────────────────

/**
 * Sub-keys из prompts/recipe_designer.md (используется в evidence
 * format prefixes: "artifact: …" / "mechanism: …"). Эти не появляются
 * напрямую как ID-ключи в UI, но если LLM напишет format-prefix —
 * лучше иметь human-readable форму.
 */
export const RECIPE_DESIGNER_LABELS = {
  artifact: 'артефакт',
  mechanism: 'механизм',
  process: 'процесс',
};

// ──────────────────── Deox advisor (view: deox.js, AI sub-tab) ────────────────────

/**
 * Confidence enum + risk_flag prefix-keys (advisor сам формирует
 * `risk_flags` как plain strings, но иногда добавляет ID-prefix —
 * см. примеры в prompts/deoxidation_advisor.md).
 *
 * Top-level field labels (al_addition_kg, al_form, etc.) уже
 * локализованы в самой view через cell()/strong-блоки, отдельный
 * mapping не нужен.
 */
export const DEOX_ADVISOR_LABELS = {
  // al_form enum (если LLM вдруг вернёт ID, а не human form)
  wire: 'проволока',
  cube: 'кубики',
  powder: 'порошок (инжекция)',
};

// ──────────────────── Deox critic (view: deox.js, AI sub-tab) ────────────────────

/**
 * Attack-vector IDs из prompts/deoxidation_critic.md →
 * adversarial_mindset[].id.
 */
export const DEOX_CRITIC_LABELS = {
  number_sanity_check: 'Проверка реалистичности чисел',
  recovery_factor_reality: 'Реалистичность фактора усвоения Al',
  form_choice_appropriateness: 'Выбор формы Al (wire/cube/powder)',
  risk_identification_completeness: 'Полнота идентификации рисков',
  inclusion_forecast_plausibility: 'Достоверность прогноза включений',
  pre_post_action_ordering: 'Порядок pre/post action',
  model_selection_rationale: 'Обоснование выбора thermo-модели',
  process_timing_realism: 'Реалистичность timing процесса',
  stoichiometry_arithmetic: 'Арифметика стехиометрии',
  slag_chemistry_constraints: 'Ограничения по химии шлака',
};

// ──────────────────── Anomaly explainer (view: predict.js) ────────────────────

/**
 * deviation_kind enum + failure_mode_catalog[].name из
 * prompts/anomaly_explainer.md. predict.js рендерит mechanism_concerns
 * как plain strings — LLM иногда префиксует строку именем
 * failure_mode_catalog item (например, "retained_austenite_excess: …");
 * splitKeyedLine переводит prefix.
 */
export const ANOMALY_EXPLAINER_LABELS = {
  // deviation_kind enum
  out_of_range_high: 'выше training range',
  out_of_range_low: 'ниже training range',
  unusual_combination: 'необычное сочетание',
  extreme_within_range: 'на краю training range',
  // failure_mode_catalog[].name
  retained_austenite_excess: 'Избыток остаточного аустенита',
  hot_shortness: 'Hot shortness',
  MnS_shape_control_failure: 'Сбой shape control MnS',
  carbide_network_brittleness: 'Хрупкость карбидной сетки',
  decarburization: 'Обезуглероживание',
  spinel_inclusion_formation: 'Образование шпинельных включений',
  temper_embrittlement_reversible: 'Обратимая отпускная хрупкость',
  TiN_cluster_brittleness: 'Хрупкость TiN-кластеров',
  nb_saturation_no_uplift: 'Насыщение Nb без эффекта',
};

// ──────────────────── Shared enums (verdict / evidence / severity) ────────────────────

/** Verdict enum — общий для всех critic-views (recipe / hypothesis / deox AI). */
export const VERDICT_LABELS = {
  ACCEPT: 'Принять',
  REVISE: 'Доработать',
  REJECT: 'Отклонить',
};

/** Evidence-check verdict — для recipe_critic / deoxidation_critic. */
export const EVIDENCE_CHECK_LABELS = {
  VALID: 'подтверждено',
  INVALID: 'опровергнуто',
  UNVERIFIABLE: 'не проверяемо',
};

/** Anomaly severity (predict.js anomaly explainer) + Pattern Library severity. */
export const SEVERITY_LABELS = {
  HIGH: 'Высокая',
  MEDIUM: 'Средняя',
  LOW: 'Низкая',
};

/** Hypothesis novelty + recipe novelty. */
export const NOVELTY_LABELS = {
  HIGH: 'Высокая',
  MEDIUM: 'Средняя',
  LOW: 'Низкая',
};

/** Critic confidence (общий для всех 4 LLM-views). */
export const CONFIDENCE_LABELS = {
  HIGH: 'высокая',
  MEDIUM: 'средняя',
  LOW: 'низкая',
};

// ──────────────────── helpers ────────────────────

/**
 * Перевод одного ключа с graceful fallback. Если ключа нет в словаре —
 * возвращает human-readable форму через replace `_` → space (так
 * новые ID, не успевшие в словарь, всё равно читаемы).
 *
 * @param {string} key
 * @param {Record<string, string>} dictionary
 * @returns {string}
 */
export function labelFor(key, dictionary) {
  if (key == null) return '';
  const k = String(key);
  if (dictionary && Object.prototype.hasOwnProperty.call(dictionary, k)) {
    return dictionary[k];
  }
  return k.replace(/_/g, ' ');
}

/**
 * Парсит строку формата "key_id: rest of the text" и переводит prefix
 * через словарь. Для ключей, отсутствующих в словаре, возвращает строку
 * как есть (consider: новые ID без перевода читаемы и не ломают UI).
 *
 * Важно: эвристика срабатывает ТОЛЬКО если prefix матчит regex
 * `^[A-Za-z][A-Za-z0-9_]{2,}: ` (латинский identifier с подчёркиванием
 * + двоеточие + пробел) — т.е. обычные русские строки "Mn=1.2 wt%, что
 * выше…" не задеваются (нет latin-only prefix перед ':').
 *
 * Минимальная длина identifier — 3 символа, чтобы не путать с
 * `Pcm:` / `Mn:` / `Cr:` (chemical-shorthand prefixes).
 *
 * @param {string} line — строка вида "logical_leap: текст" или просто "текст"
 * @param {Record<string, string>} dictionary
 * @returns {string} — "Логический скачок: текст" или исходная строка
 */
export function splitKeyedLine(line, dictionary) {
  if (line == null) return '';
  const s = String(line);
  // Anchor на начало; identifier — латинский snake_case ≥ 3 chars; затем «: »
  const m = s.match(/^([A-Za-z][A-Za-z0-9_]{2,}):\s+(.*)$/s);
  if (!m) return s;
  const [, key, rest] = m;
  if (!dictionary || !Object.prototype.hasOwnProperty.call(dictionary, key)) {
    // Неизвестный английский ключ — оставляем как есть; альтернативно
    // можно вернуть `${labelFor(key, {})}: ${rest}`, но риск ложного
    // срабатывания на mid-sentence latin-токены делает это нежелательным.
    return s;
  }
  return `${dictionary[key]}: ${rest}`;
}
