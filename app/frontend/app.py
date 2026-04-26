"""
Steel AI MVP — Streamlit UI для демо.

Запуск:
    PYTHONPATH=. streamlit run app/frontend/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so ANTHROPIC_API_KEY etc. are available when launched via `streamlit run`
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yaml
from datetime import date

from app.backend.cost_model import (
    PriceSnapshot, Material, seed_snapshot, load_snapshot,
    PriceSnapshotIncomplete,
)

st.set_page_config(page_title="Steel AI — HSLA Design", layout="wide", page_icon="⚙️")


def _gate_login() -> None:
    expected_login = os.environ.get("APP_LOGIN")
    expected_password = os.environ.get("APP_PASSWORD")
    if not expected_login or not expected_password:
        return  # auth disabled (e.g. local dev)

    if st.session_state.get("authenticated"):
        return

    st.title("🔒 Steel AI MVP")
    st.caption("Вход в демонстрационное приложение")
    with st.form("login_form"):
        login = st.text_input("Логин")
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Войти", type="primary")
        if submitted:
            if login == expected_login and password == expected_password:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Неверный логин или пароль")
    st.stop()


_gate_login()


def _snapshot_to_editor_df(snapshot: PriceSnapshot) -> pd.DataFrame:
    rows = []
    for m in snapshot.materials.values():
        elems_str = ";".join(f"{k}={v:.2f}" for k, v in m.element_content.items())
        rows.append({
            "id": m.id, "kind": m.kind,
            "price_per_kg": m.price_per_kg,
            "element_content": elems_str,
        })
    return pd.DataFrame(rows)


def _editor_df_to_snapshot(
    df: pd.DataFrame, snap_date: date, currency: str, source: str
) -> PriceSnapshot:
    materials = {}
    for _, row in df.iterrows():
        mid = str(row["id"]).strip()
        if not mid or mid == "nan":
            continue
        ec_str = str(row["element_content"])
        ec = {}
        for pair in ec_str.split(";"):
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            try:
                ec[k.strip()] = float(v)
            except ValueError:
                continue
        materials[mid] = Material(
            id=mid,
            kind=str(row["kind"]),
            price_per_kg=float(row["price_per_kg"]),
            element_content=ec,
        )
    return PriceSnapshot(
        date=snap_date, currency=currency, materials=materials, source=source
    )


# =========================================================================
# Sidebar — модель, статус
# =========================================================================

st.sidebar.title("Steel AI MVP")
st.sidebar.caption("HSLA Pipeline Steels — Demo")

models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(exist_ok=True)
available_models = sorted([d.name for d in models_dir.iterdir() if d.is_dir()])

if available_models:
    st.sidebar.success(f"Моделей обучено: {len(available_models)}")
    selected_model = st.sidebar.selectbox("Активная модель", available_models, index=len(available_models) - 1)
else:
    st.sidebar.warning("Моделей нет. Сначала обучите.")
    selected_model = None

# Class badge for active model
if selected_model:
    try:
        import json as _json
        _meta_path = PROJECT_ROOT / "models" / selected_model / "meta.json"
        _meta = _json.loads(_meta_path.read_text(encoding="utf-8"))
        _class_id = _meta.get("steel_class", "pipe_hsla")
        _class_label = {
            "pipe_hsla": "🔩 Pipe HSLA",
            "en10083_qt": "🔨 EN 10083 Q&T",
            "fatigue_carbon_steel": "🔁 Carbon Fatigue (Agrawal NIMS)",
        }.get(_class_id, _class_id)
        st.sidebar.caption(f"Класс: **{_class_label}**")
        _meta_target = _meta.get("target", "?")
        st.sidebar.caption(f"Target: `{_meta_target}`")
    except Exception:
        pass

st.sidebar.divider()

# Decision Log stats
try:
    from decision_log.logger import query_decisions
    all_decisions = query_decisions(limit=100)
    st.sidebar.metric("Решений в логе", len(all_decisions))
    if all_decisions:
        last = all_decisions[0]
        st.sidebar.caption(f"Последнее: {last['decision'][:40]}")
except Exception as e:
    st.sidebar.error(f"Decision Log: {e}")

# LLM-Critic status
_llm_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
st.sidebar.metric(
    "🤖 Critic на основе глубокой нейронной сети",
    "✓ активен" if _llm_ok else "— нет ключа",
)


# =========================================================================
# Main tabs
# =========================================================================

(tab_design, tab_train, tab_predict, tab_deox, tab_hyp, tab_recipe,
 tab_al, tab_history) = st.tabs([
    "🎯 Дизайн сплава",
    "🤖 Обучение модели",
    "📊 Прогноз",
    "🔥 Раскисление",
    "💡 Гипотезы",
    "🧪 Подбор рецепта",
    "🔭 Следующие эксперименты",
    "📚 История",
])


def _tab_intro(
    purpose: str, audience: list[str], steps: list[str],
    expanded: bool = False,
) -> None:
    """Унифицированная справка по вкладке: назначение, кому подходит,
    порядок работы. Раскрытие по требованию (collapsed по умолчанию)."""
    with st.expander("ℹ️ Назначение, аудитория, порядок работы", expanded=expanded):
        st.markdown(f"**Назначение.** {purpose}")
        st.markdown("**Кому подходит:**")
        for a in audience:
            st.markdown(f"- {a}")
        st.markdown("**Порядок работы:**")
        for i, s in enumerate(steps, 1):
            st.markdown(f"{i}. {s}")


# =========================================================================
# Tab 1: Inverse design
# =========================================================================

with tab_design:
    st.header("Поиск состава под целевые свойства")
    _tab_intro(
        purpose=(
            "Подобрать оптимальный химический состав стали под заданные "
            "целевые свойства (предел текучести, прочность, удлинение, "
            "ударная вязкость) при минимизации стоимости ферросплавов. "
            "Используется NSGA-II — multi-objective evolutionary "
            "optimization над composition+process пространством, "
            "с ML-моделью как property-predictor и ferroalloy-cost-model "
            "как €-predictor."
        ),
        audience=[
            "**R&D-инженер по разработке марок стали** "
            "(composition development engineer)",
            "**Технолог-металлург сталеплавильного производства**, "
            "формирующий рецептуру под новый заказ",
            "**Process metallurgist** в industrial R&D — оптимизация "
            "имеющихся марок под новые ценовые условия ferroalloy "
            "рынка",
            "На традиционном меткомбинате — сотрудник **ЦЗЛ "
            "(центральной заводской лаборатории)** или **отдела "
            "технологии стали**",
        ],
        steps=[
            "Убедиться что в sidebar выбрана активная модель нужного "
            "класса стали",
            "Установить **целевой диапазон** свойств "
            "(например, σт = 485-580 МПа для класса X70)",
            "Выбрать режим стоимости: `full` (alloy + scrap base) или "
            "`alloy_only` (только ферросплавы)",
            "Запустить дизайн — NSGA-II пробежит ~60 поколений с "
            "population 80, выдаст Pareto-фронт ~10-30 кандидатов",
            "Каждый кандидат показывает прогноз свойств с CI, €/т, "
            "расход ферросплавов по материалам",
        ],
    )
    st.caption("Задайте ТЗ — получите Pareto-оптимальные кандидаты с прогнозом и валидацией")

    # Inverse design is HSLA-only in this iteration
    _design_class_id = "pipe_hsla"
    if selected_model:
        try:
            import json as _json
            _meta_path_d = PROJECT_ROOT / "models" / selected_model / "meta.json"
            _design_class_id = _json.loads(_meta_path_d.read_text()).get(
                "steel_class", "pipe_hsla"
            )
        except Exception:
            pass

    if _design_class_id != "pipe_hsla":
        # Forward-compatible guard: ANY non-HSLA class blocked from inverse design.
        # Earlier draft only blocked en10083_qt — fatigue_carbon_steel slipped
        # through and crashed pymoo with KeyError on missing process columns
        # (R-006 follow-up bug fix).
        _class_label = {
            "en10083_qt": "EN 10083-2 Q&T (carbon)",
            "fatigue_carbon_steel": "Carbon Fatigue (Agrawal NIMS)",
        }.get(_design_class_id, _design_class_id)
        st.info(
            f"ℹ️ Inverse design пока работает только для **Pipe HSLA**. "
            f"Активный класс — **{_class_label}**, для него используйте "
            f"вкладку «📊 Прогноз». "
            f"Поддержка inverse design для других классов запланирована "
            f"на v2 (требует расширения NSGA-II под process-параметры "
            f"и переработку variable_bounds логики)."
        )
    else:

        col1, col2 = st.columns(2)
    
        with col1:
            st.subheader("Целевые свойства")
            yt_min = st.number_input("σт минимум, МПа", 380, 800, 485, step=5)
            yt_max = st.number_input("σт максимум, МПа", 400, 900, 580, step=5)
    
        with col2:
            st.subheader("Ограничения")
            cev_max = st.number_input("CEV(IIW) максимум", 0.30, 0.60, 0.43, step=0.01)
            pcm_max = st.number_input("Pcm максимум", 0.15, 0.35, 0.22, step=0.01)
    
        with st.expander("Дополнительные параметры NSGA-II"):
            c1, c2 = st.columns(2)
            pop_size = c1.slider("Population size", 30, 200, 80)
            n_gen = c2.slider("Generations", 20, 200, 60)
    
        st.divider()
        with st.expander("💰 Прайс материалов", expanded=True):
            if "price_snapshot" not in st.session_state:
                st.session_state["price_snapshot"] = seed_snapshot()

            snap: PriceSnapshot = st.session_state["price_snapshot"]

            cols = st.columns([2, 1, 1, 1])
            use_cost = cols[0].checkbox(
                "Учитывать стоимость в оптимизации", value=True, key="use_cost"
            )
            cols[1].metric("Валюта", snap.currency)
            cols[2].metric("Дата", snap.date.isoformat())
            cost_mode = cols[3].radio(
                "Режим cost", ["full", "incremental"],
                horizontal=False, key="cost_mode"
            )

            uploaded = st.file_uploader("⬆ Загрузить YAML-прайс", type=["yaml", "yml"])
            if uploaded is not None:
                import tempfile
                with tempfile.NamedTemporaryFile(
                    suffix=".yaml", delete=False,
                ) as tmp_file:
                    tmp_file.write(uploaded.read())
                    tmp_path = Path(tmp_file.name)
                try:
                    st.session_state["price_snapshot"] = load_snapshot(tmp_path)
                    st.success(f"Загружено: {uploaded.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Не удалось загрузить: {e}")
                finally:
                    tmp_path.unlink(missing_ok=True)

            df_editor = _snapshot_to_editor_df(snap)
            edited = st.data_editor(
                df_editor, num_rows="dynamic", key="price_editor",
                width="stretch",
                column_config={
                    "id": "ID",
                    "kind": st.column_config.SelectboxColumn(
                        "kind", options=["base", "ferroalloy", "pure"]
                    ),
                    "price_per_kg": st.column_config.NumberColumn(
                        f"{snap.currency}/кг", min_value=0.0
                    ),
                    "element_content": "element_content (Mn=0.80;Fe=0.20)",
                },
            )

            # Persist edits back into snapshot so they're used on run.
            try:
                st.session_state["price_snapshot"] = _editor_df_to_snapshot(
                    edited, snap.date, snap.currency, source="manual"
                )
            except Exception as e:
                st.error(f"Ошибка парсинга прайса: {e}")

            # Download button
            snap_now = st.session_state["price_snapshot"]
            snap_yaml = yaml.safe_dump({
                "date": snap_now.date.isoformat(),
                "currency": snap_now.currency,
                "source": "manual",
                "materials": {
                    mid: {
                        "kind": m.kind,
                        "price_per_kg": m.price_per_kg,
                        "element_content": dict(m.element_content),
                    }
                    for mid, m in snap_now.materials.items()
                },
            }, sort_keys=False, allow_unicode=True)
            st.download_button(
                "💾 Скачать текущий прайс как YAML",
                data=snap_yaml,
                file_name=f"prices_{snap.date.isoformat()}.yaml",
            )

        if st.button("🚀 Запустить дизайн", type="primary", disabled=not selected_model):
            if not selected_model:
                st.error("Сначала обучите модель")
            else:
                snapshot = (
                    st.session_state.get("price_snapshot")
                    if st.session_state.get("use_cost", True) else None
                )
                mode = st.session_state.get("cost_mode", "full")

                with st.spinner("NSGA-II оптимизация..."):
                    from app.backend.inverse_designer import run_inverse_design
                    from app.backend.validator import validate_batch

                    try:
                        result = run_inverse_design(
                            model_version=selected_model,
                            targets={"yield_strength_mpa": {"min": yt_min, "max": yt_max}},
                            hard_constraints={"cev_iiw": {"max": cev_max}, "pcm": {"max": pcm_max}},
                            population_size=pop_size,
                            n_generations=n_gen,
                            price_snapshot=snapshot,
                            cost_mode=mode,
                        )
                    except PriceSnapshotIncomplete as e:
                        st.error(
                            f"❌ В прайсе нет цен для: **{', '.join(e.missing)}**. "
                            f"Добавьте строки в таблицу «Прайс материалов» и повторите запуск."
                        )
                        st.stop()

                    val_result = validate_batch(result["pareto_candidates"])
                    st.session_state["last_design"] = {
                        "inverse": result,
                        "validation": val_result,
                    }
    
        # Отображение результата
        if "last_design" in st.session_state:
            d = st.session_state["last_design"]
            inverse = d["inverse"]
            validation = d["validation"]
        
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Найдено кандидатов", inverse["n_candidates"])
            c2.metric("Прошли валидацию", len(validation["approved"]))
            c3.metric("Отсеяно", len(validation["rejected"]))
            c4.metric("С warnings", sum(1 for c in validation["approved"] if c.get("validation", {}).get("overall") == "PASS_WITH_WARNINGS"))
        
            if validation["rejection_summary"]:
                with st.expander("Причины отсева"):
                    for reason, count in validation["rejection_summary"].items():
                        st.write(f"- **{reason}**: {count}")

            # Pareto plot (σт × cost) — Task 12
            candidates_for_plot = inverse["pareto_candidates"]
            if candidates_for_plot:
                df_pareto = pd.DataFrame([{
                    "idx": c["idx"],
                    "sigma_t": c["predicted"]["mean"],
                    "ci_half": c["predicted"]["ci_half_width"],
                    "cost": (c["cost"]["total_per_ton"] if c.get("cost")
                             else c["objectives"]["alloying_cost"]),
                    "ood": "OOD" if c["predicted"]["ood_flag"] else "ok",
                } for c in candidates_for_plot])

                cost_currency = inverse.get("cost_currency", "EUR (legacy)")
                st.subheader("Pareto front")
                chart = (
                    alt.Chart(df_pareto)
                    .mark_circle(size=140)
                    .encode(
                        x=alt.X("sigma_t:Q", title="σт, МПа"),
                        y=alt.Y("cost:Q", title=f"Стоимость, {cost_currency}/т"),
                        color=alt.Color(
                            "ood:N",
                            scale=alt.Scale(domain=["ok", "OOD"],
                                            range=["#2ecc71", "#e67e22"]),
                        ),
                        tooltip=["idx", "sigma_t", "ci_half", "cost", "ood"],
                    )
                    .interactive()
                )
                st.altair_chart(chart, width="stretch")

            st.subheader("Топ-5 кандидатов")
        
            top5 = validation["approved"][:5] if validation["approved"] else inverse["pareto_candidates"][:5]
            for i, c in enumerate(top5, 1):
                comp = c.get("composition", {})
                derived = c.get("derived", {})
                pred = c.get("predicted", {})
                val = c.get("validation", {})
                overall = val.get("overall", "PASS")
                emoji = {"PASS": "✅", "PASS_WITH_WARNINGS": "⚠️", "FAIL": "❌"}.get(overall, "❔")
            
                with st.expander(f"{emoji} Кандидат #{i} — σт = {pred.get('mean', 0):.0f} ± {pred.get('ci_half_width', 0):.0f} МПа"):
                    cc1, cc2, cc3 = st.columns(3)
                    with cc1:
                        st.markdown("**Химия (%)**")
                        non_zero = {k: v for k, v in comp.items() if v > 0.001}
                        st.dataframe(
                            pd.DataFrame([non_zero]).T.rename(columns={0: "%"}),
                            width="stretch",
                        )
                    with cc2:
                        st.markdown("**Обработка**")
                        proc = c.get("processing", {})
                        for k, v in proc.items():
                            st.write(f"**{k}:** {v}")
                        st.markdown("**Производные**")
                        st.write(f"CEV = **{derived.get('cev_iiw', 0):.3f}**")
                        st.write(f"Pcm = **{derived.get('pcm', 0):.3f}**")
                        st.write(f"CEN = **{derived.get('cen', 0):.3f}**")
                    with cc3:
                        st.markdown("**Прогноз**")
                        st.metric("σт", f"{pred.get('mean', 0):.0f} МПа",
                                 f"±{pred.get('ci_half_width', 0):.0f} (90% CI)")
                        st.write(f"Lower 90%: {pred.get('lower_90', 0):.0f}")
                        st.write(f"Upper 90%: {pred.get('upper_90', 0):.0f}")
                        st.write(f"OOD flag: {'⚠️ Да' if pred.get('ood_flag') else '✓ Нет'}")
                        # Keep the legacy summary only when no cost breakdown is available
                        if not c.get("cost"):
                            st.markdown("**Стоимость (legacy)**")
                            st.write(f"≈ {c.get('objectives', {}).get('alloying_cost', 0):.1f} €/т")

                    if c.get("cost"):
                        cb = c["cost"]
                        st.markdown(
                            f"**💰 Себестоимость:** "
                            f"{cb['total_per_ton']:,.0f} {cb['currency']}/т "
                            f"({cb['total_per_ton']/1000:,.2f} {cb['currency']}/кг, "
                            f"{cb['mode']})"
                        )
                        df_bd = pd.DataFrame(cb["contributions"])
                        if not df_bd.empty:
                            df_bd["share_%"] = (
                                df_bd["contribution_per_ton"] / cb["total_per_ton"] * 100
                            ).round(1)
                            df_bd = df_bd[[
                                "material_id",
                                "mass_kg_per_ton_steel",
                                "price_per_kg",
                                "contribution_per_ton",
                                "share_%",
                            ]]
                            df_bd.columns = [
                                "Материал", "Масса, кг/т",
                                f"Цена, {cb['currency']}/кг",
                                f"Вклад, {cb['currency']}/т", "Доля, %",
                            ]
                            st.dataframe(df_bd, width="stretch", hide_index=True)
                            st.download_button(
                                f"📋 Экспорт breakdown #{c['idx']} в CSV",
                                data=df_bd.to_csv(index=False).encode("utf-8"),
                                file_name=f"breakdown_candidate_{c['idx']}.csv",
                                key=f"dl_bd_{c['idx']}",
                            )

                    if val.get("warnings"):
                        st.warning("Предупреждения: " + "; ".join(w["message"] for w in val["warnings"]))
                    if val.get("failed_checks"):
                        st.error("Failed: " + "; ".join(w["message"] for w in val["failed_checks"]))


    # =========================================================================
# Tab 2: Train model
# =========================================================================

with tab_train:
    st.header("Обучение модели")
    st.caption("Обучает XGBoost с quantile regression для uncertainty estimation")
    _tab_intro(
        purpose=(
            "Обучить XGBoost-модель composition→property для выбранного "
            "класса стали. Применяется time-based split + GroupKFold "
            "(защита от data leakage), quantile regression для 90% CI, "
            "GMM OOD-детектор по composition, split-conformal калибровка "
            "интервалов. Optuna автоматически подбирает гиперпараметры."
        ),
        audience=[
            "**Data scientist / ML-engineer** в R&D отделе или digital "
            "advanced analytics group меткомбината",
            "**Materials informatics specialist** в академической "
            "research group",
            "**Аспирант / постдок** в materials science — обучение под "
            "научную задачу",
            "На больших комбинатах в "
            "крупного R&D отдела) — сотрудник digital metallurgy team",
        ],
        steps=[
            "Выбрать **класс стали** из dropdown "
            "(pipe_hsla / en10083_qt / fatigue_carbon_steel)",
            "Выбрать **target property** — модель строится для одного "
            "целевого свойства за раз",
            "Установить **Optuna trials** (40 — стандартный режим; "
            "80-100 для финальной production-модели; 10-20 для quick "
            "experimentation)",
            "Нажать «🤖 Обучить» — занимает 1-5 минут в зависимости "
            "от trials и размера данных",
            "После обучения модель появится в sidebar dropdown "
            "и станет доступна для прогноза, дизайна, вкладок с глубокой нейронной сетью",
        ],
    )

    from app.backend.steel_classes import (
        available_steel_classes,
        compute_features_for_class,
        get_synthetic_generator,
        load_steel_class,
    )

    _classes = available_steel_classes()
    _class_opts = {c.id: f"{c.name} ({c.standard})" for c in _classes}
    selected_class_id = st.selectbox(
        "Класс стали",
        options=[c.id for c in _classes],
        format_func=lambda cid: _class_opts[cid],
        key="train_class",
    )
    _profile = load_steel_class(selected_class_id)

    c1, c2 = st.columns(2)
    target_col = c1.selectbox(
        "Target property",
        options=[t.id for t in _profile.target_properties],
        format_func=lambda tid: next(
            t.label for t in _profile.target_properties if t.id == tid
        ),
    )
    n_trials = c2.slider(
        "Optuna trials (чем больше, тем лучше, но медленнее)", 10, 150, 40,
    )

    st.info(
        f"ℹ️ Выбран класс: **{_profile.name}** · стандарт {_profile.standard}. "
        f"Feature set: {len(_profile.feature_set)} колонок. "
        f"Обучение займёт 1-5 минут в зависимости от количества trials."
    )

    if st.button("🤖 Обучить модель", type="primary"):
        with st.spinner("Generating dataset & training..."):
            from app.backend.model_trainer import train_model

            gen = get_synthetic_generator(_profile.synthetic_generator_name)
            df_raw = gen()
            df_feat = compute_features_for_class(df_raw, selected_class_id)
            feat = [f for f in _profile.feature_set if f in df_feat.columns]

            progress = st.progress(0, text="Запускаю обучение...")
            trained = train_model(
                df_feat, target_col, feat,
                n_optuna_trials=n_trials,
                steel_class=selected_class_id,
            )
            progress.progress(100, text="Готово!")
            
            st.success(f"✅ Модель {trained.version} готова")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R² test", f"{trained.metrics.r2_test:.3f}")
            m2.metric("MAE test", f"{trained.metrics.mae_test:.2f}")
            m3.metric("R² train", f"{trained.metrics.r2_train:.3f}")
            m4.metric("Coverage 90% CI", f"{trained.metrics.coverage_90_ci:.1%}")
            
            # Critic-like warnings
            from pattern_library.patterns import run_all_patterns, Phase
            critic_ctx = {
                "r2_train": trained.metrics.r2_train,
                "r2_val": trained.metrics.r2_val,
                "r2_test": trained.metrics.r2_test,
                "mae_test": trained.metrics.mae_test,
                "rmse_test": trained.metrics.rmse_test,
                "coverage_90_ci": trained.metrics.coverage_90_ci,
                "n_train": trained.metrics.n_train,
                "n_val": trained.metrics.n_val,
                "n_test": trained.metrics.n_test,
                "prediction_has_ci": True,
                "has_time_column": True,
                "has_groups": True,
                "split_strategy": "time_based",
                "cv_strategy": "group_kfold",
                "feature_importance": trained.feature_importance,
                "training_ranges": trained.training_ranges,
                "steel_class": selected_class_id,
                "expected_top_features": _profile.expected_top_features,
                "physical_bounds": _profile.physical_bounds,
                "ood_detector_configured": True,
                "target": target_col,
            }
            warnings = run_all_patterns(critic_ctx, phase=Phase.TRAINING)
            if warnings:
                st.subheader("⚠️ Отчёт Critic")
                for w in warnings:
                    sev = w["severity"]
                    msg = f"**[{sev}] {w['pattern_id']}:** {w['message']}\n\n💡 {w['suggestion']}"
                    if sev == "HIGH":
                        st.error(msg)
                    elif sev == "MEDIUM":
                        st.warning(msg)
                    else:
                        st.info(msg)
            else:
                st.success("✓ Critic не нашёл проблем")

            # LLM-Critic — only runs with ANTHROPIC_API_KEY
            from app.backend.critic_llm import make_llm_critic
            from dataclasses import asdict
            _llm = make_llm_critic()
            if _llm is not None:
                with st.spinner("🤖 Critic на основе глубокой нейронной сети проверяет..."):
                    llm_obs = _llm.review_training(critic_ctx)
                    st.session_state["llm_observations"] = [
                        asdict(o) for o in llm_obs
                    ]

            llm_obs_rendered = st.session_state.get("llm_observations", [])
            if llm_obs_rendered:
                st.subheader("🤖 Critic на основе глубокой нейронной сети")
                for o in llm_obs_rendered:
                    sev = o["severity"]
                    msg = (f"**[{sev}] {o['category']}:** {o['message']}\n\n"
                           f"💡 {o['rationale']}")
                    if sev == "HIGH":
                        st.error(msg)
                    elif sev == "MEDIUM":
                        st.warning(msg)
                    else:
                        st.info(msg)
            elif _llm is not None:
                st.caption("🤖 Critic на основе глубокой нейронной сети: проблем не обнаружено")

            # Feature importance chart
            st.subheader("Feature importance")
            imp_df = pd.DataFrame(
                sorted(trained.feature_importance.items(), key=lambda x: -x[1])[:15],
                columns=["feature", "importance"],
            )
            st.bar_chart(imp_df.set_index("feature"))


# =========================================================================
# Tab 3: Single prediction
# =========================================================================

with tab_predict:
    st.header("Прогноз для заданного состава")
    st.caption("Введите химию и режим — получите прогноз с uncertainty")
    _tab_intro(
        purpose=(
            "Точечный прогноз механического свойства по введённой "
            "композиции и параметрам процесса. Возвращает **point "
            "estimate** + **90% доверительный интервал** "
            "(conformal-corrected), а также **OOD-флаг** — сигнал "
            "что введённая композиция вне обучающего распределения "
            "и прогноз ненадёжен."
        ),
        audience=[
            "**Технолог сталеплавильного цеха** — проверка expected "
            "свойств перед запуском плавки",
            "**Инженер ОТК / отдела технического контроля** — "
            "входной контроль scrap по химии и расчёт ожидаемых "
            "механических характеристик",
            "**Старший металлург** дуговой / конвертерной печи / ladle",
            "**Специалист производственной лаборатории** на участке "
            "термообработки (расчёт отпускных режимов под целевую "
            "твёрдость)",
        ],
        steps=[
            "Убедиться что выбрана нужная **активная модель** в sidebar "
            "(класс стали должен соответствовать вашей задаче)",
            "Ввести **композицию** (wt%) — поля подстраиваются под "
            "feature_set активной модели",
            "Ввести **параметры процесса** (температуры, времена, "
            "скорости охлаждения)",
            "Получить **прогноз свойства** + 90% CI + ⚠️ OOD-флаг "
            "если состав вне training distribution",
            "При OOD — переключиться на вкладку «🔬 Анализ "
            "аномалий» (через CLI скрипт) или скорректировать "
            "входные параметры",
        ],
    )

    if not selected_model:
        st.warning("Сначала обучите модель")
    else:
        import json as _json
        from app.backend.model_trainer import load_model, predict_with_uncertainty
        from app.backend.steel_classes import (
            compute_features_for_class, load_steel_class,
        )

        _meta_path_p = PROJECT_ROOT / "models" / selected_model / "meta.json"
        _meta_p = _json.loads(_meta_path_p.read_text())
        _class_id_p = _meta_p.get("steel_class", "pipe_hsla")
        _profile_p = load_steel_class(_class_id_p)

        st.caption(f"Класс: **{_profile_p.name}** · target: `{_meta_p['target']}`")

        row = {}
        cols_per_row = 4
        features_ui = [f for f in _profile_p.feature_set if f != "n_ppm"]
        for chunk_start in range(0, len(features_ui), cols_per_row):
            chunk = features_ui[chunk_start:chunk_start + cols_per_row]
            cc = st.columns(len(chunk))
            for col_idx, feat in enumerate(chunk):
                lo, hi = _profile_p.physical_bounds.get(feat, (0.0, 1.0))
                default = (lo + hi) / 2
                step = (hi - lo) / 100 if (hi - lo) > 0 else 0.01
                fmt = "%.4f" if feat.endswith("_pct") else "%.2f"
                row[feat] = cc[col_idx].number_input(
                    feat, min_value=float(lo), max_value=float(hi),
                    value=float(default), step=float(step),
                    key=f"pred_{feat}", format=fmt,
                )
        if "n_ppm" in _profile_p.feature_set:
            row["n_ppm"] = st.number_input(
                "n_ppm", 20.0, 100.0, 55.0, step=5.0, key="pred_n_ppm",
            )

        if st.button("🔮 Предсказать", type="primary"):
            df_input = pd.DataFrame([row])
            df_feat = compute_features_for_class(df_input, _class_id_p)

            bundle = load_model(selected_model)
            pred = predict_with_uncertainty(bundle, df_feat)

            mean = float(pred["prediction"].iloc[0])
            lo_p = float(pred["lower_90"].iloc[0])
            hi_p = float(pred["upper_90"].iloc[0])
            ood = bool(pred["ood_flag"].iloc[0])

            _tgt_label = next(
                (t.label for t in _profile_p.target_properties
                 if t.id == _meta_p["target"]),
                _meta_p["target"],
            )
            st.subheader(f"{_tgt_label}: **{mean:.1f}** ± {(hi_p - lo_p) / 2:.1f}")
            st.caption(f"90% ДИ: [{lo_p:.1f}, {hi_p:.1f}]")

            if ood:
                st.error("⚠️ Состав вне training distribution — прогноз ненадёжен!")

            if _class_id_p == "pipe_hsla" and {
                "cev_iiw", "pcm", "cen", "microalloying_sum"
            }.issubset(df_feat.columns):
                st.markdown("**Производные параметры:**")
                c1d, c2d, c3d, c4d = st.columns(4)
                c1d.metric("CEV(IIW)", f"{df_feat['cev_iiw'].iloc[0]:.3f}")
                c2d.metric("Pcm", f"{df_feat['pcm'].iloc[0]:.3f}")
                c3d.metric("CEN", f"{df_feat['cen'].iloc[0]:.3f}")
                c4d.metric("Микролегирование", f"{df_feat['microalloying_sum'].iloc[0]:.4f}")

            # Anomaly Explainer — показываем кнопку если OOD или CI слишком широкий
            ci_width = hi_p - lo_p
            target_range = next(
                (t.range for t in _profile_p.target_properties
                 if t.id == _meta_p["target"]),
                [0, 1],
            )
            target_span = target_range[1] - target_range[0]
            wide_ci = ci_width > 0.5 * target_span if target_span > 0 else False

            if (ood or wide_ci) and _llm_ok:
                st.divider()
                with st.expander(
                    "🔬 PhD-диагностика аномалии",
                    expanded=ood,  # авто-раскрытие при OOD
                ):
                    st.markdown(
                        "**Почему этот прогноз требует осторожности.** "
                        "Состав/параметры процесса либо вне training "
                        "distribution (OOD), либо CI слишком широкий "
                        "относительно целевого диапазона свойства. "
                        "PhD-металлург на основе глубокой нейронной сети разберёт по полочкам какие "
                        "фичи аномальны, какие mechanism-риски, что "
                        "произойдёт в производстве и как скорректировать."
                    )
                    if st.button(
                        "🔬 Объяснить почему рискованно",
                        type="primary", key="explain_ood_btn",
                        help="Анализ глубокой нейронной сети ~30-40 секунд, ~$0.05-0.07.",
                    ):
                        from app.backend.anomaly_explainer import (
                            make_anomaly_explainer,
                        )
                        explainer = make_anomaly_explainer()
                        if explainer is None:
                            st.error("AnomalyExplainer недоступен")
                        else:
                            tr = _meta_p["training_ranges"]
                            recipe = {
                                f: float(df_feat[f].iloc[0])
                                for f in df_feat.columns
                                if f in tr
                            }
                            out_of_range = []
                            for f, v in recipe.items():
                                lo_f, hi_f = tr[f]
                                if v < lo_f or v > hi_f:
                                    out_of_range.append({
                                        "feature": f, "value": v,
                                        "training_range": [lo_f, hi_f],
                                    })
                            with st.spinner(
                                "PhD-диагностика…"
                            ):
                                exp = explainer.explain({
                                    "model_version": selected_model,
                                    "steel_class": _class_id_p,
                                    "target": _meta_p["target"],
                                    "recipe": recipe,
                                    "training_ranges": tr,
                                    "training_medians": {
                                        f: (lo_f + hi_f) / 2
                                        for f, (lo_f, hi_f) in tr.items()
                                    },
                                    "ml_prediction": {
                                        "predicted": mean,
                                        "lower_90": lo_p,
                                        "upper_90": hi_p,
                                        "ci_width": ci_width,
                                    },
                                    "ood_flag": ood,
                                    "ood_score": float(
                                        pred["log_density"].iloc[0]
                                    ),
                                    "out_of_range_features": out_of_range,
                                })
                            if exp is None:
                                st.error("Объяснение не получено")
                            else:
                                sev_color = {
                                    "LOW": "#558ccc",
                                    "MEDIUM": "#f17105",
                                    "HIGH": "#d11149",
                                }.get(exp.severity, "#888")
                                sev_label = {
                                    "LOW": "НИЗКАЯ",
                                    "MEDIUM": "СРЕДНЯЯ",
                                    "HIGH": "ВЫСОКАЯ",
                                }.get(exp.severity, exp.severity)
                                st.markdown(
                                    f"<div style='background:{sev_color};"
                                    f"color:white;padding:8px 12px;"
                                    f"border-radius:6px;font-weight:600;"
                                    f"display:inline-block'>"
                                    f"Опасность: {sev_label}</div>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown(f"**Резюме.** {exp.summary}")

                                if exp.anomalous_features:
                                    st.markdown("**Аномальные параметры:**")
                                    for af in exp.anomalous_features:
                                        st.markdown(
                                            f"- `{af.feature}` = "
                                            f"{af.value:.4f} "
                                            f"(training "
                                            f"[{af.training_range[0]:.4f}, "
                                            f"{af.training_range[1]:.4f}], "
                                            f"тип: {af.deviation_kind})"
                                        )
                                        st.caption(af.note)

                                if exp.mechanism_concerns:
                                    st.markdown("**Mechanism-риски:**")
                                    for m in exp.mechanism_concerns:
                                        st.markdown(f"- {m}")

                                st.markdown(
                                    f"**Производственные риски.** "
                                    f"{exp.production_risks}"
                                )
                                st.info(
                                    f"**Рекомендуемая правка.** "
                                    f"{exp.suggested_correction}"
                                )


# =========================================================================
# Tab: Al Deoxidation Calculator (on-line LF advisory)
# =========================================================================

with tab_deox:
    st.header("🔥 Раскисление жидкой стали алюминием")
    st.caption(
        "Physics-based advisory на базе 3 термодинамических моделей "
        "**+ PhD ladle metallurgist на основе глубокой нейронной сети + adversarial критик**. "
        "Расчёт на каждую плавку с full operator protocol."
    )
    _tab_intro(
        purpose=(
            "Двухслойный расчёт раскисления стали в ковше "
            "(ladle furnace).\n\n"
            "**Слой 1 — physics-based.** Forward — сколько Al подать "
            "чтобы снизить активный кислород до target. Inverse — "
            "оценка эффективной чистоты Al-проволоки по факту замера "
            "post-melt. Compare — три термодинамические модели "
            "(Fruehan 1985, Sigworth-Elliott 1974, Hayashi-Yamamoto "
            "2013) рядом для cross-validation.\n\n"
            "**Слой 2 — советник на основе глубокой нейронной сети + PhD-критик** (sub-tab «🤖 "
            "советник на основе глубокой нейронной сети + критик»). агент ранга senior ladle "
            "metallurgist превращает 3 thermo числа в полный "
            "operator protocol: Al kg + форма (wire/cube/powder) + "
            "rate подачи + recovery factor + kinetic timing + "
            "конкретные риски плавки + прогноз включений + "
            "pre/post actions с цифрами + evidence (artifact + "
            "mechanism citations Turkdogan, Cramb, Cicutti, Ghosh). "
            "Второй агент-нейросеть — PhD-критик уровня journal reviewer #2 — "
            "делает adversarial peer review с **построчным fact-check "
            "evidence** (VALID / INVALID / UNVERIFIABLE) и ловит "
            "арифметические ошибки, неверные recovery factors, "
            "missed risks, mechanism inversions."
        ),
        audience=[
            "**Сталевар-разливщик / мастер ladle furnace** — "
            "оперативный расчёт Al per heat (sub-tabs Forward / "
            "Inverse / Compare)",
            "**Технолог цеха внепечной обработки** "
            "(secondary metallurgy) — на сложных плавках включает "
            "советник на основе глубокой нейронной сети для full protocol, особенно когда O_a "
            "высокий, Mn/S пограничный, или slag FeO нестабилен",
            "**Senior ladle metallurgist / руководитель secondary "
            "metallurgy** на крупных комбинатах "
            "высокого класса — "
            "советник на основе глубокой нейронной сети заменяет 30-минутное совещание со старшим "
            "коллегой 3-минутным protocol'ом с PhD-рецензией, "
            "включая math fact-check",
            "**Process engineer / R&D лаборатория секундарной "
            "металлургии** — adversarial critic выявляет ошибки в "
            "evidence на уровне журнальной рецензии (поймал "
            "арифметическую несостыковку 35.4+30=44 в первом live "
            "запуске)",
            "**Инженер по разливке стали** на МНЛЗ — pre/post actions "
            "включают рекомендации по SEN, Ar-продувке, защите от "
            "re-oxidation",
        ],
        steps=[
            "**Sub-tab «Сколько Al нужно» (Forward)**: ввести "
            "измеренную O_a (ppm), target O_a, T (°C), массу плавки "
            "(тонн) → получить Al kg/heat по выбранной thermo-модели",
            "**Sub-tab «Качество Al по факту» (Inverse)**: после "
            "плавки ввести фактический Al и фактический O_final → "
            "получить эффективную чистоту Al-проволоки (audit-метрика "
            "для поставщика)",
            "**Sub-tab «⚖️ Сравнить модели»**: 3 thermo-модели "
            "parallel — если расходятся >20%, physics в этой зоне "
            "сама по себе неточная, решение принимать по запасу",
            "**Sub-tab «🤖 советник на основе глубокой нейронной сети + критик»** (требует "
            "ANTHROPIC_API_KEY): ввести параметры плавки + опционально "
            "композицию + slag FeO + grade target → запустить полный "
            "цикл (~3 минуты, ~$0.20-0.25). Получить:\n"
            "    - operator protocol с metrics (Al kg, форма, recovery, "
            "kinetic timing) + risk flags + inclusion forecast + "
            "pre/post actions + доказательная база;\n"
            "    - PhD-вердикт критика (ACCEPT / REVISE / REJECT) с "
            "построчным fact-check evidence, strengths, weaknesses, "
            "suggested revision",
            "При желании сохранить любой расчёт в **Decision Log** для "
            "audit trail (тэги `deoxidation`, `deoxidation_advisory`, "
            "`deoxidation_review`, `deoxidation_cycle`); audit-ready "
            "для ISO 9001 / IATF 16949 / AS9100 compliance",
        ],
    )

    from app.backend.deoxidation import (
        DEFAULT_MODEL_ID, THERMO_MODELS,
        compute_al_demand, compute_al_quality, compare_all_models,
    )
    from app.backend.steel_classes import load_steel_class
    from pattern_library.patterns import Phase as _PhaseDx, run_all_patterns as _run_dx

    # Context (active model class → target O_a default)
    _active_class_id = "pipe_hsla"
    _target_o_a_default = 10.0
    if selected_model:
        try:
            import json as _json_dx
            _meta_dx = _json_dx.loads(
                (PROJECT_ROOT / "models" / selected_model / "meta.json").read_text()
            )
            _active_class_id = _meta_dx.get("steel_class", "pipe_hsla")
            _profile_dx = load_steel_class(_active_class_id)
            if _profile_dx.target_o_activity_ppm is not None:
                _target_o_a_default = _profile_dx.target_o_activity_ppm
        except Exception:
            pass

    st.markdown(
        f"**Активный класс**: `{_active_class_id}` · "
        f"**Target O_a из профиля**: `{_target_o_a_default} ppm`"
    )

    _model_id = st.selectbox(
        "Термодинамическая модель",
        options=list(THERMO_MODELS.keys()),
        index=list(THERMO_MODELS.keys()).index(DEFAULT_MODEL_ID),
        format_func=lambda mid: f"{THERMO_MODELS[mid].name} — {THERMO_MODELS[mid].citation}",
        key="deox_model_id",
    )

    sub_fwd, sub_inv, sub_cmp, sub_ai = st.tabs([
        "Сколько Al нужно", "Качество Al по факту",
        "⚖️ Сравнить модели", "🤖 советник на основе глубокой нейронной сети + критик",
    ])

    # ──────── Forward ────────
    with sub_fwd:
        cf1, cf2 = st.columns(2)
        o_a_initial = cf1.number_input("O_a измерено, ppm", 0.0, 2000.0, 450.0, step=10.0)
        T_c = cf2.number_input("T расплава, °C", 1400.0, 1700.0, 1620.0, step=5.0)
        cf3, cf4 = st.columns(2)
        mass_t = cf3.number_input("Масса стали, т", 1.0, 500.0, 180.0, step=5.0)
        target_o_a = cf4.number_input(
            "Целевой O_a, ppm", 0.5, 1000.0,
            value=float(_target_o_a_default), step=1.0,
        )
        cf5, cf6 = st.columns(2)
        purity = cf5.number_input("% активного Al", 50.0, 100.0, 100.0, step=1.0)
        burn_off = cf6.number_input("Угар, %", 0.0, 50.0, 20.0, step=1.0)
        heat_id = st.text_input("Heat ID (опционально, для audit)", value="")

        if st.button("🧮 Рассчитать", type="primary", key="deox_fwd_btn"):
            result = compute_al_demand(
                o_a_initial_ppm=o_a_initial, temperature_C=T_c,
                steel_mass_ton=mass_t, target_o_a_ppm=target_o_a,
                al_purity_pct=purity, burn_off_pct=burn_off,
                model_id=_model_id,
            )
            st.session_state["last_deox_result"] = result

            dx_warnings = _run_dx(
                {
                    "o_a_initial_ppm": o_a_initial,
                    "target_o_a_ppm": target_o_a,
                },
                phase=_PhaseDx.DEOXIDATION,
            )
            for w in dx_warnings:
                sev = w["severity"]
                msg = f"**[{sev}] {w['pattern_id']}:** {w['message']}\n\n💡 {w['suggestion']}"
                if sev == "HIGH":
                    st.error(msg)
                elif sev == "MEDIUM":
                    st.warning(msg)
                else:
                    st.info(msg)

            st.divider()
            if result.al_total_kg > 0:
                st.subheader(f"💊 Навеска Al: {result.al_total_kg:.1f} кг ({result.al_per_ton:.3f} кг/т)")
                st.markdown(
                    f"- Активный Al на реакцию: **{result.al_active_kg:.1f} кг**\n"
                    f"- Угар: {result.al_burn_off_kg:.1f} кг ({burn_off:.0f}%)\n"
                    f"- Ожидаемый остаточный O_a: **{result.o_a_expected_ppm:.1f} ppm** (цель)\n"
                    f"- 💰 Стоимость: **{result.cost_eur:.1f} {result.currency}** "
                    f"(при {THERMO_MODELS[_model_id].name})"
                )
                for w in result.warnings:
                    st.warning(w)
            else:
                st.info("Раскисление не требуется (см. warning выше).")

            if st.button("💾 Сохранить в Decision Log", key="deox_save_fwd"):
                from dataclasses import asdict as _asdict
                from decision_log.logger import log_decision
                log_decision(
                    phase="deoxidation",
                    decision=(
                        f"Al-deox {heat_id or 'без ID'}: "
                        f"{result.al_total_kg:.1f} кг на {mass_t} т "
                        f"({result.al_per_ton:.3f} кг/т)"
                    ),
                    reasoning=(
                        f"Model={result.model_id}, "
                        f"O_a {o_a_initial}→{target_o_a} ppm @ {T_c}°C, "
                        f"purity={purity}%, burn_off={burn_off}%. "
                        f"Cost={result.cost_eur:.2f} {result.currency}"
                    ),
                    context={"inputs": result.inputs, "result": _asdict(result)},
                    author="deox_calculator",
                    tags=["deoxidation", "al_deox", _active_class_id,
                          heat_id or "no_id"],
                )
                st.success("Запись сохранена в Decision Log")

    # ──────── Inverse ────────
    with sub_inv:
        st.caption("Плавка уже прошла — оценим эффективное качество поставки Al.")
        ci1, ci2 = st.columns(2)
        pre_o_a = ci1.number_input("O_a до, ppm", 0.0, 2000.0, 500.0, step=10.0, key="inv_pre")
        post_o_a = ci2.number_input("O_a после, ppm", 0.0, 2000.0, 10.0, step=1.0, key="inv_post")
        ci3, ci4 = st.columns(2)
        al_added = ci3.number_input("Al добавлено, кг", 0.1, 5000.0, 65.0, step=1.0)
        T_c_inv = ci4.number_input("T, °C", 1400.0, 1700.0, 1620.0, step=5.0, key="inv_T")
        ci5, ci6 = st.columns(2)
        mass_inv = ci5.number_input("Масса стали, т", 1.0, 500.0, 180.0, step=5.0, key="inv_mass")
        burn_inv = ci6.number_input("Угар (допущение), %", 0.0, 50.0, 20.0, step=1.0, key="inv_burn")

        if st.button("🔍 Оценить качество", type="primary", key="deox_inv_btn"):
            try:
                q_result = compute_al_quality(
                    o_a_before_ppm=pre_o_a, o_a_after_ppm=post_o_a,
                    al_added_kg=al_added, temperature_C=T_c_inv,
                    steel_mass_ton=mass_inv, burn_off_pct=burn_inv,
                    model_id=_model_id,
                )
            except ValueError as e:
                st.error(f"Ошибка ввода: {e}")
                st.stop()

            dx_warnings_inv = _run_dx(
                {"effective_purity_pct": q_result.effective_purity_pct},
                phase=_PhaseDx.DEOXIDATION,
            )
            for w in dx_warnings_inv:
                sev = w["severity"]
                msg = f"**[{sev}] {w['pattern_id']}:** {w['message']}\n\n💡 {w['suggestion']}"
                (st.error if sev == "HIGH" else st.warning)(msg)

            st.divider()
            st.subheader(f"Эффективное активное Al: {q_result.effective_purity_pct:.1f} %")
            st.markdown(
                f"- Реально сработал (связал O): **{q_result.effective_active_kg:.1f} кг**\n"
                f"- Ожидался при 100% чистоте: {q_result.expected_active_kg:.1f} кг\n"
                f"- Допущение burn_off: {q_result.assumed_burn_off_pct:.0f}%"
            )
            for w in q_result.warnings:
                st.warning(w)

    # ──────── Compare ────────
    with sub_cmp:
        st.caption("Запуск всех 3 термодинамических моделей на одних и тех же входах.")
        cc1, cc2 = st.columns(2)
        o_a_cmp = cc1.number_input("O_a измерено, ppm", 0.0, 2000.0, 450.0, step=10.0, key="cmp_o_a")
        T_cmp = cc2.number_input("T, °C", 1400.0, 1700.0, 1620.0, step=5.0, key="cmp_T")
        cc3, cc4 = st.columns(2)
        mass_cmp = cc3.number_input("Масса, т", 1.0, 500.0, 180.0, step=5.0, key="cmp_mass")
        target_cmp = cc4.number_input(
            "Целевой O_a, ppm", 0.5, 1000.0, float(_target_o_a_default),
            step=1.0, key="cmp_target",
        )
        cc5, cc6 = st.columns(2)
        purity_cmp = cc5.number_input("% Al", 50.0, 100.0, 100.0, step=1.0, key="cmp_pur")
        burn_cmp = cc6.number_input("Угар, %", 0.0, 50.0, 20.0, step=1.0, key="cmp_burn")

        if st.button("⚖️ Сравнить все 3 модели", type="primary", key="deox_cmp_btn"):
            cmp_results = compare_all_models(
                o_a_initial_ppm=o_a_cmp, temperature_C=T_cmp,
                steel_mass_ton=mass_cmp, target_o_a_ppm=target_cmp,
                al_purity_pct=purity_cmp, burn_off_pct=burn_cmp,
            )
            df_cmp = pd.DataFrame([{
                "Модель": THERMO_MODELS[r.model_id].name,
                "Al, кг": round(r.al_total_kg, 2),
                "Al, кг/т": round(r.al_per_ton, 4),
                "O_a, ppm": round(r.o_a_expected_ppm, 1),
                f"Цена, {r.currency}": round(r.cost_eur, 2),
            } for r in cmp_results])
            st.dataframe(df_cmp, hide_index=True, width="stretch")

            masses = [r.al_total_kg for r in cmp_results]
            spread_pct = (max(masses) - min(masses)) / (sum(masses) / 3.0) * 100
            st.caption(
                f"Разброс между моделями: ±{spread_pct:.1f} %. "
                f"Это ожидаемая неопределённость между академическими "
                f"термодинамическими формулами."
            )

            chart_df = pd.DataFrame({
                "Модель": [THERMO_MODELS[r.model_id].name for r in cmp_results],
                "Al, кг": [r.al_total_kg for r in cmp_results],
            })
            chart = alt.Chart(chart_df).mark_bar().encode(
                x="Модель:N", y="Al, кг:Q",
                color=alt.Color("Модель:N", legend=None),
            )
            st.altair_chart(chart, width="stretch")

    # ──────── AI Advisor + PhD Critic ────────
    with sub_ai:
        st.markdown(
            "**Полный operator protocol** на ladle treatment: к 3 thermo-"
            "числам PhD ladle metallurgist на основе глубокой нейронной сети добавляет форму Al, "
            "rate подачи, kinetic timing, риски, прогноз включений, "
            "pre/post actions. Второй агент-нейросеть — PhD-критик уровня "
            "journal reviewer #2 — даёт adversarial peer review с "
            "построчной проверкой evidence."
        )

        if not _llm_ok:
            st.warning("ANTHROPIC_API_KEY не задан — советник на основе глубокой нейронной сети недоступен.")
        else:
            st.markdown("##### Параметры плавки")
            ai_c1, ai_c2 = st.columns(2)
            ai_oa_init = ai_c1.number_input(
                "O_a измеренный, ppm",
                0.0, 2000.0, 280.0, step=10.0, key="ai_oa_init",
            )
            ai_target = ai_c2.number_input(
                "Target O_a, ppm",
                0.5, 50.0, 5.0, step=0.5, key="ai_target",
            )
            ai_c3, ai_c4 = st.columns(2)
            ai_T = ai_c3.number_input(
                "T расплава, °C",
                1400.0, 1700.0, 1580.0, step=5.0, key="ai_T",
            )
            ai_mass = ai_c4.number_input(
                "Масса стали, т",
                1.0, 500.0, 100.0, step=5.0, key="ai_mass",
            )

            st.markdown("##### Композиция (опционально, помогает критику ловить риски)")
            cm1, cm2, cm3, cm4 = st.columns(4)
            ai_c_pct = cm1.number_input("C, wt%", 0.0, 1.5, 0.20, step=0.01, key="ai_c")
            ai_mn_pct = cm2.number_input("Mn, wt%", 0.0, 3.0, 0.85, step=0.05, key="ai_mn")
            ai_si_pct = cm3.number_input("Si, wt%", 0.0, 2.5, 0.30, step=0.05, key="ai_si")
            ai_s_pct = cm4.number_input("S, wt%", 0.0, 0.05, 0.012, step=0.002, key="ai_s")
            cm5, cm6, cm7 = st.columns(3)
            ai_p_pct = cm5.number_input("P, wt%", 0.0, 0.05, 0.018, step=0.002, key="ai_p")
            ai_slag_feo = cm6.number_input(
                "Slag FeO, %", 0.0, 15.0, 2.5, step=0.5, key="ai_slag_feo",
            )
            ai_grade = cm7.text_input(
                "Целевой grade / задача", value="строительная конструкционная",
                key="ai_grade",
            )

            run_ai_btn = st.button(
                "🤖 Получить полный protocol с PhD-рецензией",
                type="primary", key="ai_run_btn",
                help="Полный цикл ~3 минуты, ~$0.20-0.25.",
            )

            if run_ai_btn:
                from dataclasses import asdict as _asdict
                from app.backend.deoxidation import compare_all_models
                from app.backend.deoxidation_advisor import (
                    make_deoxidation_advisor,
                )
                from app.backend.deoxidation_critic import (
                    make_deoxidation_critic,
                )

                advisor = make_deoxidation_advisor()
                critic = make_deoxidation_critic()
                if advisor is None or critic is None:
                    st.error("Advisor или critic недоступны")
                else:
                    progress = st.progress(0, text="Считаю 3 thermo-модели…")
                    cmp_res = compare_all_models(
                        o_a_initial_ppm=ai_oa_init,
                        target_o_a_ppm=ai_target,
                        temperature_C=ai_T,
                        steel_mass_ton=ai_mass,
                        al_purity_pct=99.7,
                        burn_off_pct=20.0,
                    )
                    thermo_estimates = {
                        r.model_id: round(r.al_total_kg, 2)
                        for r in cmp_res
                    }
                    heat_context = {
                        "o_a_init_ppm": float(ai_oa_init),
                        "target_o_a_ppm": float(ai_target),
                        "temp_c": float(ai_T),
                        "mass_t": float(ai_mass),
                        "composition": {
                            "c_pct": float(ai_c_pct),
                            "mn_pct": float(ai_mn_pct),
                            "si_pct": float(ai_si_pct),
                            "s_pct": float(ai_s_pct),
                            "p_pct": float(ai_p_pct),
                            "mn_s_ratio": float(ai_mn_pct / max(ai_s_pct, 1e-6)),
                        },
                        "slag_feo_pct": float(ai_slag_feo),
                        "grade_target": ai_grade,
                    }
                    ctx = {
                        "heat_context": heat_context,
                        "thermo_estimates": thermo_estimates,
                    }

                    progress.progress(20, text="нейросеть формирует protocol (~80 с)…")
                    advisory = advisor.advise(ctx)
                    if advisory is None:
                        progress.empty()
                        st.error("Advisor вернул None")
                    else:
                        progress.progress(60, text="PhD-критик делает peer-review (~80 с)…")
                        verdict = critic.review(ctx, _asdict(advisory))
                        progress.progress(95, text="Сохраняю результат…")
                        from decision_log.logger import log_decision
                        log_decision(
                            phase="deoxidation",
                            decision=(
                                f"Deox cycle: Al={advisory.al_addition_kg:.1f} kg "
                                f"({advisory.al_form}), "
                                f"критик={verdict.verdict if verdict else 'N/A'}"
                            ),
                            reasoning=(
                                f"O_a {ai_oa_init}→{ai_target} ppm, "
                                f"T={ai_T}°C, mass={ai_mass}т"
                            ),
                            context={
                                "heat_context": heat_context,
                                "thermo_estimates": thermo_estimates,
                                "advisory": _asdict(advisory),
                                "review": (
                                    _asdict(verdict) if verdict else None
                                ),
                            },
                            author="ui",
                            tags=["deoxidation_cycle", "sonnet-4-6"],
                        )
                        progress.progress(100, text="Готово")
                        st.success(
                            f"Protocol получен. Вердикт критика: "
                            f"{verdict.verdict if verdict else 'нет'}"
                        )

                        # Render advisory
                        st.divider()
                        st.markdown("### 🤖 Operator protocol")
                        st.markdown(f"**Резюме.** {advisory.summary}")

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Al total, кг", f"{advisory.al_addition_kg:.1f}")
                        m2.metric("Форма", advisory.al_form)
                        m3.metric("Recovery, %", f"{advisory.expected_recovery_pct:.0f}")
                        m4.metric(
                            "Время до target",
                            f"{advisory.kinetic_timing_min[0]:.0f}-"
                            f"{advisory.kinetic_timing_min[1]:.0f} мин",
                        )

                        st.markdown(f"**Стратегия подачи.** {advisory.addition_strategy}")
                        st.markdown(
                            f"**Сходимость 3 thermo-моделей.** "
                            f"{advisory.model_convergence_note}"
                        )

                        if advisory.risk_flags:
                            st.markdown("**⚠️ Риски этой плавки:**")
                            for r in advisory.risk_flags:
                                st.markdown(f"- {r}")

                        st.markdown(
                            f"**Прогноз включений.** {advisory.inclusion_forecast}"
                        )

                        pp1, pp2 = st.columns(2)
                        pp1.markdown("**До добавки Al:**")
                        for a in advisory.pre_actions:
                            pp1.markdown(f"- {a}")
                        pp2.markdown("**После добавки Al:**")
                        for a in advisory.post_actions:
                            pp2.markdown(f"- {a}")

                        st.markdown("**Доказательная база:**")
                        for ev in advisory.evidence:
                            st.markdown(f"- {ev}")

                        st.caption(
                            f"Уверенность советника: {advisory.confidence}  ·  "
                            f"id={advisory.id}"
                        )

                        # Render critic verdict
                        if verdict is not None:
                            st.divider()
                            v_color = {
                                "ACCEPT": "#3a9d23",
                                "REVISE": "#e0a800",
                                "REJECT": "#c0392b",
                            }.get(verdict.verdict, "#888")
                            v_label = {
                                "ACCEPT": "ПРИНЯТО",
                                "REVISE": "ТРЕБУЕТ ПРАВОК",
                                "REJECT": "ОТКЛОНЕНО",
                            }.get(verdict.verdict, verdict.verdict)
                            st.markdown(
                                f"<div style='display:flex;align-items:center;"
                                f"gap:12px;margin-bottom:6px'>"
                                f"<span style='background:{v_color};color:white;"
                                f"padding:4px 10px;border-radius:4px;"
                                f"font-weight:600'>👨‍🔬 PhD-критик: "
                                f"{v_label}</span>"
                                f"<span style='color:#666;font-size:0.9em'>"
                                f"уверенность {verdict.confidence}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"_{verdict.summary}_")

                            if verdict.evidence_check:
                                st.markdown("**Fact-check доказательной базы:**")
                                ec_mark = {
                                    "VALID": "✓", "INVALID": "✗",
                                    "UNVERIFIABLE": "?",
                                }
                                for ec in verdict.evidence_check:
                                    mark = ec_mark.get(ec.verdict, "•")
                                    st.markdown(
                                        f"- {mark} **{ec.claim}** — {ec.note}"
                                    )

                            sl, sr = st.columns(2)
                            if verdict.strengths:
                                sl.markdown("**Сильные стороны**")
                                for s in verdict.strengths:
                                    sl.markdown(f"- {s}")
                            if verdict.weaknesses:
                                sr.markdown("**Слабые стороны**")
                                for w in verdict.weaknesses:
                                    sr.markdown(f"- {w}")

                            if verdict.suggested_revision:
                                st.info(
                                    f"**Предложение правки.** "
                                    f"{verdict.suggested_revision}"
                                )


# =========================================================================
# Tab 4: Decision Log
# =========================================================================

# =========================================================================
# Tab 5: Hypotheses (LLM-generated, A2 from AI integration roadmap)
# =========================================================================

with tab_hyp:
    st.header("💡 Гипотезы от наблюдателя на основе глубокой нейронной сети")
    st.caption(
        "Глубокая нейронная сеть просматривает обученную модель и предлагает testable гипотезы "
        "с оценкой экономического эффекта vs классическая практика."
    )
    _tab_intro(
        purpose=(
            "агент-нейросеть смотрит на обученную модель и формулирует 3-5 "
            "testable research hypotheses — predictions, которые можно "
            "проверить экспериментально и которые имеют **экономический "
            "эффект vs классическая практика** (trial-and-error / "
            "handbook recipe / Thermo-Calc / substitution baseline). "
            "Второй агент-нейросеть — PhD-критик — делает adversarial peer review "
            "каждой гипотезы и выдаёт ACCEPT / REVISE / REJECT с "
            "построчной проверкой evidence."
        ),
        audience=[
            "**Materials scientist в академической research group** — "
            "гипотезы для следующей публикации в Acta Materialia, MSE A, "
            "Metall Mater Trans A",
            "**R&D research engineer** — formulation testable claims "
            "для experimental campaign на pilot line",
            "**Scientific PI / руководитель R&D-группы** — выявление "
            "новых исследовательских направлений из накопленных данных",
            "**Postgraduate / postdoc в materials informatics** — "
            "обоснование experimental hypothesis defense'а",
        ],
        steps=[
            "Убедиться что выбрана модель и **ANTHROPIC_API_KEY** "
            "задан в `.env`",
            "Нажать «🔮 Сгенерировать гипотезы и провести рецензию» "
            "(полный цикл ~3 минуты, ~$0.14)",
            "Получить 5 карточек: каждая с **statement / обоснование "
            "/ proposed_experiment / expected_outcome / экономический "
            "эффект**",
            "Под каждой карточкой — **PhD-рецензия**: вердикт + "
            "fact-check evidence + сильные / слабые стороны + правки",
            "ACCEPT-гипотезы — кандидаты на experimental verification; "
            "REVISE — посмотреть suggested_revision и переформулировать; "
            "REJECT — пропустить",
        ],
    )

    if not selected_model:
        st.warning("Сначала выберите активную модель в sidebar (или обучите).")
    elif not _llm_ok:
        st.warning(
            "ANTHROPIC_API_KEY не задан в окружении. "
            "Hypothesis Generator недоступен."
        )
    else:
        from decision_log.logger import query_decisions

        st.markdown(f"**Активная модель:** `{selected_model}`")

        existing_runs = [
            d for d in query_decisions(phase="training", limit=200)
            if "hypothesis_cycle" in (d.get("tags") or [])
            and d.get("context", {}).get("model_version") == selected_model
        ]
        st.caption(
            f"Прошлых циклов на этой модели: **{len(existing_runs)}**"
            + (
                f" · последний {existing_runs[0]['timestamp'][:16]}"
                if existing_runs else ""
            )
        )

        col_n, col_filter = st.columns([2, 3])
        with col_n:
            max_hyp = st.slider(
                "Сколько идей сгенерировать (потолок)", 3, 15, 5,
                help=(
                    "Sonnet выдаст до этого числа гипотез, "
                    "но остановится раньше, если идеи кончатся "
                    "(во избежание filler'а)."
                ),
            )
        with col_filter:
            show_only_accept = st.checkbox(
                "Показывать только ACCEPT от PhD-Critic",
                value=True,
                help=(
                    "Если включено — REJECT/REVISE гипотезы скрыты. "
                    "Это и есть ответ на вопрос «сколько реально интересных»."
                ),
            )

        run_btn = st.button(
            "🔮 Сгенерировать гипотезы и провести рецензию",
            type="primary",
            help=(
                "Полный цикл: генератор (~100 с, ~$0.08) → критик-PhD "
                "(~45 с, ~$0.06). Итого ~2.5 минуты, ~$0.14."
            ),
        )

        if run_btn:
            from dataclasses import asdict as _asdict
            from scripts.generate_hypotheses_for_model import build_context
            from app.backend.hypothesis_generator import make_hypothesis_generator
            from app.backend.hypothesis_critic import make_hypothesis_critic
            from decision_log.logger import log_decision

            gen = make_hypothesis_generator()
            crit = make_hypothesis_critic()
            if gen is None or crit is None:
                st.error("Генератор или критик недоступны (нет ключа?)")
            else:
                progress = st.progress(0, text="Подготовка контекста модели…")
                ctx = build_context(selected_model)
                ctx["max_hypotheses"] = max_hyp
                progress.progress(15, text=f"Генератор формулирует до {max_hyp} гипотез (~100 с)…")
                new_hypotheses = gen.generate(ctx)
                if not new_hypotheses:
                    progress.empty()
                    st.error(
                        "Получено 0 гипотез. Проверьте логи / model artifact."
                    )
                else:
                    progress.progress(60, text="Критик-PhD рецензирует (~45 с)…")
                    hyp_dicts = [_asdict(h) for h in new_hypotheses]
                    verdicts = crit.review(ctx, hyp_dicts)
                    reviews_dicts = [_asdict(v) for v in verdicts]
                    progress.progress(95, text="Сохраняю результаты…")

                    verdict_counts = {"ACCEPT": 0, "REVISE": 0, "REJECT": 0}
                    for v in verdicts:
                        verdict_counts[v.verdict] = verdict_counts.get(v.verdict, 0) + 1

                    log_decision(
                        phase="training",
                        decision=(
                            f"Hypothesis cycle: {len(new_hypotheses)} гипотез, "
                            f"вердикты A={verdict_counts['ACCEPT']} "
                            f"R={verdict_counts['REVISE']} "
                            f"X={verdict_counts['REJECT']}"
                        ),
                        reasoning=(
                            f"Model={selected_model}, "
                            f"hypotheses={len(new_hypotheses)}, "
                            f"reviews={len(verdicts)}"
                        ),
                        context={
                            "model_version": selected_model,
                            "hypotheses": hyp_dicts,
                            "reviews": reviews_dicts,
                            "verdict_counts": verdict_counts,
                        },
                        author="ui",
                        tags=["hypothesis_cycle", "sonnet-4-6"],
                    )
                    progress.progress(100, text="Готово")
                    st.success(
                        f"Получено {len(new_hypotheses)} гипотез, "
                        f"{len(verdicts)} рецензий "
                        f"(ACCEPT={verdict_counts['ACCEPT']} · "
                        f"REVISE={verdict_counts['REVISE']} · "
                        f"REJECT={verdict_counts['REJECT']})"
                    )
                    st.rerun()

        display_runs = existing_runs[:1]
        if not display_runs:
            st.info(
                "Циклов ещё нет — нажмите кнопку выше чтобы запустить "
                "первый разбор (генератор + рецензия)."
            )
        else:
            run = display_runs[0]
            ctx_data = run.get("context", {})
            cycle_hyps = ctx_data.get("hypotheses", [])
            cycle_reviews = {
                r["hypothesis_id"]: r
                for r in ctx_data.get("reviews", [])
            }
            verdict_counts = ctx_data.get("verdict_counts", {})

            cols = st.columns(4)
            cols[0].metric("Гипотез", len(cycle_hyps))
            cols[1].metric(
                "ACCEPT",
                verdict_counts.get("ACCEPT", 0),
                help="Принято критиком как есть",
            )
            cols[2].metric(
                "REVISE",
                verdict_counts.get("REVISE", 0),
                help="Требует правок",
            )
            cols[3].metric(
                "REJECT",
                verdict_counts.get("REJECT", 0),
                help="Отклонено",
            )

            novelty_color = {
                "HIGH": "#d11149",
                "MEDIUM": "#f17105",
                "LOW": "#558ccc",
            }
            novelty_label = {
                "HIGH": "ВЫСОКАЯ", "MEDIUM": "СРЕДНЯЯ", "LOW": "НИЗКАЯ",
            }
            cost_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            cost_label = {
                "LOW": "низкая", "MEDIUM": "средняя", "HIGH": "высокая",
            }
            verdict_color = {
                "ACCEPT": "#3a9d23",
                "REVISE": "#e0a800",
                "REJECT": "#c0392b",
            }
            verdict_label = {
                "ACCEPT": "ПРИНЯТО",
                "REVISE": "ТРЕБУЕТ ПРАВОК",
                "REJECT": "ОТКЛОНЕНО",
            }
            confidence_label = {
                "HIGH": "высокая", "MEDIUM": "средняя", "LOW": "низкая",
            }

            visible_hyps = cycle_hyps
            if show_only_accept:
                visible_hyps = [
                    h for h in cycle_hyps
                    if cycle_reviews.get(h.get("hypothesis_id"), {}).get("verdict") == "ACCEPT"
                ]
                hidden = len(cycle_hyps) - len(visible_hyps)
                if hidden:
                    st.caption(
                        f"Скрыто {hidden} гипотез с вердиктом REVISE/REJECT. "
                        "Снимите галочку «только ACCEPT» чтобы увидеть все."
                    )
                if not visible_hyps:
                    st.warning(
                        "Ни одна из сгенерированных гипотез не получила "
                        "ACCEPT от PhD-Critic — модель насыщенная или "
                        "идеи требуют доработки. Снимите галочку чтобы "
                        "увидеть REVISE/REJECT с обоснованиями."
                    )

            for i, h in enumerate(visible_hyps, start=1):
                novelty = h.get("novelty", "?")
                cost = h.get("experiment_cost_estimate", "?")
                color = novelty_color.get(novelty, "#888")
                with st.container(border=True):
                    title_col, badge_col = st.columns([8, 2])
                    title_col.markdown(
                        f"### {i}. {h.get('statement', '—')}"
                    )
                    badge_col.markdown(
                        f"<div style='text-align:right'>"
                        f"<span style='background:{color};color:white;"
                        f"padding:3px 8px;border-radius:4px;"
                        f"font-size:0.85em'>новизна: "
                        f"{novelty_label.get(novelty, novelty)}</span><br>"
                        f"<span style='font-size:0.85em'>"
                        f"{cost_emoji.get(cost, '⚪')} стоимость: "
                        f"{cost_label.get(cost, cost)}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Обоснование.** {h.get('rationale', '—')}")

                    pe = h.get("proposed_experiment", {})
                    fix = pe.get("fix", {})
                    sweep = pe.get("sweep", {})
                    st.markdown("**Предлагаемый эксперимент.**")
                    fc, sc = st.columns(2)
                    fc.markdown("Зафиксировать:")
                    fc.json(fix)
                    sc.markdown("Варьировать:")
                    sc.json(sweep)

                    st.markdown(
                        f"**Ожидаемый результат.** {h.get('expected_outcome', '—')}"
                    )

                    ei = h.get("economic_impact", {})
                    st.markdown("**Экономический эффект.**")
                    st.markdown(
                        f"- Сравнение с классикой: "
                        f"{ei.get('vs_classical_baseline', '—')}\n"
                        f"- Оценка экономии: "
                        f"**{ei.get('estimated_saving', '—')}**\n"
                        f"- Метод проверки: "
                        f"{ei.get('measurement_method', '—')}"
                    )

                    rv = cycle_reviews.get(h.get("id"))
                    if rv:
                        st.divider()
                        v_color = verdict_color.get(rv["verdict"], "#888")
                        v_label = verdict_label.get(rv["verdict"], rv["verdict"])
                        c_label = confidence_label.get(
                            rv["confidence"], rv["confidence"]
                        )
                        st.markdown(
                            f"<div style='display:flex;align-items:center;"
                            f"gap:12px;margin-bottom:6px'>"
                            f"<span style='background:{v_color};color:white;"
                            f"padding:4px 10px;border-radius:4px;"
                            f"font-weight:600'>👨‍🔬 Рецензия PhD: "
                            f"{v_label}</span>"
                            f"<span style='color:#666;font-size:0.9em'>"
                            f"уверенность {c_label}</span></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"_{rv['summary']}_")

                        sc_l, sc_r = st.columns(2)
                        if rv.get("strengths"):
                            sc_l.markdown("**Сильные стороны**")
                            for s in rv["strengths"]:
                                sc_l.markdown(f"- {s}")
                        if rv.get("weaknesses"):
                            sc_r.markdown("**Слабые стороны**")
                            for w in rv["weaknesses"]:
                                sc_r.markdown(f"- {w}")

                        if rv.get("suggested_revision"):
                            st.info(
                                f"**Предложение правки:** "
                                f"{rv['suggested_revision']}"
                            )

                    st.caption(
                        f"id={h.get('id', '?')} · теги: "
                        f"{', '.join(h.get('tags', []))}"
                    )


# =========================================================================
# Tab 6: Recipe Designer + PhD Critic — composition cycle
# =========================================================================

with tab_recipe:
    st.header("🧪 Подбор рецепта — PhD-пара на основе глубокой нейронной сети")
    st.caption(
        "агент-нейросеть проектирует рецепт с двойной evidence (artifact + mechanism), "
        "ML+cost проверяет численно, второй агент-критик делает PhD peer-review с "
        "построчной проверкой evidence."
    )
    _tab_intro(
        purpose=(
            "AI-driven подбор production-рецептa с двойной "
            "доказательной базой. designer-нейросеть формирует 3-4 "
            "альтернативные композиции; каждое изменение легирующего "
            "элемента обосновано **одновременно** artifact-данными "
            "(feature_importance, training_ranges) **и** classical "
            "metallurgical mechanism (Hall-Petch, Hollomon-Jaffe, "
            "Grossmann's DI, Pickering, Andrews equations, CEV/Pcm). "
            "Затем XGBoost+cost_model численно проверяют каждый "
            "рецепт. критик на основе глубокой нейронной сети уровня journal reviewer #2 делает "
            "adversarial review с построчным fact-check evidence."
        ),
        audience=[
            "**Senior R&D engineer по разработке составов** "
            "(composition design lead)",
            "**Materials scientist в industrial R&D-лаборатории** — "
            "при работе над новыми grade'ами стали или оптимизации "
            "существующих под обновлённые ferroalloy цены",
            "**Process metallurgist готовящий новую recipe для "
            "pilot heat** — рецепт с full evidence + PhD review "
            "выходит в опытное производство с минимальным риском",
            "**Старший инженер ЦЗЛ** или **отдела технологии стали** "
            "на традиционных меткомбинатах",
        ],
        steps=[
            "Сформулировать **задачу проектирования** в текстовом поле "
            "(default — снизить cost при сохранении σ свойства)",
            "Запустить полный цикл (~3 минуты, ~$0.20-0.25)",
            "Получить 3-4 рецепта с **verdict-бейджами** "
            "(ACCEPT / REVISE / REJECT)",
            "Для каждого рецепта изучить: **обоснование**, "
            "**доказательную базу** (artifact + mechanism), **PhD "
            "fact-check** каждой строки evidence (✓ VALID, "
            "✗ INVALID, ? UNVERIFIABLE), **strengths / weaknesses**, "
            "**предложение правки**",
            "**ACCEPT-рецепты** — готовые кандидаты на pilot heat "
            "с прозрачной evidence-картой; **REVISE** — посмотреть "
            "suggested_revision и переподать; **REJECT** — отклонить",
        ],
    )

    if not selected_model:
        st.warning("Сначала выберите активную модель в sidebar.")
    elif not _llm_ok:
        st.warning("ANTHROPIC_API_KEY не задан — recipe pair недоступен.")
    elif _class_id != "fatigue_carbon_steel":
        st.warning(
            "Recipe pair сейчас работает только на классе "
            "`fatigue_carbon_steel` (Agrawal NIMS). Выберите такую модель "
            "в sidebar."
        )
    else:
        from decision_log.logger import query_decisions

        st.markdown(f"**Активная модель:** `{selected_model}`")

        existing_runs = [
            d for d in query_decisions(phase="inverse_design", limit=200)
            if "recipe_cycle" in (d.get("tags") or [])
            and d.get("context", {}).get("model_version") == selected_model
        ]
        st.caption(
            f"Прошлых циклов на этой модели: **{len(existing_runs)}**"
            + (
                f" · последний {existing_runs[0]['timestamp'][:16]}"
                if existing_runs else ""
            )
        )

        default_task = (
            "Снизить ferroalloy cost vs baseline при сохранении или "
            "улучшении fatigue strength. Целевой компромисс: каждый "
            "−€10/т имеет ценность если |Δσf| ≤ 30 МПа."
        )
        task_text = st.text_area(
            "Задача проектирования", value=default_task, height=85,
        )

        run_btn = st.button(
            "🧪 Запустить полный цикл (designer + ML + critic)",
            type="primary",
            help="~3 минуты, ~$0.20-0.25 за полный цикл.",
        )

        if run_btn:
            from dataclasses import asdict as _asdict
            import json as _json
            import numpy as _np
            import pandas as _pd

            from app.backend.cost_model import compute_cost, seed_snapshot
            from app.backend.data_curator import (
                load_real_agrawal_fatigue_dataset,
            )
            from app.backend.model_trainer import (
                load_model as _load_m, predict_with_uncertainty,
            )
            from app.backend.recipe_designer import make_recipe_designer
            from app.backend.recipe_critic import make_recipe_critic
            from decision_log.logger import log_decision

            DESIGN_COMPOSITION = [
                "si_pct", "mn_pct", "ni_pct", "cr_pct", "cu_pct", "mo_pct",
            ]
            DESIGN_PROCESS = [
                "normalizing_temp_c",
                "carburizing_temp_c",
                "carburizing_time_min",
                "tempering_temp_c",
                "tempering_time_min",
                "through_hardening_cooling_rate_c_per_s",
            ]

            designer = make_recipe_designer()
            critic = make_recipe_critic()
            if designer is None or critic is None:
                st.error("Designer или critic недоступны")
            else:
                progress = st.progress(0, text="Загрузка модели и baseline…")

                bundle = _load_m(selected_model)
                meta = bundle["meta"]
                target = meta["target"]

                df_raw = load_real_agrawal_fatigue_dataset()
                if "sub_class" in df_raw.columns:
                    sub = df_raw[df_raw["sub_class"] == "carbon_low_alloy"]
                    if len(sub) < 50:
                        sub = df_raw
                else:
                    sub = df_raw
                baseline = sub.select_dtypes(include=[_np.number]).median()

                snapshot = seed_snapshot()
                feature_list = meta["feature_list"]
                X_base = _pd.DataFrame(
                    [[float(baseline[f]) for f in feature_list]],
                    columns=feature_list,
                )
                base_pred = predict_with_uncertainty(bundle, X_base).iloc[0]
                baseline_predicted = float(base_pred["prediction"])
                baseline_comp = {
                    k: float(v) for k, v in baseline.items()
                    if k.endswith("_pct")
                }
                baseline_cost = compute_cost(
                    baseline_comp, snapshot, mode="full",
                ).total_per_ton

                avail_comp = [c for c in DESIGN_COMPOSITION if c in feature_list]
                avail_proc = [p for p in DESIGN_PROCESS if p in feature_list]
                baseline_recipe = {
                    **{k: float(baseline[k]) for k in avail_comp},
                    **{k: float(baseline[k]) for k in avail_proc},
                }

                designer_ctx = {
                    "task": task_text,
                    "steel_class": meta.get("steel_class"),
                    "target": target,
                    "data_source": meta.get("data_source"),
                    "model_version": selected_model,
                    "r2_test": meta["metrics"]["r2_test"],
                    "mae_test": meta["metrics"]["mae_test"],
                    "coverage_90_ci": meta["metrics"]["coverage_90_ci"],
                    "conformal_correction_mpa": meta.get(
                        "conformal_correction_mpa", 0,
                    ),
                    "feature_importance": meta["feature_importance"],
                    "training_ranges": meta["training_ranges"],
                    "target_distribution": {
                        "min": float(df_raw[target].min()),
                        "max": float(df_raw[target].max()),
                        "mean": float(df_raw[target].mean()),
                        "std": float(df_raw[target].std()),
                        "n": int(len(df_raw)),
                    },
                    "baseline_recipe": baseline_recipe,
                    "baseline_predicted_property": baseline_predicted,
                    "baseline_cost_per_ton": float(baseline_cost),
                    "available_composition": avail_comp,
                    "available_process": avail_proc,
                }

                progress.progress(15, text="Designer формулирует рецепты (~80 с)…")
                recipes = designer.design(designer_ctx)
                if not recipes:
                    progress.empty()
                    st.error("Designer вернул 0 рецептов")
                else:
                    progress.progress(50, text="ML+cost проверяет рецепты…")
                    recipes_with_v = []
                    for r in recipes:
                        row = baseline.copy()
                        for k, v in r.composition.items():
                            if k in row.index: row[k] = float(v)
                        for k, v in r.process_params.items():
                            if k in row.index: row[k] = float(v)
                        X_r = _pd.DataFrame(
                            [[float(row[f]) for f in feature_list]],
                            columns=feature_list,
                        )
                        pp = predict_with_uncertainty(bundle, X_r).iloc[0]
                        comp = {
                            k: float(v) for k, v in row.items()
                            if k.endswith("_pct")
                        }
                        try:
                            cb = compute_cost(comp, snapshot, mode="full")
                            cost_pt = cb.total_per_ton
                            ferro = [
                                {
                                    "material": c.material_id,
                                    "kg_per_ton": round(c.mass_kg_per_ton_steel, 2),
                                    "eur_per_ton": round(c.contribution_per_ton, 2),
                                }
                                for c in cb.contributions
                                if c.material_id != "scrap"
                            ]
                        except Exception:
                            cost_pt = None; ferro = []
                        ml = {
                            "predicted_property": float(pp["prediction"]),
                            "lower_90": float(pp["lower_90"]),
                            "upper_90": float(pp["upper_90"]),
                            "ood_flag": bool(pp["ood_flag"]),
                            "cost_per_ton": cost_pt,
                            "delta_property": float(pp["prediction"]) - baseline_predicted,
                            "delta_cost": (
                                cost_pt - baseline_cost
                                if cost_pt is not None else None
                            ),
                            "ferroalloy_breakdown": ferro,
                        }
                        d = _asdict(r); d["ml_verification"] = ml
                        recipes_with_v.append(d)

                    progress.progress(70, text="Critic делает PhD-рецензию (~100 с)…")
                    verdicts = critic.review(designer_ctx, recipes_with_v)
                    progress.progress(95, text="Сохраняю результаты…")

                    counts = {"ACCEPT": 0, "REVISE": 0, "REJECT": 0}
                    for v in verdicts:
                        counts[v.verdict] = counts.get(v.verdict, 0) + 1

                    log_decision(
                        phase="inverse_design",
                        decision=(
                            f"Recipe cycle: {len(recipes)} рецептов, "
                            f"A={counts['ACCEPT']} R={counts['REVISE']} "
                            f"X={counts['REJECT']}"
                        ),
                        reasoning=(
                            f"model={selected_model}, "
                            f"baseline σf={baseline_predicted:.0f} МПа, "
                            f"cost={baseline_cost:.2f} €/т"
                        ),
                        context={
                            "model_version": selected_model,
                            "baseline": {
                                "recipe": baseline_recipe,
                                "predicted_property": baseline_predicted,
                                "cost_per_ton": float(baseline_cost),
                            },
                            "recipes": recipes_with_v,
                            "reviews": [_asdict(v) for v in verdicts],
                            "verdict_counts": counts,
                        },
                        author="ui",
                        tags=["recipe_cycle", "sonnet-4-6"],
                    )
                    progress.progress(100, text="Готово")
                    st.success(
                        f"Получено {len(recipes)} рецептов, "
                        f"вердикты A={counts['ACCEPT']} "
                        f"R={counts['REVISE']} X={counts['REJECT']}"
                    )
                    st.rerun()

        if not existing_runs:
            st.info(
                "Циклов ещё нет — нажмите кнопку выше чтобы запустить "
                "первый подбор рецепта."
            )
        else:
            run = existing_runs[0]
            ctx_d = run.get("context", {})
            base = ctx_d.get("baseline", {})
            cycle_recipes = ctx_d.get("recipes", [])
            cycle_reviews = {
                rv["recipe_id"]: rv for rv in ctx_d.get("reviews", [])
            }
            counts = ctx_d.get("verdict_counts", {})

            cb1, cb2, cb3, cb4 = st.columns(4)
            cb1.metric(
                "Baseline σf",
                f"{base.get('predicted_property', 0):.0f} МПа",
            )
            cb2.metric(
                "Baseline cost",
                f"{base.get('cost_per_ton', 0):.2f} €/т",
            )
            cb3.metric("Рецептов", len(cycle_recipes))
            cb4.metric(
                "ACCEPT / REVISE / REJECT",
                f"{counts.get('ACCEPT', 0)} / {counts.get('REVISE', 0)} / "
                f"{counts.get('REJECT', 0)}",
            )

            verdict_color = {
                "ACCEPT": "#3a9d23",
                "REVISE": "#e0a800",
                "REJECT": "#c0392b",
            }
            verdict_label = {
                "ACCEPT": "ПРИНЯТО",
                "REVISE": "ТРЕБУЕТ ПРАВОК",
                "REJECT": "ОТКЛОНЕНО",
            }
            confidence_label = {
                "HIGH": "высокая", "MEDIUM": "средняя", "LOW": "низкая",
            }
            ec_mark = {"VALID": "✓", "INVALID": "✗", "UNVERIFIABLE": "?"}

            for i, r in enumerate(cycle_recipes, 1):
                ml = r.get("ml_verification", {})
                rv = cycle_reviews.get(r.get("id"))
                with st.container(border=True):
                    title_col, badge_col = st.columns([7, 3])
                    title_col.markdown(
                        f"### {i}. {r.get('name', '—')}"
                    )
                    novelty = r.get("novelty", "?")
                    badge_col.markdown(
                        f"<div style='text-align:right'>"
                        f"<span style='background:#666;color:white;"
                        f"padding:3px 8px;border-radius:4px;"
                        f"font-size:0.85em'>новизна: {novelty}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    dp = ml.get("delta_property", 0) or 0
                    dc = ml.get("delta_cost") or 0
                    m1.metric(
                        "Δσf, МПа",
                        f"{dp:+.0f}",
                        help=(
                            f"Прогноз {ml.get('predicted_property', 0):.0f} "
                            f"[{ml.get('lower_90', 0):.0f}, "
                            f"{ml.get('upper_90', 0):.0f}]"
                        ),
                    )
                    m2.metric("Δcost, €/т", f"{dc:+.2f}")
                    m3.metric(
                        "Прогноз σf, МПа",
                        f"{ml.get('predicted_property', 0):.0f}",
                        help=(
                            f"90% CI [{ml.get('lower_90', 0):.0f}, "
                            f"{ml.get('upper_90', 0):.0f}]"
                        ),
                    )
                    ood = ml.get("ood_flag", False)
                    m4.metric(
                        "OOD",
                        "⚠️ ДА" if ood else "✓ нет",
                    )

                    st.markdown(f"**Обоснование.** {r.get('rationale', '—')}")

                    st.markdown("**Доказательная база.**")
                    for ev in r.get("evidence", []):
                        st.markdown(f"- {ev}")

                    st.markdown(
                        f"**Ожидание автора.** {r.get('expected_outcome', '—')}"
                    )
                    if r.get("risk_notes"):
                        st.markdown(f"**Риски.** {r['risk_notes']}")

                    if rv:
                        st.divider()
                        v_color = verdict_color.get(rv["verdict"], "#888")
                        v_label = verdict_label.get(
                            rv["verdict"], rv["verdict"],
                        )
                        c_label = confidence_label.get(
                            rv["confidence"], rv["confidence"],
                        )
                        st.markdown(
                            f"<div style='display:flex;align-items:center;"
                            f"gap:12px;margin-bottom:6px'>"
                            f"<span style='background:{v_color};color:white;"
                            f"padding:4px 10px;border-radius:4px;"
                            f"font-weight:600'>👨‍🔬 PhD-рецензия: "
                            f"{v_label}</span>"
                            f"<span style='color:#666;font-size:0.9em'>"
                            f"уверенность {c_label}</span></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"_{rv['summary']}_")

                        if rv.get("evidence_check"):
                            st.markdown("**Fact-check доказательной базы:**")
                            for ec in rv["evidence_check"]:
                                mark = ec_mark.get(ec["verdict"], "•")
                                st.markdown(
                                    f"- {mark} **{ec['claim']}** — {ec['note']}"
                                )

                        sl, sr = st.columns(2)
                        if rv.get("strengths"):
                            sl.markdown("**Сильные стороны**")
                            for s in rv["strengths"]:
                                sl.markdown(f"- {s}")
                        if rv.get("weaknesses"):
                            sr.markdown("**Слабые стороны**")
                            for w in rv["weaknesses"]:
                                sr.markdown(f"- {w}")

                        if rv.get("suggested_revision"):
                            st.info(
                                f"**Предложение правки:** "
                                f"{rv['suggested_revision']}"
                            )

                    if ml.get("ferroalloy_breakdown"):
                        with st.expander("Расход ферросплавов в рецепте"):
                            st.dataframe(
                                ml["ferroalloy_breakdown"],
                                width="stretch",
                            )

                    st.caption(f"id={r.get('id', '?')}")


# =========================================================================
# Tab 7: Active Learning — следующие эксперименты по cost-weighted EI
# =========================================================================

with tab_al:
    st.header("🔭 Следующие эксперименты — cost-weighted EI")
    st.caption(
        "Стохастический LHS-скан над feasible space модели. Top-K кандидатов "
        "ранжированы по Expected Improvement / cost — предпочитаются дешёвые "
        "эксперименты с высоким ожидаемым приростом свойства."
    )
    _tab_intro(
        purpose=(
            "Sequential planning экспериментов. Latin Hypercube Sampling "
            "случайно покрывает feasible space модели, для каждой точки "
            "считается **Expected Improvement** (Jones et al. 1998) "
            "относительно best-observed-training f*, делится на cost — "
            "получается ranked queue \"какой следующий эксперимент даст "
            "максимальный ожидаемый прирост свойства на каждый €\". "
            "Чисто numerical, без глубокой нейронной сети, дёшево (~150 мс)."
        ),
        audience=[
            "**Planning lead в R&D** — формирование experimental queue "
            "на месяц / квартал",
            "**Materials scientist в академии** — выбор следующей "
            "серии плавок при ограниченном бюджете",
            "**Test laboratory engineer** — приоритизация expensive "
            "testing в production R&D",
            "**Главный инженер R&D / руководитель опытного "
            "производства** на меткомбинате — где каждая плавка "
            "стоит €5-15k и нужно минимизировать число итераций",
        ],
        steps=[
            "Настроить **n_samples** (2000 — стандарт; 5000 для "
            "финального plan; 500 для quick-look)",
            "Установить **top_k** — сколько лучших экспериментов "
            "вернуть (3-10)",
            "Нажать **«🔭 Найти top-K экспериментов»** — расчёт 1-3 "
            "секунды",
            "Каждая карточка: **EI/cost score**, **прогноз свойства "
            "+ CI**, **Δσ vs base, Δcost vs base, OOD-флаг**, "
            "композиция и process per кандидат",
            "Использовать ranked queue для planning последовательности "
            "experimental campaign; **OOD-флаг ⚠️** означает что "
            "композиция вне training distribution — модель там "
            "неуверенна, эксперимент нужен особенно",
        ],
    )

    if not selected_model:
        st.warning("Сначала выберите активную модель в sidebar.")
    elif _class_id != "fatigue_carbon_steel":
        st.warning(
            "Active learner сейчас работает только на классе "
            "`fatigue_carbon_steel` (Agrawal NIMS). Выберите такую модель."
        )
    else:
        from decision_log.logger import query_decisions

        st.markdown(f"**Активная модель:** `{selected_model}`")

        existing_runs = [
            d for d in query_decisions(phase="inverse_design", limit=200)
            if "active_learning" in (d.get("tags") or [])
            and d.get("context", {}).get("model_version") == selected_model
        ]
        st.caption(
            f"Прошлых запусков: **{len(existing_runs)}**"
            + (
                f" · последний {existing_runs[0]['timestamp'][:16]}"
                if existing_runs else ""
            )
        )

        c1, c2, c3 = st.columns(3)
        n_samples = c1.slider(
            "LHS samples", 500, 5000, 2000, step=500,
            help="Число точек для скана. Больше = точнее, но медленнее.",
        )
        top_k = c2.slider(
            "Top-K", 3, 10, 5,
            help="Сколько лучших кандидатов вернуть.",
        )
        seed = c3.number_input(
            "Seed", value=42, step=1, format="%d",
        )

        run_btn = st.button(
            "🔭 Найти top-K экспериментов",
            type="primary",
            help="LHS-скан + EI ранжировка. ~1-3 секунды, $0.",
        )

        if run_btn:
            from dataclasses import asdict as _asdict
            import numpy as _np
            import pandas as _pd

            from app.backend.active_learner import propose_next_experiments
            from app.backend.cost_model import compute_cost, seed_snapshot
            from app.backend.data_curator import load_real_agrawal_fatigue_dataset
            from app.backend.model_trainer import (
                load_model as _load_m, predict_with_uncertainty,
            )
            from decision_log.logger import log_decision

            DESIGN_COMPOSITION = [
                "si_pct", "mn_pct", "ni_pct", "cr_pct", "cu_pct", "mo_pct",
            ]
            DESIGN_PROCESS = [
                "normalizing_temp_c",
                "carburizing_temp_c",
                "carburizing_time_min",
                "tempering_temp_c",
                "tempering_time_min",
                "through_hardening_cooling_rate_c_per_s",
            ]

            with st.spinner("LHS-скан и predict_with_uncertainty…"):
                bundle = _load_m(selected_model)
                meta = bundle["meta"]
                feature_list = meta["feature_list"]
                target = meta["target"]
                training_ranges = meta["training_ranges"]

                df_raw = load_real_agrawal_fatigue_dataset()
                if "sub_class" in df_raw.columns:
                    sub = df_raw[df_raw["sub_class"] == "carbon_low_alloy"]
                    if len(sub) < 50: sub = df_raw
                else:
                    sub = df_raw
                baseline = sub.select_dtypes(include=[_np.number]).median()
                f_star = float(df_raw[target].max())
                snapshot = seed_snapshot()

                X_base = _pd.DataFrame(
                    [[float(baseline[f]) for f in feature_list]],
                    columns=feature_list,
                )
                base_pred = float(
                    predict_with_uncertainty(bundle, X_base).iloc[0]["prediction"]
                )
                base_comp = {
                    k: float(v) for k, v in baseline.items()
                    if k.endswith("_pct")
                }
                base_cost = compute_cost(
                    base_comp, snapshot, mode="full",
                ).total_per_ton

                decision_vars = [
                    v for v in DESIGN_COMPOSITION + DESIGN_PROCESS
                    if v in feature_list and v in training_ranges
                ]
                bounds = {v: tuple(training_ranges[v]) for v in decision_vars}

                def cost_fn(comp):
                    return compute_cost(
                        comp, snapshot, mode="full",
                    ).total_per_ton

                proposals = propose_next_experiments(
                    model_bundle=bundle,
                    baseline_row=baseline,
                    feature_list=feature_list,
                    decision_vars=decision_vars,
                    bounds=bounds,
                    f_star=f_star,
                    cost_fn=cost_fn,
                    baseline_cost=base_cost,
                    baseline_property=base_pred,
                    n_samples=int(n_samples),
                    top_k=int(top_k),
                    seed=int(seed),
                )

                log_decision(
                    phase="inverse_design",
                    decision=f"ActiveLearner: top-{top_k} via UI",
                    reasoning=(
                        f"model={selected_model}, n_samples={n_samples}, "
                        f"f*={f_star:.0f}, top1 score="
                        f"{proposals[0].acquisition_score:.4f}"
                        if proposals else "no proposals"
                    ),
                    context={
                        "model_version": selected_model,
                        "baseline": {
                            "predicted_property": base_pred,
                            "cost_per_ton": float(base_cost),
                        },
                        "f_star": f_star,
                        "n_samples": int(n_samples),
                        "decision_vars": decision_vars,
                        "proposals": [_asdict(p) for p in proposals],
                    },
                    author="ui",
                    tags=["active_learning"],
                )
            st.success(f"Получено {len(proposals)} кандидатов")
            st.rerun()

        if not existing_runs:
            st.info(
                "Запусков ещё нет — нажмите кнопку выше чтобы получить "
                "первый ranked queue экспериментов."
            )
        else:
            run = existing_runs[0]
            ctx_d = run.get("context", {})
            base = ctx_d.get("baseline", {})
            proposals = ctx_d.get("proposals", [])
            f_star_v = ctx_d.get("f_star", 0)

            cb1, cb2, cb3 = st.columns(3)
            cb1.metric("Baseline σ", f"{base.get('predicted_property', 0):.0f} МПа")
            cb2.metric("Baseline cost", f"{base.get('cost_per_ton', 0):.2f} €/т")
            cb3.metric("f* (target)", f"{f_star_v:.0f} МПа")

            for i, p in enumerate(proposals, 1):
                with st.container(border=True):
                    h1, h2 = st.columns([6, 4])
                    h1.markdown(f"### #{i} EI/cost = {p['acquisition_score']:.4f}")
                    ood = p.get("ood_flag", False)
                    h2.markdown(
                        f"<div style='text-align:right'>"
                        f"<span style='background:{'#c0392b' if ood else '#3a9d23'};"
                        f"color:white;padding:4px 10px;border-radius:4px;"
                        f"font-size:0.85em'>"
                        f"OOD: {'⚠️ да' if ood else '✓ нет'}</span></div>",
                        unsafe_allow_html=True,
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(
                        "Прогноз σ, МПа",
                        f"{p['predicted_property']:.0f}",
                        f"{p['delta_vs_baseline_property']:+.0f}",
                        help=f"90% CI [{p['lower_90']:.0f}, {p['upper_90']:.0f}]",
                    )
                    m2.metric(
                        "Cost, €/т",
                        f"{p['cost_per_ton']:.2f}",
                        f"{p['delta_vs_baseline_cost']:+.2f}",
                        delta_color="inverse",
                    )
                    m3.metric("EI, МПа", f"{p['expected_improvement']:.2f}")
                    m4.metric(
                        "CI ширина",
                        f"{p['uncertainty_width']:.0f}",
                        help="upper_90 - lower_90",
                    )

                    cc1, cc2 = st.columns(2)
                    cc1.markdown("**Composition vs baseline**")
                    if p.get("composition"):
                        cc1.dataframe(
                            [
                                {
                                    "элемент": k,
                                    "wt%": round(v, 4),
                                    "Δ vs base": round(
                                        v - float(
                                            existing_runs[0].get("context", {})
                                            .get("baseline", {})
                                            .get("recipe", {})
                                            .get(k, v)
                                            if "recipe" in base else v
                                        ), 4,
                                    ) if "recipe" in base else None,
                                }
                                for k, v in p["composition"].items()
                            ],
                            width="stretch", hide_index=True,
                        )
                    cc2.markdown("**Process vs baseline**")
                    if p.get("process_params"):
                        cc2.dataframe(
                            [
                                {"параметр": k, "значение": round(v, 1)}
                                for k, v in p["process_params"].items()
                            ],
                            width="stretch", hide_index=True,
                        )

                    st.caption(f"id={p.get('id', '?')}")


with tab_history:
    st.header("История решений проекта")
    st.caption("Structured memory — все архитектурные решения с контекстом и reasoning")
    _tab_intro(
        purpose=(
            "Audit trail всех решений принятых системой за всё время "
            "работы. SQLite база `decision_log/decisions.db`. Каждая "
            "запись: фаза, что решено, reasoning, контекст (полный "
            "snapshot входных и выходных данных), теги, автор. "
            "Это **persistent memory проекта** — компенсирует "
            "отсутствие memory у сессий глубокой нейронной сети и обеспечивает "
            "auditability для regulated industries."
        ),
        audience=[
            "**Project manager / R&D lead** — отслеживание "
            "истории решений по заказу или продукту",
            "**QA / compliance officer** — auditability требование "
            "для aerospace / nuclear / автомобильной "
            "сертификации (ISO 9001, IATF 16949, AS9100)",
            "**Любой пользователь** для воспроизведения предыдущих "
            "циклов (replay какой-то AI-генерации, recipe-цикла, "
            "deoxidation расчёта)",
            "**Senior research engineer** — cross-project review "
            "(посмотреть какие решения принимались и почему)",
        ],
        steps=[
            "Фильтровать по **фазе** (data_acquisition / preprocessing "
            "/ training / inverse_design / validation / reporting / "
            "meta) или показать все",
            "Установить **лимит** (по умолчанию 20)",
            "**Развернуть expander** каждой записи — там полный "
            "context (JSON) + reasoning + alternatives + теги + "
            "outcome",
            "Для batch анализа использовать SQL-доступ напрямую: "
            "`sqlite3 decision_log/decisions.db`",
        ],
    )

    from decision_log.logger import query_decisions, summarize_project_history
    
    c1, c2 = st.columns(2)
    phase_filter = c1.selectbox("Фильтр по фазе", 
                                ["Все"] + ["data_acquisition", "preprocessing",
                                          "feature_engineering", "training",
                                          "inverse_design", "validation",
                                          "reporting", "meta"])
    limit = c2.slider("Максимум записей", 5, 100, 20)
    
    phase = None if phase_filter == "Все" else phase_filter
    decisions = query_decisions(phase=phase, limit=limit)
    
    st.metric("Найдено", len(decisions))
    
    if decisions:
        for d in decisions:
            with st.expander(
                f"[{d['phase']}] {d['decision'][:80]} — {d['timestamp'][:10]} ({d.get('author', '?')})"
            ):
                st.markdown(f"**Reasoning:** {d['reasoning']}")
                if d.get("alternatives_considered"):
                    st.markdown(f"**Альтернативы:** {', '.join(d['alternatives_considered'])}")
                if d.get("context"):
                    st.markdown("**Context:**")
                    st.json(d["context"])
                if d.get("tags"):
                    st.markdown(f"**Теги:** {', '.join(d['tags'])}")
                if d.get("outcome"):
                    st.success(f"**Outcome:** {d['outcome']}")


# Footer
st.divider()
st.caption("Steel AI MVP · HSLA design · Synthetic dataset · Demo only")
