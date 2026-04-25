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
    "🤖 LLM-Critic",
    "✓ активен" if _llm_ok else "— нет ключа",
)


# =========================================================================
# Main tabs
# =========================================================================

tab_design, tab_train, tab_predict, tab_deox, tab_hyp, tab_history = st.tabs([
    "🎯 Дизайн сплава",
    "🤖 Обучение модели",
    "📊 Прогноз",
    "🔥 Раскисление",
    "💡 Гипотезы",
    "📚 История",
])


# =========================================================================
# Tab 1: Inverse design
# =========================================================================

with tab_design:
    st.header("Поиск состава под целевые свойства")
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

    if _design_class_id == "en10083_qt":
        st.info(
            "ℹ️ Inverse design пока работает только для **Pipe HSLA**. "
            "Для класса EN 10083-2 Q&T используйте вкладку «📊 Прогноз». "
            "Поддержка inverse design для Q&T запланирована на v2."
        )
        st.stop()

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
            use_container_width=True,
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
            st.altair_chart(chart, use_container_width=True)

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
                        use_container_width=True,
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
                        st.dataframe(df_bd, use_container_width=True, hide_index=True)
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

            # LLM-Critic (Claude Sonnet 4.6) — only runs with ANTHROPIC_API_KEY
            from app.backend.critic_llm import make_llm_critic
            from dataclasses import asdict
            _llm = make_llm_critic()
            if _llm is not None:
                with st.spinner("🤖 LLM-Critic проверяет..."):
                    llm_obs = _llm.review_training(critic_ctx)
                    st.session_state["llm_observations"] = [
                        asdict(o) for o in llm_obs
                    ]

            llm_obs_rendered = st.session_state.get("llm_observations", [])
            if llm_obs_rendered:
                st.subheader("🤖 LLM-Critic (Claude Sonnet 4.6)")
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
                st.caption("🤖 LLM-Critic: проблем не обнаружено")

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


# =========================================================================
# Tab: Al Deoxidation Calculator (on-line LF advisory)
# =========================================================================

with tab_deox:
    st.header("🔥 Раскисление жидкой стали алюминием")
    st.caption(
        "Physics-based advisory на базе 3 термодинамических моделей. "
        "Без ML. Расчёт на каждую плавку."
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

    sub_fwd, sub_inv, sub_cmp = st.tabs([
        "Сколько Al нужно", "Качество Al по факту", "⚖️ Сравнить модели",
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
            st.dataframe(df_cmp, hide_index=True, use_container_width=True)

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
            st.altair_chart(chart, use_container_width=True)


# =========================================================================
# Tab 4: Decision Log
# =========================================================================

# =========================================================================
# Tab 5: Hypotheses (LLM-generated, A2 from AI integration roadmap)
# =========================================================================

with tab_hyp:
    st.header("💡 Гипотезы от ИИ-наблюдателя")
    st.caption(
        "LLM просматривает обученную модель и предлагает testable гипотезы "
        "с оценкой экономического эффекта vs классическая практика."
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
            if d.get("author") == "hypothesis_generator"
            and d.get("context", {}).get("model_version") == selected_model
        ]
        st.caption(
            f"Прошлых запусков на этой модели: **{len(existing_runs)}**"
            + (
                f" · последний {existing_runs[0]['timestamp'][:16]}"
                if existing_runs else ""
            )
        )

        run_btn = st.button(
            "🔮 Сгенерировать гипотезы",
            type="primary",
            help=(
                "Один запуск ~100 секунд, ~$0.08. "
                "Sonnet 4.6 анализирует артефакт модели."
            ),
        )

        if run_btn:
            from scripts.generate_hypotheses_for_model import build_context
            from app.backend.hypothesis_generator import make_hypothesis_generator

            gen = make_hypothesis_generator()
            if gen is None:
                st.error("HypothesisGenerator unavailable (anthropic SDK?)")
            else:
                with st.spinner(
                    "Sonnet анализирует артефакт модели… ~1-2 минуты"
                ):
                    ctx = build_context(selected_model)
                    new_hypotheses = gen.generate(ctx)
                if not new_hypotheses:
                    st.error(
                        "Получено 0 гипотез. Проверьте логи / model artifact."
                    )
                else:
                    st.success(f"Получено {len(new_hypotheses)} гипотез")
                    st.rerun()

        display_runs = existing_runs[:1]
        if not display_runs:
            st.info(
                "Запусков ещё нет — нажмите кнопку выше чтобы получить "
                "первые гипотезы."
            )
        else:
            run = display_runs[0]
            usage = run.get("context", {}).get("usage", {})
            cols = st.columns(4)
            cols[0].metric("Гипотез", len(run["context"].get("hypotheses", [])))
            cols[1].metric("Время отклика, с", f"{usage.get('latency_s', 0):.1f}")
            cols[2].metric(
                "Токены на выходе",
                int(usage.get("output_tokens", 0)),
            )
            cache_hit = usage.get("cache_read", 0)
            cols[3].metric(
                "Кэш попадание",
                "✓" if cache_hit > 100 else "—",
                help=f"cache_read={cache_hit} токенов",
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

            for i, h in enumerate(run["context"]["hypotheses"], start=1):
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

                    st.caption(
                        f"id={h.get('id', '?')} · теги: "
                        f"{', '.join(h.get('tags', []))}"
                    )


with tab_history:
    st.header("История решений проекта")
    st.caption("Structured memory — все архитектурные решения с контекстом и reasoning")
    
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
