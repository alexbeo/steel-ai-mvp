"""
Steel AI MVP — Streamlit UI для демо.

Запуск:
    PYTHONPATH=. streamlit run app/frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Steel AI — HSLA Design", layout="wide", page_icon="⚙️")


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


# =========================================================================
# Main tabs
# =========================================================================

tab_design, tab_train, tab_predict, tab_history = st.tabs([
    "🎯 Дизайн сплава", "🤖 Обучение модели", "📊 Прогноз", "📚 История"
])


# =========================================================================
# Tab 1: Inverse design
# =========================================================================

with tab_design:
    st.header("Поиск состава под целевые свойства")
    st.caption("Задайте ТЗ — получите Pareto-оптимальные кандидаты с прогнозом и валидацией")
    
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
    
    if st.button("🚀 Запустить дизайн", type="primary", disabled=not selected_model):
        if not selected_model:
            st.error("Сначала обучите модель")
        else:
            with st.spinner("NSGA-II оптимизация..."):
                from app.backend.inverse_designer import run_inverse_design
                from app.backend.validator import validate_batch
                
                result = run_inverse_design(
                    model_version=selected_model,
                    targets={"yield_strength_mpa": {"min": yt_min, "max": yt_max}},
                    hard_constraints={"cev_iiw": {"max": cev_max}, "pcm": {"max": pcm_max}},
                    population_size=pop_size,
                    n_generations=n_gen,
                )
                
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
                    st.markdown("**Стоимость**")
                    st.write(f"≈ {c.get('objectives', {}).get('alloying_cost', 0):.1f} €/т")
                
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
    
    c1, c2 = st.columns(2)
    target_col = c1.selectbox(
        "Target property",
        ["yield_strength_mpa", "tensile_strength_mpa", "elongation_pct", "kcv_neg60_j_cm2"],
    )
    n_trials = c2.slider("Optuna trials (чем больше, тем лучше, но медленнее)", 10, 150, 40)
    
    st.info("ℹ️ Обучение займёт 1-5 минут в зависимости от количества trials.")
    
    if st.button("🤖 Обучить модель", type="primary"):
        with st.spinner("Generating dataset & training..."):
            from app.backend.data_curator import save_sample_dataset, clean_dataset
            from app.backend.feature_eng import compute_hsla_features, PIPE_HSLA_FEATURE_SET
            from app.backend.model_trainer import train_model
            
            data_path = PROJECT_ROOT / "data" / "hsla_synthetic.parquet"
            if not data_path.exists():
                save_sample_dataset()
            
            df = pd.read_parquet(data_path)
            df_clean, _ = clean_dataset(df)
            df_feat = compute_hsla_features(df_clean)
            feat = [f for f in PIPE_HSLA_FEATURE_SET if f in df_feat.columns]
            
            progress = st.progress(0, text="Запускаю обучение...")
            trained = train_model(df_feat, target_col, feat, n_optuna_trials=n_trials)
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
                "coverage_90_ci": trained.metrics.coverage_90_ci,
                "prediction_has_ci": True,
                "has_time_column": True,
                "has_groups": True,
                "split_strategy": "time_based",
                "cv_strategy": "group_kfold",
                "feature_importance": trained.feature_importance,
                "steel_class": "pipe_hsla",
                "ood_detector_configured": True,
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
        c1, c2, c3, c4 = st.columns(4)
        c_pct = c1.number_input("C %", 0.02, 0.2, 0.08, step=0.005, format="%.4f")
        si_pct = c2.number_input("Si %", 0.0, 0.8, 0.3, step=0.01)
        mn_pct = c3.number_input("Mn %", 0.5, 2.0, 1.5, step=0.05)
        p_pct = c4.number_input("P %", 0.0, 0.03, 0.012, step=0.001, format="%.4f")
        
        c5, c6, c7, c8 = st.columns(4)
        s_pct = c5.number_input("S %", 0.0, 0.02, 0.003, step=0.001, format="%.4f")
        cr_pct = c6.number_input("Cr %", 0.0, 0.5, 0.1, step=0.01)
        ni_pct = c7.number_input("Ni %", 0.0, 0.5, 0.15, step=0.01)
        mo_pct = c8.number_input("Mo %", 0.0, 0.2, 0.02, step=0.01)
        
        c9, c10, c11, c12 = st.columns(4)
        cu_pct = c9.number_input("Cu %", 0.0, 0.5, 0.15, step=0.01)
        al_pct = c10.number_input("Al %", 0.0, 0.1, 0.035, step=0.005, format="%.4f")
        v_pct = c11.number_input("V %", 0.0, 0.15, 0.04, step=0.005, format="%.4f")
        nb_pct = c12.number_input("Nb %", 0.0, 0.1, 0.03, step=0.005, format="%.4f")
        
        c13, c14, c15, c16 = st.columns(4)
        ti_pct = c13.number_input("Ti %", 0.0, 0.05, 0.015, step=0.005, format="%.4f")
        n_ppm = c14.number_input("N ppm", 20.0, 100.0, 55.0, step=5.0)
        rolling_t = c15.number_input("T прокатки °C", 700.0, 900.0, 810.0, step=5.0)
        cooling = c16.number_input("Скорость охл. °C/с", 5.0, 40.0, 18.0, step=1.0)
        
        if st.button("🔮 Предсказать", type="primary"):
            from app.backend.model_trainer import load_model, predict_with_uncertainty
            from app.backend.feature_eng import compute_hsla_features
            
            row = {
                "c_pct": c_pct, "si_pct": si_pct, "mn_pct": mn_pct, "p_pct": p_pct,
                "s_pct": s_pct, "cr_pct": cr_pct, "ni_pct": ni_pct, "mo_pct": mo_pct,
                "cu_pct": cu_pct, "al_pct": al_pct, "v_pct": v_pct, "nb_pct": nb_pct,
                "ti_pct": ti_pct, "n_ppm": n_ppm,
                "rolling_finish_temp": rolling_t, "cooling_rate_c_per_s": cooling,
            }
            df_input = pd.DataFrame([row])
            df_feat = compute_hsla_features(df_input)
            
            bundle = load_model(selected_model)
            pred = predict_with_uncertainty(bundle, df_feat)
            
            mean = float(pred["prediction"].iloc[0])
            lo = float(pred["lower_90"].iloc[0])
            hi = float(pred["upper_90"].iloc[0])
            ood = bool(pred["ood_flag"].iloc[0])
            
            st.subheader(f"{bundle['meta']['target']}: **{mean:.1f}** ± {(hi - lo) / 2:.1f}")
            st.caption(f"90% ДИ: [{lo:.1f}, {hi:.1f}]")
            
            if ood:
                st.error("⚠️ Состав вне training distribution — прогноз ненадёжен!")
            
            st.markdown("**Производные параметры:**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CEV(IIW)", f"{df_feat['cev_iiw'].iloc[0]:.3f}")
            c2.metric("Pcm", f"{df_feat['pcm'].iloc[0]:.3f}")
            c3.metric("CEN", f"{df_feat['cen'].iloc[0]:.3f}")
            c4.metric("Микролегирование", f"{df_feat['microalloying_sum'].iloc[0]:.4f}")


# =========================================================================
# Tab 4: Decision Log
# =========================================================================

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
