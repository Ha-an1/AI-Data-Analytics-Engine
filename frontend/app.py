import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import json

# Ensure backend imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import PipelineOrchestrator
from backend.database.db_manager import SessionLocal, init_db
from backend.database.models import DatasetSummary, PipelineMetadata, ModelMetrics, FeatureImportance
from backend.planner.gemini_planner import GeminiPlanner

st.set_page_config(page_title="AI Pipeline System", layout="wide", page_icon="🧠")
init_db()

# ── Session State ──
if "run_results" not in st.session_state:
    st.session_state.run_results = None
if "llm_analysis" not in st.session_state:
    st.session_state.llm_analysis = None

# ── Sidebar ──
st.sidebar.title("🧠 AI Analytics Engine")
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🔌 Power BI Integration
Results are stored in `analytics_engine.db`.  
Connect Power BI via ODBC to visualize:
- `model_metrics`
- `feature_importance`
- `dataset_summary`
- `pipeline_metadata`
""")

# ── Main Tabs ──
tab_run, tab_results, tab_inference = st.tabs(["🚀 Pipeline Runner", "📊 Results Dashboard", "🔮 Inference"])

# ═══════════════════════════════════════════════
# TAB 1: PIPELINE RUNNER
# ═══════════════════════════════════════════════
with tab_run:
    st.header("🚀 Autonomous Pipeline Runner")
    st.markdown("Upload a dataset, describe your goal, and the AI will automatically build, train, and evaluate ML models.")

    col_upload, col_config = st.columns(2)

    with col_upload:
        st.subheader("📂 Data Ingestion")
        uploaded_file = st.file_uploader("Upload CSV or JSON", type=['csv', 'json'], key="pipeline_upload")

    with col_config:
        st.subheader("🎯 What do you want to achieve?")
        problem_statement = st.text_area(
            "Describe your problem / objective",
            placeholder="e.g. 'I want to predict which customers will churn based on their usage patterns' or 'Detect fraudulent transactions in this credit card data'",
            height=120,
            key="problem_stmt"
        )

    st.markdown("---")

    # ── LLM Auto-Detection Section ──
    use_llm = st.checkbox("🤖 Use Gemini AI to auto-detect task type & target column", value=True)

    if not use_llm:
        st.info("Manual mode: specify the task and target column below.")
        manual_col1, manual_col2 = st.columns(2)
        with manual_col1:
            target_column_manual = st.text_input("Target Column", key="manual_target")
        with manual_col2:
            task_type_manual = st.selectbox("Task Objective",
                                            ["classification", "regression", "clustering", "anomaly_detection",
                                             "time_series_forecasting", "nlp_classification"], key="manual_task")

    # ── Show preview of uploaded data ──
    if uploaded_file is not None:
        try:
            preview_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_json(uploaded_file)
            uploaded_file.seek(0)  # Reset file pointer after preview read
            with st.expander("📋 Dataset Preview (first 5 rows)", expanded=False):
                st.dataframe(preview_df.head(), width='stretch')
                st.caption(f"Shape: {preview_df.shape[0]} rows × {preview_df.shape[1]} columns | Columns: {', '.join(preview_df.columns.tolist())}")
        except Exception:
            preview_df = None

    st.markdown("---")

    if st.button("⚡ Execute Autonomous Pipeline", width='stretch', type="primary"):
        if uploaded_file is None:
            st.error("Please upload a dataset first.")
        else:
            os.makedirs("data", exist_ok=True)
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            progress = st.progress(0)
            status = st.empty()

            # Determine task and target
            detected_task = None
            detected_target = None

            if use_llm:
                status.info("🤖 Asking Gemini to analyze your dataset and objective...")
                progress.progress(10)

                try:
                    temp_df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_json(file_path)
                    sample_json = temp_df.head(5).to_json(orient='records', indent=2)
                    columns = temp_df.columns.tolist()

                    gemini = GeminiPlanner()
                    llm_result = gemini.analyze_with_llm(sample_json, columns, problem_statement or "Analyze this dataset and suggest the best ML task.")

                    if llm_result:
                        detected_task = llm_result.get("task")
                        detected_target = llm_result.get("target")
                        reasoning = llm_result.get("reasoning", "")
                        detected_drops = llm_result.get("drop_columns", [])
                        drop_reasons = llm_result.get("drop_reasons", {})

                        st.session_state.llm_analysis = llm_result

                        # Extract metric config
                        primary_metric = llm_result.get("primary_metric")
                        secondary_metrics = llm_result.get("secondary_metrics", [])
                        optimization_goal = llm_result.get("optimization_goal")

                        st.info(f"""
**🤖 Gemini Analysis:**
- **Task:** `{detected_task}`
- **Target Column:** `{detected_target}`
- **Primary Metric:** `{primary_metric}` ({optimization_goal})
- **Secondary Metrics:** {', '.join(f'`{m}`' for m in secondary_metrics) if secondary_metrics else 'none'}
- **Reasoning:** {reasoning}
                        """)

                        # Display data leakage warnings
                        if detected_drops:
                            drop_text = "\n".join([f"- **`{col}`**: {drop_reasons.get(col, 'potential leakage')}" for col in detected_drops])
                            st.warning(f"""
⚠️ **Data Leakage Detected!** The following columns will be **dropped** before training:

{drop_text}

These columns would give the model unfair access to the answer, producing misleadingly high accuracy.
                            """)
                    else:
                        st.warning("Gemini could not analyze the dataset. Falling back to manual defaults.")
                except Exception as e:
                    st.warning(f"Gemini analysis failed: {e}. Falling back to manual mode.")

            if not use_llm or detected_task is None:
                detected_task = task_type_manual if not use_llm else "classification"
                detected_target = target_column_manual if not use_llm else None
                detected_drops = []
                primary_metric = None
                secondary_metrics = []
                optimization_goal = None

            status.info("🔄 Booting orchestrator...")
            progress.progress(25)

            orchestrator = PipelineOrchestrator()
            user_goal = {
                "target": detected_target if detected_target else None,
                "task": detected_task,
                "drop_columns": detected_drops if detected_drops else [],
                "primary_metric": primary_metric,
                "secondary_metrics": secondary_metrics,
                "optimization_goal": optimization_goal
            }

            try:
                status.info(f"📊 Running pipeline: task=`{detected_task}`, target=`{detected_target}`...")
                progress.progress(40)

                results = orchestrator.run_pipeline(file_path, user_goal)

                progress.progress(100)
                status.success("✅ Pipeline executed successfully! Switch to the **Results Dashboard** or **Inference** tab.")
                st.session_state.run_results = results

                # Quick summary
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Best Model", results.get("best_model", "N/A"))
                col_m2.metric("Models Trained", len(results.get("results", {})))
                col_m3.metric("Task Type", results.get("task_type", "N/A"))

            except Exception as e:
                progress.progress(100)
                status.error(f"❌ Pipeline Failed: {e}")

# ═══════════════════════════════════════════════
# TAB 2: RESULTS DASHBOARD
# ═══════════════════════════════════════════════
with tab_results:
    st.header("📊 Results Dashboard")

    with SessionLocal() as db:
        all_pipelines = db.query(PipelineMetadata).order_by(PipelineMetadata.id.desc()).all()

    if not all_pipelines:
        st.info("No pipeline runs found yet. Go to the **Pipeline Runner** tab to execute one.")
    else:
        pipeline_options = {f"Run #{p.id} — {p.task_objective} ({p.timestamp})": p.id for p in all_pipelines}
        selected_label = st.selectbox("Select Pipeline Run", list(pipeline_options.keys()))
        selected_pipeline_id = pipeline_options[selected_label]

        with SessionLocal() as db:
            metrics_rows = db.query(ModelMetrics).filter(ModelMetrics.pipeline_id == selected_pipeline_id).all()
            pipeline_meta = db.query(PipelineMetadata).filter(PipelineMetadata.id == selected_pipeline_id).first()
            dataset_meta = db.query(DatasetSummary).filter(DatasetSummary.id == pipeline_meta.dataset_id).first() if pipeline_meta else None

        if not metrics_rows:
            st.warning("No model metrics found for this pipeline run.")
        else:
            # ── Dataset Overview ──
            if dataset_meta and dataset_meta.metadata_json:
                st.subheader("📋 Dataset Overview")
                meta = dataset_meta.metadata_json if isinstance(dataset_meta.metadata_json, dict) else json.loads(dataset_meta.metadata_json)
                overview_cols = st.columns(4)
                overview_cols[0].metric("Rows", f"{meta.get('rows', 'N/A'):,}")
                overview_cols[1].metric("Columns", meta.get('columns', 'N/A'))
                overview_cols[2].metric("Numerical Features", meta.get('numerical_features', 0))
                overview_cols[3].metric("Categorical Features", meta.get('categorical_features', 0))

                if meta.get('missing_values'):
                    with st.expander("🔍 Missing Values Detail"):
                        missing_df = pd.DataFrame(list(meta['missing_values'].items()), columns=["Column", "Missing %"])
                        missing_df["Missing %"] = (missing_df["Missing %"] * 100).round(2)
                        st.dataframe(missing_df, width='stretch')

            st.markdown("---")

            # ── Model Performance Table ──
            st.subheader("🏆 Model Comparison")
            task = pipeline_meta.task_objective if pipeline_meta else "classification"

            metrics_data = []
            for m in metrics_rows:
                row = {"Model": m.model_name}
                if task in ["classification", "nlp_classification"]:
                    row["Accuracy"] = round(m.accuracy, 4) if m.accuracy is not None else None
                    row["Precision"] = round(m.precision, 4) if m.precision is not None else None
                    row["Recall"] = round(m.recall, 4) if m.recall is not None else None
                    row["F1 Score"] = round(m.f1_score, 4) if m.f1_score is not None else None
                elif task in ["regression", "time_series_forecasting"]:
                    row["RMSE"] = round(m.rmse, 4) if m.rmse is not None else None
                    row["MAE"] = round(m.mae, 4) if m.mae is not None else None
                    row["R² Score"] = round(m.r2_score, 4) if m.r2_score is not None else None
                elif task == "clustering":
                    row["Silhouette"] = round(m.silhouette_score, 4) if m.silhouette_score is not None else None
                row["Train Time (s)"] = round(m.training_time_seconds, 2) if m.training_time_seconds is not None else None
                row["Best"] = "⭐" if m.is_best_model else ""
                metrics_data.append(row)

            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, width='stretch', hide_index=True)

            # ── Charts ──
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.subheader("📈 Performance Chart")
                chart_df = metrics_df.set_index("Model")
                if task in ["classification", "nlp_classification"]:
                    numeric_cols = [c for c in ["Accuracy", "F1 Score"] if c in chart_df.columns]
                    if numeric_cols:
                        st.bar_chart(chart_df[numeric_cols].dropna())
                elif task in ["regression", "time_series_forecasting"]:
                    numeric_cols = [c for c in ["RMSE", "MAE"] if c in chart_df.columns]
                    if numeric_cols:
                        st.bar_chart(chart_df[numeric_cols].dropna())

            with chart_cols[1]:
                st.subheader("⏱️ Training Duration")
                if "Train Time (s)" in chart_df.columns:
                    st.bar_chart(chart_df[["Train Time (s)"]].dropna())

            st.markdown("---")

            # ── Feature Importance ──
            best_metric = next((m for m in metrics_rows if m.is_best_model), metrics_rows[0])
            st.subheader(f"🔍 Feature Importance — {best_metric.model_name}")

            with SessionLocal() as db:
                features = db.query(FeatureImportance).filter(
                    FeatureImportance.metric_id == best_metric.id
                ).order_by(FeatureImportance.importance_score.desc()).limit(15).all()

            if features:
                feat_df = pd.DataFrame([{"Feature": f.feature_name, "Importance": round(f.importance_score, 5)} for f in features])
                feat_df = feat_df.set_index("Feature")
                st.bar_chart(feat_df)
            else:
                st.info("No feature importance data available for this model.")

# ═══════════════════════════════════════════════
# TAB 3: INFERENCE & EVALUATION
# ═══════════════════════════════════════════════
with tab_inference:
    st.header("🔮 Inference & Evaluation")

    model_dir = "trained_models"
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        st.info("No trained models found. Please run a pipeline first in the **Pipeline Runner** tab.")
    else:
        available_models = [f.replace(".joblib", "") for f in os.listdir(model_dir) if f.endswith(".joblib")]
        selected_model = st.selectbox("Select Trained Model", available_models, key="inf_model")

        st.markdown("---")
        inference_mode = st.radio(
            "Inference Mode",
            ["🧪 Eval Inference", "📝 Manual Input", "📂 Batch Upload", "📜 Experiment History"],
            horizontal=True
        )

        # ══════════════════════════════════════════
        # EVAL INFERENCE + AUTO-RETRAIN
        # ══════════════════════════════════════════
        if inference_mode == "🧪 Eval Inference":
            st.subheader("🧪 Evaluate on Held-Out Eval Set")
            st.markdown("Run the selected model on the **15% eval set** held out during training. "
                         "If metrics are below threshold, trigger the intelligent retraining controller.")

            eval_ready = os.path.exists("eval_data/X_eval.csv") and os.path.exists("eval_data/y_eval.csv")

            if not eval_ready:
                st.warning("No eval set found. Run a pipeline first.")
            else:
                run_meta = {}
                if os.path.exists("eval_data/run_meta.json"):
                    with open("eval_data/run_meta.json", "r") as f:
                        run_meta = json.load(f)

                eval_X = pd.read_csv("eval_data/X_eval.csv")
                eval_y = pd.read_csv("eval_data/y_eval.csv")
                task_type_eval = run_meta.get("task_type", "classification")

                st.caption(f"Eval set: **{len(eval_X)} rows** | Task: `{task_type_eval}`")

                # Threshold config
                col_th1, col_th2 = st.columns(2)
                with col_th1:
                    if task_type_eval in ["classification", "nlp_classification"]:
                        threshold = st.number_input("Minimum F1 Score", value=0.75, step=0.05, key="eval_threshold")
                    elif task_type_eval in ["regression", "time_series_forecasting"]:
                        threshold = st.number_input("Minimum R² Score", value=0.70, step=0.05, key="eval_threshold")
                    else:
                        threshold = st.number_input("Threshold", value=0.50, step=0.05, key="eval_threshold")
                with col_th2:
                    auto_retrain = st.checkbox(
                        "🔄 Auto-retrain if below threshold",
                        value=True,
                        help="If enabled, the system will automatically run 3 retraining strategies when the model fails evaluation."
                    )

                if st.button("🧪 Run Eval Inference", key="eval_btn", type="primary"):
                    try:
                        eval_result = PipelineOrchestrator.run_eval_inference(selected_model)
                        metrics = eval_result["metrics"]
                        eval_task = eval_result["task_type"]

                        st.write("### 📊 Eval Metrics")
                        metric_cols = st.columns(5)
                        if eval_task in ["classification", "nlp_classification"]:
                            metric_cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                            metric_cols[1].metric("Precision", f"{metrics.get('precision', 0):.4f}")
                            metric_cols[2].metric("Recall", f"{metrics.get('recall', 0):.4f}")
                            metric_cols[3].metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
                            metric_cols[4].metric("ROC AUC", f"{metrics.get('roc_auc', 'N/A')}")

                            primary_score = metrics.get("f1_score", 0)
                            passed = primary_score >= threshold
                        elif eval_task in ["regression", "time_series_forecasting"]:
                            metric_cols[0].metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                            metric_cols[1].metric("MAE", f"{metrics.get('mae', 0):.4f}")
                            metric_cols[2].metric("R²", f"{metrics.get('r2_score', 0):.4f}")

                            primary_score = metrics.get("r2_score", 0)
                            passed = primary_score >= threshold
                        else:
                            primary_score = 0
                            passed = True

                        # Predicted vs Actual
                        with st.expander("🔍 Predicted vs Actual (first 20 rows)"):
                            comparison = pd.DataFrame({
                                "Actual": eval_result["y_eval"][:20],
                                "Predicted": eval_result["y_pred"][:20],
                                "Match": ["✅" if str(a) == str(p) else "❌"
                                          for a, p in zip(eval_result["y_eval"][:20], eval_result["y_pred"][:20])]
                            })
                            st.dataframe(comparison, width='stretch', hide_index=True)

                        if passed:
                            st.success(f"✅ **PASSED!** Model meets the threshold "
                                       f"(score: `{primary_score:.4f}`, threshold: `{threshold}`).\n\n"
                                       f"Model is **production-ready**. ✨")
                        else:
                            st.error(f"❌ **FAILED!** Score: `{primary_score:.4f}` < threshold: `{threshold}`")

                            if auto_retrain:
                                st.markdown("---")
                                st.write("### 🔄 Intelligent Retraining Controller")
                                st.markdown("Running 3-strategy improvement loop...")

                                progress_area = st.empty()
                                attempt_container = st.container()

                                def progress_cb(attempt, strategy, msg):
                                    progress_area.info(f"**Attempt {attempt}/3**: {strategy} — {msg}")

                                with st.spinner("Retraining in progress... This may take a few minutes."):
                                    orchestrator = PipelineOrchestrator()
                                    retrain_result = orchestrator.trigger_retraining(
                                        selected_model,
                                        threshold_override=threshold,
                                        progress_callback=progress_cb
                                    )

                                progress_area.empty()

                                # Show attempts log
                                if retrain_result["attempts"]:
                                    st.write("#### 📋 Retraining Attempts")
                                    attempts_df = pd.DataFrame(retrain_result["attempts"])
                                    attempts_df["improved"] = attempts_df["improved"].map({True: "✅", False: "❌"})
                                    st.dataframe(attempts_df, width='stretch', hide_index=True)

                                if retrain_result["improved"]:
                                    st.success(f"""
✅ **Model Improved!**
- **New model:** `{retrain_result['model_name']}`
- **Score:** `{retrain_result.get('baseline_score', 0):.4f}` → `{retrain_result['final_score']:.4f}`
- Model has been saved to disk. Run eval inference again to verify.
                                    """)
                                else:
                                    st.warning(f"""
⚠️ **Retraining did not produce a better model.**
- Best score achieved: `{retrain_result['final_score']:.4f}`
- Original model has been retained.
- Consider: more data, different features, or domain-specific preprocessing.
                                    """)
                            else:
                                st.info("Auto-retrain is disabled. Enable it above to trigger automatic improvement.")

                    except Exception as e:
                        st.error(f"Eval inference failed: {e}")

        # ══════════════════════════════════════════
        # MANUAL INPUT
        # ══════════════════════════════════════════
        elif inference_mode == "📝 Manual Input":
            st.subheader("Enter Feature Values")

            if st.session_state.run_results and st.session_state.run_results.get("feature_columns"):
                feature_cols = st.session_state.run_results["feature_columns"]
                st.caption(f"**Expected features ({len(feature_cols)}):** {', '.join(feature_cols)}")
            else:
                feature_cols = None
                if os.path.exists("eval_data/run_meta.json"):
                    with open("eval_data/run_meta.json", "r") as f:
                        rm = json.load(f)
                    feature_cols = rm.get("feature_columns")
                    if feature_cols:
                        st.caption(f"**Expected features ({len(feature_cols)}):** {', '.join(feature_cols)}")
                if not feature_cols:
                    st.caption("Feature names not available. Enter values in the same order as training data columns.")

            feature_input = st.text_area("Feature values (comma-separated)", placeholder="e.g. 25, 50000, 1, 0, 3.5, ...")

            if st.button("🔮 Predict", key="manual_predict", type="primary"):
                if not feature_input.strip():
                    st.error("Please enter feature values.")
                else:
                    try:
                        pipeline = joblib.load(os.path.join(model_dir, f"{selected_model}.joblib"))
                        values = []
                        for v in feature_input.split(","):
                            v = v.strip()
                            try:
                                values.append(float(v))
                            except ValueError:
                                values.append(v)

                        input_array = np.array([values])
                        if feature_cols and len(feature_cols) == len(values):
                            input_df = pd.DataFrame(input_array, columns=feature_cols)
                        else:
                            input_df = pd.DataFrame(input_array)

                        prediction = pipeline.predict(input_df)
                        st.success(f"### Prediction: `{prediction[0]}`")

                        if hasattr(pipeline, "predict_proba"):
                            try:
                                proba = pipeline.predict_proba(input_df)
                                st.write("**Prediction Probabilities:**")
                                proba_df = pd.DataFrame(proba, columns=[f"Class {i}" for i in range(proba.shape[1])])
                                st.dataframe(proba_df, width='stretch')
                            except Exception:
                                pass
                    except Exception as e:
                        st.error(f"Inference failed: {e}")

        # ══════════════════════════════════════════
        # BATCH UPLOAD
        # ══════════════════════════════════════════
        elif inference_mode == "📂 Batch Upload":
            st.subheader("Upload CSV for Batch Predictions")
            st.caption("CSV should have the same feature columns as training data (without target).")
            batch_file = st.file_uploader("Upload CSV", type=['csv'], key="batch_upload")

            if batch_file and st.button("🔮 Run Batch Predictions", key="batch_predict", type="primary"):
                try:
                    pipeline = joblib.load(os.path.join(model_dir, f"{selected_model}.joblib"))
                    batch_df = pd.read_csv(batch_file)
                    st.write(f"**Input Shape:** {batch_df.shape[0]} rows × {batch_df.shape[1]} columns")

                    predictions = pipeline.predict(batch_df)
                    batch_df["Prediction"] = predictions

                    if hasattr(pipeline, "predict_proba"):
                        try:
                            probas = pipeline.predict_proba(batch_df.drop(columns=["Prediction"]))
                            batch_df["Confidence"] = np.max(probas, axis=1).round(4)
                        except Exception:
                            pass

                    st.success(f"✅ Generated {len(predictions)} predictions!")
                    st.dataframe(batch_df, width='stretch')

                    csv_output = batch_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Predictions CSV",
                        data=csv_output,
                        file_name="predictions_output.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Batch inference failed: {e}")

        # ══════════════════════════════════════════
        # EXPERIMENT HISTORY
        # ══════════════════════════════════════════
        elif inference_mode == "📜 Experiment History":
            st.subheader("📜 Model Experiment History")
            st.markdown("All retraining attempts are logged to the `model_runs` database table.")

            try:
                from backend.database.models import ModelRuns
                with SessionLocal() as db:
                    runs = db.query(ModelRuns).order_by(ModelRuns.id.desc()).limit(50).all()

                if not runs:
                    st.info("No retraining experiments found yet.")
                else:
                    runs_data = []
                    for r in runs:
                        runs_data.append({
                            "ID": r.id,
                            "Timestamp": str(r.timestamp),
                            "Model": r.model_name,
                            "Strategy": r.strategy,
                            "Attempt": r.attempt,
                            "Val Score": round(r.validation_score, 4) if r.validation_score else None,
                            "Eval Score": round(r.evaluation_score, 4) if r.evaluation_score else None,
                            "Duration (s)": round(r.training_duration, 2) if r.training_duration else None,
                            "Improved": "✅" if r.improved else "❌",
                        })
                    st.dataframe(pd.DataFrame(runs_data), width='stretch', hide_index=True)
            except Exception as e:
                st.error(f"Could not load experiment history: {e}")


