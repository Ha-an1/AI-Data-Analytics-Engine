import pandas as pd
import numpy as np
import time
import json
import os
import joblib
import logging
from sklearn.model_selection import train_test_split

from backend.database.models import DatasetSummary, PipelineMetadata, ModelMetrics, FeatureImportance, Predictions
from backend.database.db_manager import SessionLocal, init_db
from backend.analyzer.dataset_analyzer import DatasetAnalyzer
from backend.planner.pipeline_planner import PipelinePlanner
from backend.pipeline.pipeline_builder import PipelineBuilder
from backend.evaluation.evaluator import PipelineEvaluator
from backend.explainability.explainer import ModelExplainer
from backend.features.time_series_features import TimeSeriesFeatureGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    def __init__(self):
        init_db()
        self.analyzer = DatasetAnalyzer()
        self.planner = PipelinePlanner()
        self.builder = PipelineBuilder()
        self.evaluator = PipelineEvaluator()
        self.explainer = ModelExplainer()
        
    def _resolve_target_column(self, df, target_name):
        """Case-insensitive target column resolution."""
        if not target_name:
            return None
        if target_name in df.columns:
            return target_name
        col_map = {c.lower().strip(): c for c in df.columns}
        resolved = col_map.get(target_name.lower().strip())
        if resolved:
            logger.info(f"Resolved target column '{target_name}' -> '{resolved}'")
        return resolved

    def run_pipeline(self, dataset_path, user_goal):
        """Executes the full automated data science pipeline."""
        target_column = user_goal.get('target')
        drop_columns = user_goal.get('drop_columns', [])
        
        # LLM-determined metric configuration
        primary_metric = user_goal.get('primary_metric')
        optimization_goal = user_goal.get('optimization_goal')  # "maximize" or "minimize"
        
        # ── 1. Load & Clean ──
        logger.info("Loading data...")
        df = self.analyzer.load_data(dataset_path)
        
        if drop_columns:
            actual_drops = [c for c in drop_columns if c in df.columns]
            if actual_drops:
                logger.info(f"Dropping {len(actual_drops)} leaky/irrelevant columns: {actual_drops}")
                df = df.drop(columns=actual_drops)
        
        # ── 2. Analyze ──
        logger.info("Analyzing dataset...")
        metadata = self.analyzer.analyze(df, target_column=target_column)
        
        dataset_name = metadata.get("dataset_type", "uploaded_dataset")
        with SessionLocal() as db:
            db_summary = DatasetSummary(
                dataset_name=dataset_name,
                rows=metadata['rows'],
                columns=metadata['columns'],
                metadata_json=metadata
            )
            db.add(db_summary)
            db.commit()
            db.refresh(db_summary)
            dataset_id = db_summary.id
              
        # ── 3. Plan ──
        logger.info("Planning pipeline...")
        plan = self.planner.generate_plan(metadata, user_goal)
        task_type = plan.get('task_type')
        
        with SessionLocal() as db:
            db_plan = PipelineMetadata(
                dataset_id=dataset_id,
                task_objective=task_type,
                preprocessing_steps=plan.get('pipeline_steps', []),
                selected_models=plan.get('models', [])
            )
            db.add(db_plan)
            db.commit()
            db.refresh(db_plan)
            pipeline_id = db_plan.id
              
        # ── 4. Build Pipelines ──
        logger.info("Building pipelines...")
        pipelines = self.builder.build_pipelines(plan, metadata)
        
        # ── 5. Resolve Target & Guard ──
        target_column = self._resolve_target_column(df, target_column)
        task_type = plan.get('task_type')
        
        supervised_tasks = ['classification', 'regression', 'nlp_classification', 'time_series_forecasting']
        if task_type in supervised_tasks and (target_column is None or target_column not in df.columns):
            raise ValueError(
                f"Supervised task '{task_type}' requires a valid target column, but got '{target_column}'. "
                f"Available columns: {df.columns.tolist()}"
            )
        
        # ── 5b. Time-Series Feature Engineering ──
        is_time_series = metadata.get("is_time_series", False)
        time_column = metadata.get("time_column")
        group_columns = metadata.get("group_columns", [])
        
        if is_time_series and target_column and time_column:
            logger.info(f"Generating time-series features (time={time_column}, groups={group_columns})...")
            ts_generator = TimeSeriesFeatureGenerator(
                time_column=time_column,
                target_column=target_column,
                group_columns=group_columns
            )
            df = ts_generator.generate(df)
            
            # Re-analyze metadata after feature generation (new columns added)
            metadata = self.analyzer.analyze(df, target_column=target_column)
            metadata["is_time_series"] = True
            metadata["time_column"] = time_column
            metadata["group_columns"] = group_columns
            
            # Rebuild pipelines with updated feature set
            plan_updated = self.planner.generate_plan(metadata, user_goal)
            pipelines = self.builder.build_pipelines(plan_updated, metadata)
            logger.info(f"Time-series features complete. New shape: {df.shape}")
        
        # ── 6. Three-Way Split: Train (70%) / Validation (15%) / Eval (15%) ──
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df.copy()
            y = None
        
        if y is not None:
            if is_time_series:
                # TEMPORAL SPLIT — preserve chronological order
                n = len(df)
                train_end = int(n * 0.70)
                val_end = int(n * 0.85)
                
                X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
                X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
                X_eval, y_eval = X.iloc[val_end:], y.iloc[val_end:]
                
                logger.info(f"Temporal split: train={len(X_train)} (70%), "
                             f"val={len(X_val)} (15%), eval={len(X_eval)} (15%)")
            else:
                # RANDOM SPLIT — standard for non-TS tasks
                stratify_col = y if task_type in ['classification', 'nlp_classification'] else None
                try:
                    X_temp, X_eval, y_temp, y_eval = train_test_split(
                        X, y, test_size=0.15, random_state=42, stratify=stratify_col
                    )
                    stratify_temp = y_temp if task_type in ['classification', 'nlp_classification'] else None
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=0.176, random_state=42, stratify=stratify_temp
                    )
                except ValueError:
                    X_temp, X_eval, y_temp, y_eval = train_test_split(X, y, test_size=0.15, random_state=42)
                    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)
                
                logger.info(f"Random split: train={len(X_train)} ({len(X_train)/len(X)*100:.0f}%), "
                             f"val={len(X_val)} ({len(X_val)/len(X)*100:.0f}%), "
                             f"eval={len(X_eval)} ({len(X_eval)/len(X)*100:.0f}%)")
            
            # Persist eval set + train/val for retraining
            os.makedirs("eval_data", exist_ok=True)
            X_eval.to_csv("eval_data/X_eval.csv", index=False)
            y_eval.to_csv("eval_data/y_eval.csv", index=False)
            X_train.to_csv("eval_data/X_train.csv", index=False)
            y_train.to_csv("eval_data/y_train.csv", index=False)
            X_val.to_csv("eval_data/X_val.csv", index=False)
            y_val.to_csv("eval_data/y_val.csv", index=False)
            
            run_meta = {
                "dataset_path": dataset_path,
                "target_column": target_column,
                "task_type": task_type,
                "is_time_series": is_time_series,
                "drop_columns": drop_columns,
                "pipeline_id": pipeline_id,
                "feature_columns": X_train.columns.tolist(),
                "primary_metric": primary_metric,
                "optimization_goal": optimization_goal,
                "plan": plan,
                "metadata": {
                    "pipeline_steps": plan.get("pipeline_steps", []),
                    "preprocessing_details": plan.get("preprocessing_details", {}),
                    "feature_names": metadata.get("feature_names", {})
                }
            }
            with open("eval_data/run_meta.json", "w") as f:
                json.dump(run_meta, f)
        else:
            X_train, X_val = X.copy(), X.copy()
            X_eval = X.copy()
            y_train, y_val, y_eval = None, None, None
              
        # ── 7. Model Training & Evaluation (uses VALIDATION set for selection) ──
        best_model_name = None
        # Initialize based on optimization goal
        if optimization_goal == 'minimize':
            best_metric_val = float('inf')
        else:
            best_metric_val = -float('inf')
        
        results = {}
        fitted_pipelines = {}
        os.makedirs("trained_models", exist_ok=True)
        
        for model_name, pipeline in pipelines.items():
            logger.info(f"Training {model_name}...")
            start_time = time.time()
            try:
                if y_train is not None:
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_val)  # Evaluate on VALIDATION set
                else:
                    pipeline.fit(X_train)
                    y_pred = pipeline.predict(X_val)
                    
                train_time = time.time() - start_time
                
                # Get probabilities for ROC AUC
                y_prob = None
                if hasattr(pipeline, "predict_proba"):
                    try:
                        y_prob = pipeline.predict_proba(X_val)
                    except Exception:
                        pass
                
                logger.info(f"Evaluating {model_name} on validation set...")
                metrics = self.evaluator.evaluate(
                    y_true=y_val if y_val is not None else X_val,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    task_type=task_type,
                    training_time=train_time
                )
                
                # ── Dynamic best model selection using LLM-determined metric ──
                # Fallback defaults if LLM didn't specify
                if not primary_metric:
                    if task_type in ['classification', 'nlp_classification']:
                        primary_metric = 'f1_score'
                    elif task_type in ['regression', 'time_series_forecasting']:
                        primary_metric = 'r2_score'
                    elif task_type == 'clustering':
                        primary_metric = 'silhouette_score'
                    else:
                        primary_metric = 'f1_score'
                if not optimization_goal:
                    optimization_goal = 'minimize' if primary_metric in ['rmse', 'mae'] else 'maximize'
                
                current_score = metrics.get(primary_metric)
                if current_score is not None:
                    if optimization_goal == 'maximize' and current_score > best_metric_val:
                        best_metric_val = current_score
                        best_model_name = model_name
                    elif optimization_goal == 'minimize' and current_score < best_metric_val:
                        best_metric_val = current_score
                        best_model_name = model_name
                
                logger.info(f"{model_name}: {primary_metric}={current_score} ({optimization_goal})")
                    
                # Explainability
                logger.info(f"Extracting explainability for {model_name}...")
                importances = self.explainer.explain(pipeline, X_train, task_type)
                
                # DB Storage
                with SessionLocal() as db:
                    db_metrics = ModelMetrics(
                        pipeline_id=pipeline_id,
                        model_name=model_name,
                        accuracy=metrics.get('accuracy'),
                        precision=metrics.get('precision'),
                        recall=metrics.get('recall'),
                        f1_score=metrics.get('f1_score'),
                        rmse=metrics.get('rmse'),
                        mae=metrics.get('mae'),
                        r2_score=metrics.get('r2_score'),
                        silhouette_score=metrics.get('silhouette_score'),
                        roc_auc=metrics.get('roc_auc'),
                        training_time_seconds=metrics.get('training_time_seconds'),
                        is_best_model=0
                    )
                    db.add(db_metrics)
                    db.commit()
                    db.refresh(db_metrics)
                    
                    for f_name, i_val in importances.items():
                        db.add(FeatureImportance(metric_id=db_metrics.id, feature_name=f_name, importance_score=i_val))
                    db.commit()
                    
                results[model_name] = {"metrics": metrics, "importances": importances}
                fitted_pipelines[model_name] = pipeline
                joblib.dump(pipeline, f"trained_models/{model_name}.joblib")
                
            except Exception as e:
                logger.error(f"Pipeline for {model_name} failed: {e}")

        # Mark best model
        if best_model_name:
            with SessionLocal() as db:
                db.query(ModelMetrics).filter(
                    ModelMetrics.pipeline_id == pipeline_id,
                    ModelMetrics.model_name == best_model_name
                ).update({"is_best_model": 1})
                db.commit()

        logger.info("Orchestration complete. All results written to local SQL db.")
        return {
            "dataset_id": dataset_id,
            "pipeline_id": pipeline_id,
            "best_model": best_model_name,
            "results": results,
            "fitted_pipelines": fitted_pipelines,
            "feature_columns": X_train.columns.tolist(),
            "task_type": task_type
        }

    @staticmethod
    def run_eval_inference(model_name):
        """Run the model on the held-out eval set and return metrics."""
        from backend.evaluation.evaluator import PipelineEvaluator

        model_path = f"trained_models/{model_name}.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists("eval_data/X_eval.csv"):
            raise FileNotFoundError("No eval set found. Run a pipeline first.")

        pipeline = joblib.load(model_path)
        X_eval = pd.read_csv("eval_data/X_eval.csv")
        y_eval = pd.read_csv("eval_data/y_eval.csv").iloc[:, 0]

        with open("eval_data/run_meta.json", "r") as f:
            run_meta = json.load(f)

        task_type = run_meta.get("task_type", "classification")

        y_pred = pipeline.predict(X_eval)

        y_prob = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X_eval)
            except Exception:
                pass

        evaluator = PipelineEvaluator()
        metrics = evaluator.evaluate(y_true=y_eval, y_pred=y_pred, y_prob=y_prob, task_type=task_type)

        return {
            "metrics": metrics,
            "task_type": task_type,
            "y_eval": y_eval.tolist(),
            "y_pred": y_pred.tolist(),
            "eval_size": len(X_eval)
        }

    def trigger_retraining(self, model_name, threshold_override=None, progress_callback=None):
        """Delegates to RetrainingController for intelligent model improvement."""
        from backend.retraining.retrain_controller import RetrainingController

        if not os.path.exists("eval_data/run_meta.json"):
            raise FileNotFoundError("No run metadata found. Run a pipeline first.")

        with open("eval_data/run_meta.json", "r") as f:
            run_meta = json.load(f)

        task_type = run_meta["task_type"]
        pipeline_id = run_meta["pipeline_id"]
        meta_info = run_meta.get("metadata", {})
        primary_metric = run_meta.get("primary_metric")
        optimization_goal = run_meta.get("optimization_goal")

        # Load persisted train/val data
        X_train = pd.read_csv("eval_data/X_train.csv")
        y_train = pd.read_csv("eval_data/y_train.csv").iloc[:, 0]
        X_val = pd.read_csv("eval_data/X_val.csv")
        y_val = pd.read_csv("eval_data/y_val.csv").iloc[:, 0]

        # Load current model
        model_path = f"trained_models/{model_name}.joblib"
        current_pipeline = joblib.load(model_path)

        controller = RetrainingController(
            task_type=task_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            pipeline_id=pipeline_id,
            metadata=meta_info,
            current_best_model_name=model_name,
            current_best_pipeline=current_pipeline,
            threshold_override=threshold_override,
            primary_metric=primary_metric,
            optimization_goal=optimization_goal
        )

        result = controller.run(progress_callback=progress_callback)

        # If improved, save the new model
        if result["improved"]:
            joblib.dump(result["model"], model_path)
            logger.info(f"Replaced model on disk: {model_path}")

            # If model name changed (alternative model), save under new name too
            if result["model_name"] != model_name:
                new_path = f"trained_models/{result['model_name']}.joblib"
                joblib.dump(result["model"], new_path)
                logger.info(f"Also saved as: {new_path}")

        return result
