import time
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from backend.models.model_factory import ModelFactory
from backend.pipeline.pipeline_builder import PipelineBuilder
from backend.evaluation.evaluator import PipelineEvaluator
from backend.database.db_manager import SessionLocal
from backend.database.models import ModelRuns

logger = logging.getLogger(__name__)

# ── Configuration ──
MAX_RETRAIN_ATTEMPTS = 3
THRESHOLDS = {
    "classification":           {"metric": "f1_score",  "direction": "maximize", "min": 0.75},
    "nlp_classification":       {"metric": "f1_score",  "direction": "maximize", "min": 0.75},
    "regression":               {"metric": "r2_score",  "direction": "maximize", "min": 0.70},
    "time_series_forecasting":  {"metric": "r2_score",  "direction": "maximize", "min": 0.70},
}


class RetrainingController:
    """
    Intelligent retraining controller that systematically improves model
    performance through 3 strategies:
      1. Hyperparameter tuning (RandomizedSearchCV)
      2. Alternative model exploration
      3. Data-level improvements (SMOTE, feature engineering)
    """

    def __init__(self, task_type, X_train, y_train, X_val, y_val,
                 pipeline_id, metadata, current_best_model_name, current_best_pipeline,
                 threshold_override=None, primary_metric=None, optimization_goal=None):
        self.task_type = task_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.pipeline_id = pipeline_id
        self.metadata = metadata
        self.current_best_model_name = current_best_model_name
        self.current_best_pipeline = current_best_pipeline
        self.evaluator = PipelineEvaluator()
        self.builder = PipelineBuilder()

        # Use LLM-determined metric or fall back to defaults
        cfg = THRESHOLDS.get(task_type, THRESHOLDS["classification"])
        self.primary_metric = primary_metric or cfg["metric"]
        self.direction = optimization_goal or cfg["direction"]
        self.threshold = threshold_override if threshold_override is not None else cfg["min"]

        # Track all attempts
        self.attempts_log = []
        self.best_score = None
        self.best_pipeline = current_best_pipeline
        self.best_model_name = current_best_model_name

    def _score(self, pipeline, X, y):
        """Evaluate pipeline and return (metrics_dict, primary_score)."""
        y_pred = pipeline.predict(X)

        # Try to get probabilities for ROC AUC
        y_prob = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X)
            except Exception:
                pass

        metrics = self.evaluator.evaluate(
            y_true=y, y_pred=y_pred, y_prob=y_prob,
            task_type=self.task_type
        )
        score = metrics.get(self.primary_metric, 0)
        return metrics, score

    def _is_better(self, new_score, old_score):
        if old_score is None:
            return True
        if self.direction == "maximize":
            return new_score > old_score
        return new_score < old_score

    def _meets_threshold(self, score):
        if self.direction == "maximize":
            return score >= self.threshold
        return score <= self.threshold

    def _log_attempt(self, attempt, strategy, model_name, params, val_score, train_time, improved):
        record = {
            "attempt": attempt,
            "strategy": strategy,
            "model_name": model_name,
            "parameters": params,
            "validation_score": round(val_score, 4) if val_score is not None else None,
            "training_duration": round(train_time, 2),
            "improved": improved
        }
        self.attempts_log.append(record)
        logger.info(f"Retrain attempt {attempt}: strategy={strategy}, model={model_name}, "
                     f"val_score={val_score:.4f}, improved={improved}")

        # Persist to database
        try:
            with SessionLocal() as db:
                db.add(ModelRuns(
                    pipeline_id=self.pipeline_id,
                    model_name=model_name,
                    strategy=strategy,
                    attempt=attempt,
                    parameters=params,
                    validation_score=val_score,
                    training_duration=train_time,
                    improved=1 if improved else 0
                ))
                db.commit()
        except Exception as e:
            logger.warning(f"Failed to log experiment to DB: {e}")

    # ─────────────────────────────────────────────
    # Strategy 1: Hyperparameter Tuning
    # ─────────────────────────────────────────────
    def _strategy_hyperparam_tuning(self, attempt):
        """RandomizedSearchCV on the current best model."""
        logger.info(f"[Strategy 1] Hyperparameter tuning for {self.best_model_name}...")

        param_grid = ModelFactory.get_param_grid(self.best_model_name)
        if not param_grid:
            logger.warning(f"No param grid for {self.best_model_name}, skipping hyperparameter tuning.")
            self._log_attempt(attempt, "hyperparameter_tuning", self.best_model_name, {}, 0, 0, False)
            return False

        # Map primary metric to sklearn scoring string
        SCORING_MAP = {
            'f1_score': 'f1_macro', 'accuracy': 'accuracy', 'precision': 'precision_macro',
            'recall': 'recall_macro', 'roc_auc': 'roc_auc_ovr',
            'r2_score': 'r2', 'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error',
        }
        scoring = SCORING_MAP.get(self.primary_metric, 'f1_macro')

        search = RandomizedSearchCV(
            self.best_pipeline,
            param_distributions=param_grid,
            n_iter=min(20, max(5, len(param_grid) * 3)),
            scoring=scoring,
            cv=3,
            random_state=42,
            n_jobs=-1,
            error_score='raise'
        )

        start = time.time()
        try:
            search.fit(self.X_train, self.y_train)
        except Exception as e:
            logger.error(f"Hyperparameter search failed: {e}")
            self._log_attempt(attempt, "hyperparameter_tuning", self.best_model_name, {}, 0, 0, False)
            return False
        train_time = time.time() - start

        tuned_pipeline = search.best_estimator_
        metrics, val_score = self._score(tuned_pipeline, self.X_val, self.y_val)
        improved = self._is_better(val_score, self.best_score)

        best_params = {k: str(v) for k, v in search.best_params_.items()}
        self._log_attempt(attempt, "hyperparameter_tuning", self.best_model_name, best_params, val_score, train_time, improved)

        if improved:
            self.best_score = val_score
            self.best_pipeline = tuned_pipeline
            logger.info(f"[Strategy 1] Improved! New score: {val_score:.4f}")

        return self._meets_threshold(val_score if improved else self.best_score or 0)

    # ─────────────────────────────────────────────
    # Strategy 2: Alternative Model Exploration
    # ─────────────────────────────────────────────
    def _strategy_alternative_models(self, attempt):
        """Try model families not yet explored."""
        logger.info("[Strategy 2] Exploring alternative model families...")

        tried_models = [log["model_name"] for log in self.attempts_log]
        tried_models.append(self.current_best_model_name)
        alternatives = ModelFactory.get_alternative_models(self.task_type, exclude_models=tried_models)

        if not alternatives:
            logger.warning("No alternative models available to try.")
            self._log_attempt(attempt, "alternative_model", "none", {}, 0, 0, False)
            return False

        overall_improved = False

        for alt_model_name in alternatives:
            logger.info(f"  Training alternative: {alt_model_name}...")
            try:
                # Build a fresh pipeline with this model
                plan_stub = {
                    "task_type": self.task_type,
                    "pipeline_steps": self.metadata.get("pipeline_steps", []),
                    "preprocessing_details": self.metadata.get("preprocessing_details", {}),
                    "models": [alt_model_name]
                }
                alt_pipelines = self.builder.build_pipelines(plan_stub, self.metadata)
                if alt_model_name not in alt_pipelines:
                    continue

                alt_pipeline = alt_pipelines[alt_model_name]
                start = time.time()
                alt_pipeline.fit(self.X_train, self.y_train)
                train_time = time.time() - start

                metrics, val_score = self._score(alt_pipeline, self.X_val, self.y_val)
                improved = self._is_better(val_score, self.best_score)

                self._log_attempt(attempt, "alternative_model", alt_model_name, {}, val_score, train_time, improved)

                if improved:
                    self.best_score = val_score
                    self.best_pipeline = alt_pipeline
                    self.best_model_name = alt_model_name
                    overall_improved = True
                    logger.info(f"  {alt_model_name} improved score to {val_score:.4f}!")

                if self._meets_threshold(val_score):
                    return True

            except Exception as e:
                logger.error(f"  Alternative model {alt_model_name} failed: {e}")

        return self._meets_threshold(self.best_score or 0)

    # ─────────────────────────────────────────────
    # Strategy 3: Data-Level Improvements
    # ─────────────────────────────────────────────
    def _strategy_data_improvements(self, attempt):
        """Apply SMOTE, remove correlated features, and retrain."""
        logger.info("[Strategy 3] Data-level improvements...")

        X_improved = self.X_train.copy()
        y_improved = self.y_train.copy()

        # 3a. Remove highly correlated features (>0.95)
        try:
            numeric_cols = X_improved.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = X_improved[numeric_cols].corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
                if to_drop:
                    logger.info(f"  Dropping {len(to_drop)} highly correlated features: {to_drop}")
                    X_improved = X_improved.drop(columns=to_drop)
                    # Also drop from validation
                    X_val_improved = self.X_val.drop(columns=[c for c in to_drop if c in self.X_val.columns])
                else:
                    X_val_improved = self.X_val.copy()
            else:
                X_val_improved = self.X_val.copy()
        except Exception as e:
            logger.warning(f"  Correlation removal failed: {e}")
            X_val_improved = self.X_val.copy()

        # 3b. SMOTE for imbalanced classification
        if self.task_type in ["classification", "nlp_classification"]:
            try:
                from imblearn.over_sampling import SMOTE
                # Only apply on numeric data
                numeric_cols = X_improved.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    class_counts = y_improved.value_counts()
                    minority_count = class_counts.min()
                    if minority_count >= 6:  # SMOTE needs >= k_neighbors + 1 samples
                        k = min(5, minority_count - 1)
                        smote = SMOTE(random_state=42, k_neighbors=k)
                        X_improved, y_improved = smote.fit_resample(X_improved, y_improved)
                        logger.info(f"  SMOTE applied: {len(self.X_train)} -> {len(X_improved)} samples")
                    else:
                        logger.warning(f"  SMOTE skipped: minority class too small ({minority_count} samples)")
            except Exception as e:
                logger.warning(f"  SMOTE failed: {e}")

        # 3c. Retrain best model on improved data
        try:
            plan_stub = {
                "task_type": self.task_type,
                "pipeline_steps": self.metadata.get("pipeline_steps", []),
                "preprocessing_details": self.metadata.get("preprocessing_details", {}),
                "models": [self.best_model_name]
            }

            # Rebuild metadata with potentially fewer features
            improved_metadata = dict(self.metadata)
            improved_metadata["feature_names"] = {
                "numerical": X_improved.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical": X_improved.select_dtypes(exclude=[np.number]).columns.tolist(),
                "text": []
            }

            improved_pipelines = self.builder.build_pipelines(plan_stub, improved_metadata)
            if self.best_model_name not in improved_pipelines:
                self._log_attempt(attempt, "data_improvement", self.best_model_name, {}, 0, 0, False)
                return False

            pipeline = improved_pipelines[self.best_model_name]

            start = time.time()
            pipeline.fit(X_improved, y_improved)
            train_time = time.time() - start

            metrics, val_score = self._score(pipeline, X_val_improved, self.y_val)
            improved = self._is_better(val_score, self.best_score)

            data_actions = []
            if len(X_improved) != len(self.X_train):
                data_actions.append(f"SMOTE: {len(self.X_train)}->{len(X_improved)} rows")
            if X_improved.shape[1] != self.X_train.shape[1]:
                data_actions.append(f"Dropped {self.X_train.shape[1] - X_improved.shape[1]} correlated features")

            self._log_attempt(attempt, "data_improvement", self.best_model_name,
                              {"actions": data_actions}, val_score, train_time, improved)

            if improved:
                self.best_score = val_score
                self.best_pipeline = pipeline
                logger.info(f"[Strategy 3] Improved! New score: {val_score:.4f}")

            return self._meets_threshold(val_score if improved else self.best_score or 0)

        except Exception as e:
            logger.error(f"Data improvement strategy failed: {e}")
            self._log_attempt(attempt, "data_improvement", self.best_model_name, {}, 0, 0, False)
            return False

    # ─────────────────────────────────────────────
    # Main Retraining Loop
    # ─────────────────────────────────────────────
    def run(self, progress_callback=None):
        """
        Execute the 3-strategy retraining loop.
        
        Args:
            progress_callback: Optional callable(attempt, strategy, status_msg) for UI updates.
            
        Returns:
            dict with keys: model, model_name, attempts, improved, final_score
        """
        logger.info(f"=== Retraining Controller started ===")
        logger.info(f"Task: {self.task_type} | Metric: {self.primary_metric} | Threshold: {self.threshold}")

        # Compute baseline score
        _, baseline_score = self._score(self.best_pipeline, self.X_val, self.y_val)
        self.best_score = baseline_score
        logger.info(f"Baseline validation score: {baseline_score:.4f}")

        if self._meets_threshold(baseline_score):
            logger.info("Baseline already meets threshold. No retraining needed.")
            return {
                "model": self.best_pipeline,
                "model_name": self.best_model_name,
                "attempts": self.attempts_log,
                "improved": False,
                "final_score": baseline_score,
                "already_passed": True
            }

        strategies = [
            (1, "Hyperparameter Tuning", self._strategy_hyperparam_tuning),
            (2, "Alternative Model Exploration", self._strategy_alternative_models),
            (3, "Data-Level Improvements", self._strategy_data_improvements),
        ]

        for attempt, strategy_name, strategy_fn in strategies:
            if attempt > MAX_RETRAIN_ATTEMPTS:
                break

            logger.info(f"\n--- Attempt {attempt}/{MAX_RETRAIN_ATTEMPTS}: {strategy_name} ---")
            if progress_callback:
                progress_callback(attempt, strategy_name, f"Running {strategy_name}...")

            threshold_met = strategy_fn(attempt)

            if threshold_met:
                logger.info(f"Threshold met after strategy: {strategy_name}")
                if progress_callback:
                    progress_callback(attempt, strategy_name, f"✅ Threshold met!")
                break

        improved = self._is_better(self.best_score, baseline_score)
        logger.info(f"=== Retraining complete === Improved: {improved}, "
                     f"Score: {baseline_score:.4f} -> {self.best_score:.4f}")

        return {
            "model": self.best_pipeline,
            "model_name": self.best_model_name,
            "attempts": self.attempts_log,
            "improved": improved,
            "final_score": self.best_score,
            "baseline_score": baseline_score,
            "already_passed": False
        }
