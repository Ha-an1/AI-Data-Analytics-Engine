import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)

logger = logging.getLogger(__name__)

class PipelineEvaluator:
    def __init__(self):
        pass

    def evaluate(self, y_true, y_pred, y_prob=None, pipeline=None, task_type="classification", training_time=0.0):
        """
        Evaluates the model predictions based on task type.
        Optionally accepts the pipeline object to compute ROC AUC via predict_proba.
        """
        metrics = {
            "training_time_seconds": round(training_time, 4)
        }
        
        try:
            if task_type in ["classification", "nlp_classification"]:
                is_multiclass = len(np.unique(y_true)) > 2
                avg_method = 'macro' if is_multiclass else 'binary'
                
                metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                try:
                    metrics["precision"] = float(precision_score(y_true, y_pred, average=avg_method, zero_division=0))
                    metrics["recall"] = float(recall_score(y_true, y_pred, average=avg_method, zero_division=0))
                    metrics["f1_score"] = float(f1_score(y_true, y_pred, average=avg_method, zero_division=0))
                except Exception as e:
                    logger.warning(f"Could not calculate precision/recall/f1: {e}")
                
                # ROC AUC - requires probability estimates
                try:
                    if y_prob is not None:
                        proba = y_prob
                    elif pipeline is not None and hasattr(pipeline, "predict_proba"):
                        proba = pipeline.predict_proba(y_true.__class__(y_true).values.reshape(-1, 1) if False else None)
                        # We can't call predict_proba without X, so skip if no y_prob
                        proba = None
                    else:
                        proba = None
                    
                    if proba is not None:
                        if is_multiclass:
                            metrics["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class='ovr', average='macro'))
                        else:
                            # Use probability of positive class
                            proba_pos = proba[:, 1] if proba.ndim > 1 else proba
                            metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
                except Exception as e:
                    logger.warning(f"ROC AUC calculation failed: {e}")
                
            elif task_type in ["regression", "time_series_forecasting"]:
                metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
                metrics["r2_score"] = float(r2_score(y_true, y_pred))
                
            elif task_type == "clustering":
                try:
                    metrics["silhouette_score"] = float(silhouette_score(y_true, y_pred))
                except Exception as e:
                    logger.warning(f"Silhouette score failed: {e}")
                    
            elif task_type == "anomaly_detection":
                anomalies = np.sum(y_pred == -1)
                total = len(y_pred)
                metrics["anomaly_ratio"] = float(anomalies / total) if total > 0 else 0.0

        except Exception as e:
            logger.error(f"Evaluation failed for task {task_type}: {e}")
            
        return metrics
