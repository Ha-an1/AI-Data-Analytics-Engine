import json
import logging

logger = logging.getLogger(__name__)

class PipelinePlanner:
    def __init__(self):
        # Base steps that almost all supervised pipelines will have
        self.base_supervised_steps = [
            "train_test_split",
            "model_training",
            "evaluation",
            "explainability",
            "database_export"
        ]
        
    def generate_plan(self, metadata, user_goal):
        """
        Generates a pipeline execution plan based on dataset metadata and user objectives.
        
        Args:
            metadata (dict): The output from DatasetAnalyzer.analyze
            user_goal (dict): User preferences, e.g. {"task": "classification", "target": "churn"}
        
        Returns:
            dict: The structured pipeline plan.
        """
        task = user_goal.get("task", metadata.get("detected_task", "classification")).lower()
        
        plan = {
            "task_type": task,
            "pipeline_steps": [],
            "preprocessing_details": {},
            "models": []
        }
        
        # 1. PREPROCESSING RULES
        # Missing value imputation
        if metadata.get("missing_values"):
            plan["pipeline_steps"].append("missing_value_imputation")
            plan["preprocessing_details"]["imputation"] = "simple_imputer" # mean for num, most_frequent for cat
            
        # Categorical encoding
        if metadata.get("categorical_features", 0) > 0:
            plan["pipeline_steps"].append("categorical_encoding")
            plan["preprocessing_details"]["encoding"] = "one_hot" 
            
        # Feature scaling (always good practice for linear models and clustering)
        if metadata.get("numerical_features", 0) > 0:
            plan["pipeline_steps"].append("feature_scaling")
            plan["preprocessing_details"]["scaling"] = "standard"

        # Text features / NLP
        if metadata.get("text_features", 0) > 0 and task == "nlp_classification":
            plan["pipeline_steps"].append("tfidf_vectorization")

        # Class imbalance handling
        imbalance = metadata.get("imbalance_ratio")
        if task == "classification" and imbalance:
            try:
                # parse "1:X"
                ratio_val = int(imbalance.split(":")[1])
                if ratio_val >= 10:
                    plan["pipeline_steps"].append("handle_imbalance")
                    plan["preprocessing_details"]["imbalance"] = "class_weight" # easier than SMOTE for initial MVP
            except Exception as e:
                logger.warning(f"Could not parse imbalance ratio {imbalance}: {e}")

        # Add base execution steps
        plan["pipeline_steps"].extend(self.base_supervised_steps)
        
        # Override for unsupervised tasks
        if task in ["clustering", "anomaly_detection"]:
            plan["pipeline_steps"] = [step for step in plan["pipeline_steps"] if step not in ["train_test_split", "explainability"]]

        # 2. MODEL SELECTION RULES
        dataset_size = metadata.get("rows", 0)
        
        if task == "classification":
            if dataset_size > 100000:
                # Fast models for large data
                plan["models"] = ["logistic_regression", "xgboost"]
            else:
                plan["models"] = ["logistic_regression", "random_forest", "xgboost", "gradient_boosting"]
                
        elif task == "regression":
             if dataset_size > 100000:
                 plan["models"] = ["linear_regression", "xgboost_regressor"]
             else:
                 plan["models"] = ["linear_regression", "random_forest_regressor", "xgboost_regressor"]
                 
        elif task == "clustering":
            if dataset_size > 50000:
                plan["models"] = ["kmeans"] # DBSCAN is too slow O(N^2)
            else:
                plan["models"] = ["kmeans", "dbscan"]
                
        elif task == "anomaly_detection":
            plan["models"] = ["isolation_forest"]
            
        elif task == "nlp_classification":
            plan["models"] = ["logistic_regression"] # TF-IDF + LR is standard baseline
            
        elif task == "time_series_forecasting":
             plan["models"] = ["random_forest_regressor", "xgboost_regressor"] # Using sliding window features
             
        else:
            plan["models"] = ["random_forest"] # Fallback
            
        return plan
