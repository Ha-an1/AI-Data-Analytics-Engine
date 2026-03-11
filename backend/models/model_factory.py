from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    IsolationForest
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans, DBSCAN

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class ModelFactory:
    @staticmethod
    def get_model(model_name, task_type="classification", random_state=42):
        model_name = model_name.lower()
        if model_name == "logistic_regression":
            return LogisticRegression(random_state=random_state, max_iter=1000)
        elif model_name == "linear_regression":
            return LinearRegression()
        elif model_name == "random_forest":
            return RandomForestClassifier(random_state=random_state)
        elif model_name == "random_forest_regressor":
            return RandomForestRegressor(random_state=random_state)
        elif model_name == "xgboost":
            return XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        elif model_name == "xgboost_regressor":
            return XGBRegressor(random_state=random_state)
        elif model_name == "gradient_boosting":
            return GradientBoostingClassifier(random_state=random_state)
        elif model_name == "gradient_boosting_regressor":
            return GradientBoostingRegressor(random_state=random_state)
        elif model_name == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not installed. Run: pip install lightgbm")
            return LGBMClassifier(random_state=random_state, verbose=-1)
        elif model_name == "lightgbm_regressor":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not installed. Run: pip install lightgbm")
            return LGBMRegressor(random_state=random_state, verbose=-1)
        elif model_name == "kmeans":
            return KMeans(random_state=random_state)
        elif model_name == "dbscan":
            return DBSCAN()
        elif model_name == "isolation_forest":
            return IsolationForest(random_state=random_state)
        else:
            raise ValueError(f"Model {model_name} is not supported by ModelFactory.")

    @staticmethod
    def get_param_grid(model_name):
        """Returns hyperparameter search space for RandomizedSearchCV."""
        grids = {
            "random_forest": {
                "model__n_estimators": [50, 100, 200, 300],
                "model__max_depth": [5, 10, 20, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "random_forest_regressor": {
                "model__n_estimators": [50, 100, 200, 300],
                "model__max_depth": [5, 10, 20, None],
                "model__min_samples_split": [2, 5, 10],
            },
            "xgboost": {
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [3, 5, 7, 10],
                "model__subsample": [0.6, 0.8, 1.0],
                "model__n_estimators": [50, 100, 200, 300],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
            },
            "xgboost_regressor": {
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [3, 5, 7, 10],
                "model__subsample": [0.6, 0.8, 1.0],
                "model__n_estimators": [50, 100, 200, 300],
            },
            "gradient_boosting": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            },
            "gradient_boosting_regressor": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            },
            "lightgbm": {
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__num_leaves": [15, 31, 63, 127],
                "model__n_estimators": [50, 100, 200, 300],
                "model__subsample": [0.6, 0.8, 1.0],
            },
            "lightgbm_regressor": {
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__num_leaves": [15, 31, 63, 127],
                "model__n_estimators": [50, 100, 200, 300],
            },
            "logistic_regression": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__solver": ["lbfgs", "liblinear"],
            },
        }
        return grids.get(model_name.lower(), {})

    @staticmethod
    def get_alternative_models(task_type, exclude_models=None):
        """Returns model names not yet tried for the given task type."""
        exclude = set(m.lower() for m in (exclude_models or []))
        
        if task_type in ["classification", "nlp_classification"]:
            all_models = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost", "lightgbm"]
        elif task_type in ["regression", "time_series_forecasting"]:
            all_models = ["linear_regression", "random_forest_regressor", "gradient_boosting_regressor", "xgboost_regressor", "lightgbm_regressor"]
        else:
            return []
        
        return [m for m in all_models if m not in exclude]
