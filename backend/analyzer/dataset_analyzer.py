import pandas as pd
import numpy as np
import json
import logging
from sqlalchemy import create_engine
from backend.features.time_series_features import detect_time_series

logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    def __init__(self, sample_size=100000):
        self.sample_size = sample_size

    def load_data(self, source, source_type="csv"):
        """Loads data from CSV, JSON, or SQL."""
        try:
            if source_type == "csv":
                df = pd.read_csv(source)
            elif source_type == "json":
                df = pd.read_json(source)
            elif source_type == "sql":
                # source is expected to be a tuple (connection_string, query)
                conn_str, query = source
                engine = create_engine(conn_str)
                df = pd.read_sql(query, con=engine)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            # Sampling for extremely large datasets
            if len(df) > self.sample_size:
                logger.warning(f"Dataset has {len(df)} rows. Downsampling to {self.sample_size} for analysis.")
                df = df.sample(n=self.sample_size, random_state=42)
                
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def analyze(self, df, target_column=None):
        """Analyzes dataset and emits structured JSON metadata for the planner."""
        rows, columns = df.shape
        
        # Determine feature types
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Text feature heuristics (categorical columns with many unique values, usually longer strings)
        text_features = []
        actual_categorical = []
        for col in categorical_cols:
            # Exclude boolean or very low cardinality from text check
            if df[col].nunique() > 50 and df[col].dtype == 'object':
                # check average length
                avg_len = df[col].dropna().astype(str).str.len().mean()
                if avg_len > 20:
                    text_features.append(col)
                else:
                    actual_categorical.append(col)
            else:
                actual_categorical.append(col)
                
        # Remove target column from feature lists to avoid data leakage in planner
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
        if target_column in actual_categorical:
            actual_categorical.remove(target_column)
        if target_column in text_features:
            text_features.remove(target_column)

        # Missing values assessment (only columns with > 0 missing)
        missing_ratios = (df.isnull().sum() / rows).to_dict()
        missing_values = {k: round(v, 4) for k, v in missing_ratios.items() if v > 0}

        # Class imbalance assessment (if classification target)
        imbalance_ratio = None
        task_type = "unknown"
        if target_column and target_column in df.columns:
            # Simple heuristic for classification vs regression
            if df[target_column].dtype in ['object', 'category', 'bool'] or df[target_column].nunique() < 20:
                task_type = "classification"
                val_counts = df[target_column].value_counts()
                if len(val_counts) > 1:
                    majority = val_counts.max()
                    minority = val_counts.min()
                    ratio = majority / minority
                    # Formatting as 1:X approximation
                    imbalance_ratio = f"1:{int(round(ratio, 0))}"
            else:
                task_type = "regression"

        metadata = {
            "rows": len(df), 
            "columns": columns,
            "numerical_features": len(numerical_cols),
            "categorical_features": len(actual_categorical),
            "text_features": len(text_features),
            "missing_values": missing_values,
            "imbalance_ratio": imbalance_ratio,
            "dataset_type": "tabular",
            "detected_task": task_type,
            "feature_names": {
                "numerical": numerical_cols,
                "categorical": actual_categorical,
                "text": text_features,
                "target": target_column
            }
        }

        # Time-series detection
        ts_info = detect_time_series(df, target_column=target_column)
        if ts_info:
            metadata["is_time_series"] = True
            metadata["time_column"] = ts_info["time_column"]
            metadata["group_columns"] = ts_info["group_columns"]
            metadata["dataset_type"] = "time_series"
            logger.info(f"Time-series detected: time_col={ts_info['time_column']}, groups={ts_info['group_columns']}")
        else:
            metadata["is_time_series"] = False
        
        return metadata
