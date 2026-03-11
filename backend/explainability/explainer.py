import shap
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ModelExplainer:
    def __init__(self):
        pass
        
    def extract_feature_names(self, preprocessor, raw_features):
        """Attempts to extract feature names from a scikit-learn ColumnTransformer."""
        try:
            if preprocessor == 'passthrough':
                return raw_features
                
            feature_names = []
            for name, transformer, columns in preprocessor.transformers_:
                if name == 'remainder' and transformer == 'drop':
                    continue
                if hasattr(transformer, 'get_feature_names_out'):
                    try:
                        names = transformer.get_feature_names_out(columns)
                        feature_names.extend(names)
                    except Exception:
                        feature_names.extend([f"{name}_{c}" for c in columns])
                else:
                    feature_names.extend([f"{name}_{c}" for c in columns])
            return feature_names
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")
            return [f"feature_{i}" for i in range(1000)] # fallback

    def explain(self, pipeline, X_background, task_type="classification"):
        """
        Generates feature importance scores using native attributes or SHAP.
        Returns a dict: { "feature_name": importance_score }
        """
        importance_dict = {}
        try:
            # Safely extract and transform the data
            if 'preprocessor' in pipeline.named_steps and pipeline.named_steps['preprocessor'] != 'passthrough':
                preprocessor = pipeline.named_steps['preprocessor']
                X_transformed = preprocessor.transform(X_background)
                if hasattr(X_transformed, "toarray"):
                    X_transformed = X_transformed.toarray()
                feature_names = self.extract_feature_names(preprocessor, X_background.columns.tolist())
            else:
                X_transformed = X_background.values
                feature_names = X_background.columns.tolist()

            model = pipeline.named_steps['model']
            
            # 1. Try native feature importances (Fastest for Trees)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Align lengths safely
                matched_length = min(len(importances), len(feature_names))
                for i in range(matched_length):
                    importance_dict[feature_names[i]] = float(importances[i])
                if len(importances) > len(feature_names):
                    for i in range(matched_length, len(importances)):
                         importance_dict[f"feature_{i}"] = float(importances[i])
                        
            # 2. Try coefficients (For Linear Models)
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0]) if hasattr(model.coef_[0], '__iter__') else np.abs(model.coef_)
                matched_length = min(len(importances), len(feature_names))
                for i in range(matched_length):
                    importance_dict[feature_names[i]] = float(importances[i])
                        
            # 3. Fallback to SHAP (e.g. for SVMs or complex ensembles)
            else:
                logger.info("Falling back to SHAP explainer.")
                sample_size = min(100, X_transformed.shape[0])
                X_sample = X_transformed[:sample_size]
                
                try:
                    explainer = shap.Explainer(model, X_sample)
                    shap_values = explainer(X_sample)
                    
                    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                    if len(mean_abs_shap.shape) > 1:
                        mean_abs_shap = mean_abs_shap.mean(axis=1)
                        
                    for i, val in enumerate(mean_abs_shap):
                        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                        importance_dict[name] = float(val)
                except Exception as e:
                    logger.warning(f"SHAP explainer failed: {e}")

            # Top 20 features to save DB space and UI clutter
            sorted_importances = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)[:20])
            return sorted_importances

        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            return {}
