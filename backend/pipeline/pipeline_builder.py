import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)

class PipelineBuilder:
    def __init__(self):
        self.factory = ModelFactory()

    def build_pipelines(self, plan, metadata):
        """
        Builds a list of scikit-learn Pipeline objects for training.
        Returns a dict: {"model_name": Pipeline}
        """
        task_type = plan.get("task_type", "classification")
        steps = plan.get("pipeline_steps", [])
        details = plan.get("preprocessing_details", {})
        candidate_models = plan.get("models", [])
        
        feature_names = metadata.get("feature_names", {})
        num_cols = feature_names.get("numerical", [])
        cat_cols = feature_names.get("categorical", [])
        text_cols = feature_names.get("text", [])
        
        transformers = []
        
        # Build Numerical Pipeline
        if ("missing_value_imputation" in steps or "feature_scaling" in steps) and num_cols:
            num_steps = []
            if "missing_value_imputation" in steps:
                num_steps.append(('imputer', SimpleImputer(strategy='mean')))
            if "feature_scaling" in steps:
                num_steps.append(('scaler', StandardScaler()))
            if num_steps:
                transformers.append(('num', Pipeline(num_steps), num_cols))
                
        # Build Categorical Pipeline
        if ("categorical_encoding" in steps or "missing_value_imputation" in steps) and cat_cols:
            cat_steps = []
            if "missing_value_imputation" in steps:
                cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            if "categorical_encoding" in steps:
                # sparse_output=False since some downstream models/SHAP dislike sparse arrays
                cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
            if cat_steps:
                transformers.append(('cat', Pipeline(cat_steps), cat_cols))
                
        # Build Text Pipeline
        if "tfidf_vectorization" in steps and text_cols:
            # We assume a single text column for MVP NLP analysis
            transformers.append(('text', TfidfVectorizer(max_features=5000), text_cols[0]))
            
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop') if transformers else 'passthrough'

        pipelines = {}
        for model_name in candidate_models:
            try:
                model_instance = self.factory.get_model(model_name, task_type)
                if preprocessor == 'passthrough':
                    pipe = Pipeline(steps=[('model', model_instance)])
                else:
                    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_instance)])
                pipelines[model_name] = pipe
            except Exception as e:
                logger.error(f"Failed to build pipeline for model {model_name}: {e}")
            
        return pipelines
