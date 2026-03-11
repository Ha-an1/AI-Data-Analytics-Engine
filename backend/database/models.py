from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class DatasetSummary(Base):
    __tablename__ = 'dataset_summary'
    
    id = Column(Integer, primary_key=True)
    dataset_name = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    rows = Column(Integer)
    columns = Column(Integer)
    metadata_json = Column(JSON)

class PipelineMetadata(Base):
    __tablename__ = 'pipeline_metadata'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('dataset_summary.id'))
    task_objective = Column(String)
    preprocessing_steps = Column(JSON)
    selected_models = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True)
    pipeline_id = Column(Integer, ForeignKey('pipeline_metadata.id'))
    model_name = Column(String)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    rmse = Column(Float)
    mae = Column(Float)
    r2_score = Column(Float)
    silhouette_score = Column(Float)
    training_time_seconds = Column(Float)
    is_best_model = Column(Integer) # SQLite representation of boolean
    roc_auc = Column(Float)

class FeatureImportance(Base):
    __tablename__ = 'feature_importance'
    
    id = Column(Integer, primary_key=True)
    metric_id = Column(Integer, ForeignKey('model_metrics.id'))
    feature_name = Column(String)
    importance_score = Column(Float)

class Predictions(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    metric_id = Column(Integer, ForeignKey('model_metrics.id'))
    record_id = Column(String) 
    predicted_value = Column(String) # Cast logic based on objective
    prediction_probability = Column(Float)

class ModelRuns(Base):
    __tablename__ = 'model_runs'
    
    id = Column(Integer, primary_key=True)
    pipeline_id = Column(Integer, ForeignKey('pipeline_metadata.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String)
    strategy = Column(String)  # hyperparameter_tuning, alternative_model, data_improvement
    attempt = Column(Integer)
    parameters = Column(JSON)
    validation_score = Column(Float)
    evaluation_score = Column(Float)
    training_duration = Column(Float)
    improved = Column(Integer, default=0)  # SQLite boolean
