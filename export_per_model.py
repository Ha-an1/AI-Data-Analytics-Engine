"""Export per-model CSVs from analytics_engine.db for Power BI."""
import sqlite3
import pandas as pd
import os

def export():
    conn = sqlite3.connect("analytics_engine.db")
    os.makedirs("exports", exist_ok=True)

    # Get all model names
    models = pd.read_sql("SELECT DISTINCT model_name FROM model_metrics", conn)["model_name"].tolist()
    print(f"Found {len(models)} models: {models}")

    for model in models:
        # Metrics
        metrics = pd.read_sql("SELECT * FROM model_metrics WHERE model_name=?", conn, params=[model])
        metrics.to_csv(f"exports/metrics_{model}.csv", index=False)

        # Feature importance
        metric_ids = metrics["id"].tolist()
        if metric_ids:
            placeholders = ",".join("?" for _ in metric_ids)
            fi = pd.read_sql(f"SELECT * FROM feature_importance WHERE metric_id IN ({placeholders})", conn, params=metric_ids)
            fi.to_csv(f"exports/features_{model}.csv", index=False)

    # Also export shared tables
    for table in ["dataset_summary", "pipeline_metadata", "model_runs"]:
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            df.to_csv(f"exports/{table}.csv", index=False)
        except Exception:
            pass

    conn.close()
    print(f"Exported to exports/ folder: {os.listdir('exports')}")

if __name__ == "__main__":
    export()
