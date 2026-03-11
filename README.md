#  AI Data Analytics Engine

An **autonomous data science pipeline** that automatically analyzes datasets, selects ML models, trains them, evaluates performance, and intelligently retrains underperforming models — all with minimal human intervention.

Upload any dataset, describe your goal in plain English, and the system handles everything from task detection to model deployment.

---

##  Project Summary

The AI Data Analytics Engine is a **production-grade autonomous ML pipeline** designed to:

1. **Accept any tabular dataset** (CSV/JSON) and a natural language problem statement
2. **Automatically detect** task type (classification, regression, time-series forecasting, clustering, anomaly detection)
3. **Identify data leakage**, drop leaky columns, and select the optimal target variable
4. **Determine the best evaluation metric** per problem (e.g., F1 for classification, RMSE for regression)
5. **Train multiple candidate models** and select the best one on a validation set
6. **Intelligently retrain** models that fail evaluation thresholds using a 3-strategy improvement loop
7. **Detect time-series datasets** and automatically generate temporal features (lag, rolling, calendar)
8. **Store all results** in a SQLite database for dashboard visualization via Power BI

---

##  Key Features

###  LLM-Powered Planning (Google Gemini)
- Analyzes dataset structure and user problem statement
- Determines task type, target column, primary metric, and optimization goal
- Detects and removes data leakage columns automatically

###  Multi-Model Training
- Trains multiple candidate models simultaneously
- Supports: Logistic Regression, Linear Regression, Random Forest, XGBoost, LightGBM, Gradient Boosting, KMeans, DBSCAN, Isolation Forest
- Selects the best model based on the LLM-determined primary metric

###  Intelligent Retraining Controller
When a model fails evaluation thresholds, the system automatically runs a **3-strategy improvement loop**:

| Attempt | Strategy | Description |
|---------|----------|-------------|
| 1 | **Hyperparameter Tuning** | `RandomizedSearchCV` with predefined param grids |
| 2 | **Alternative Model Exploration** | Tries untested model families (LightGBM, GradientBoosting, etc.) |
| 3 | **Data-Level Improvements** | SMOTE for imbalance, remove highly correlated features |

Each attempt is logged to the `model_runs` database table. Max 3 attempts to prevent infinite loops.

###  Time-Series Feature Engineering
Automatically detects time-series datasets and generates:
- **Lag features**: `lag_1`, `lag_7`, `lag_14`, `lag_30` (per entity group)
- **Rolling statistics**: `rolling_mean_7`, `rolling_std_7`, `rolling_mean_30`, `rolling_std_30`
- **Calendar features**: `day_of_week`, `month`, `week_of_year`, `quarter`, `is_weekend`
- Uses **temporal (chronological) splitting** instead of random splits

###  Three-Way Data Split
- **Training (70%)** — used to train models
- **Validation (15%)** — used for model selection and hyperparameter tuning
- **Evaluation (15%)** — held out completely; used to simulate real-world inference

###  Inference Modes
- **Eval Inference** — Run the model on the held-out eval set with auto-retrain toggle
- **Manual Input** — Predict on a single row with comma-separated values
- **Batch Upload** — Upload a CSV and get predictions for all rows
- **Experiment History** — View all retraining attempts and their results

---

##  System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     STREAMLIT FRONTEND                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │  Pipeline    │  │   Results    │  │     Inference &     │ │
│  │  Runner Tab  │  │  Dashboard   │  │   Eval/Retrain Tab  │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘ │
└─────────┼─────────────────┼─────────────────────┼────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌──────────────────────────────────────────────────────────────┐
│                   CORE ORCHESTRATOR                          │
│  Load → Analyze → Plan → Feature Eng → Split → Train → Eval │
└──────────────────────┬───────────────────────────────────────┘
                       │
          ┌────────────┼────────────────────────┐
          ▼            ▼                        ▼
┌──────────────┐ ┌───────────────┐  ┌────────────────────────┐
│   Backend    │ │  LLM Planner  │  │  Retraining Controller │
│  Modules     │ │  (Gemini API) │  │  (3-Strategy Loop)     │
│              │ └───────────────┘  └────────────────────────┘
│ • Analyzer   │
│ • Builder    │         ┌──────────────────────┐
│ • Evaluator  │         │    SQLite Database   │
│ • Explainer  │────────▶│  analytics_engine.db │
│ • Features   │         └──────────────────────┘
│ • Models     │                    │
└──────────────┘                    ▼
                            ┌──────────────┐
                            │   Power BI   │
                            │  (via ODBC)  │
                            └──────────────┘
```

---

##  Project Structure

```
AIDataAnalyticsEngine/
│
├── frontend/
│   └── app.py                          # Streamlit UI (3 tabs: Runner, Results, Inference)
│
├── core/
│   └── orchestrator.py                 # Main pipeline orchestrator (load→train→eval→retrain)
│
├── backend/
│   ├── analyzer/
│   │   └── dataset_analyzer.py         # Dataset profiling, TS detection, metadata generation
│   │
│   ├── planner/
│   │   ├── pipeline_planner.py         # Rule-based pipeline step selection
│   │   └── gemini_planner.py           # LLM-powered task/target/metric detection (Gemini API)
│   │
│   ├── pipeline/
│   │   └── pipeline_builder.py         # Builds sklearn Pipelines from plan specifications
│   │
│   ├── models/
│   │   └── model_factory.py            # Model instantiation, param grids, alternative model lookup
│   │
│   ├── evaluation/
│   │   └── evaluator.py                # Metric computation (accuracy, F1, RMSE, R², ROC AUC)
│   │
│   ├── explainability/
│   │   └── explainer.py                # Feature importance extraction (SHAP/permutation)
│   │
│   ├── features/
│   │   └── time_series_features.py     # TS detection, lag/rolling/calendar feature generation
│   │
│   ├── retraining/
│   │   └── retrain_controller.py       # 3-strategy intelligent retraining loop
│   │
│   └── database/
│       ├── models.py                   # SQLAlchemy ORM models (tables schema)
│       └── db_manager.py               # Database session management
│
├── export_per_model.py                 # Utility: export per-model CSVs for Power BI
├── requirements.txt                    # Python dependencies
├── .env                                # API keys and database configuration
└── README.md                           # This file
```

**Runtime directories (auto-created):**
- `data/` — Uploaded datasets
- `trained_models/` — Saved `.joblib` model files
- `eval_data/` — Persisted train/val/eval splits and run metadata
- `exports/` — Per-model CSV exports for Power BI

---

##  Technologies Used

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.10+ | Core language |
| **Frontend** | Streamlit | Interactive web UI |
| **LLM** | Google Gemini API (2.0/2.5 Flash) | Task detection, metric selection, leakage detection |
| **ML Framework** | scikit-learn | Pipelines, preprocessing, model training, cross-validation |
| **Gradient Boosting** | XGBoost, LightGBM | High-performance tree-based models |
| **Imbalanced Data** | imbalanced-learn (SMOTE) | Class imbalance handling during retraining |
| **Explainability** | SHAP | Feature importance extraction |
| **Database** | SQLite + SQLAlchemy ORM | Structured result storage |
| **Dashboarding** | Power BI (external) | Connect via ODBC or CSV exports |
| **Data Processing** | Pandas, NumPy | Data manipulation and analysis |

---

##  Setup & Installation

### Prerequisites
- **Python 3.10 or higher**
- **pip** (Python package manager)
- **Google Gemini API key** ([Get one here](https://aistudio.google.com/app/apikey))

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/AIDataAnalyticsEngine.git
cd AIDataAnalyticsEngine
```

### Step 2: Create a Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install pandas numpy scikit-learn streamlit sqlalchemy shap xgboost lightgbm imbalanced-learn python-dotenv requests joblib
```

Or if `requirements.txt` is up to date:
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Google Gemini API Key (required for LLM-powered task detection)
GEMINI_API_KEY=your_gemini_api_key_here

# Gemini Model (default: gemini-2.0-flash)
GEMINI_MODEL=gemini-2.0-flash

# Database URL (default: local SQLite)
DATABASE_URL=sqlite:///analytics_engine.db
```

> **Note:** Replace `your_gemini_api_key_here` with your actual Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Step 5: Initialize the Database

The database is automatically created when you first run the app. If you need to reset it:

```bash
# Delete existing database (Windows)
del analytics_engine.db

# Delete existing database (macOS/Linux)
rm analytics_engine.db
```

---

##  Environment Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` |  Yes | — | Google Gemini API key for LLM planning |
| `GEMINI_MODEL` | No | `gemini-2.0-flash` | Gemini model variant to use |
| `DATABASE_URL` | No | `sqlite:///analytics_engine.db` | Database connection string (supports PostgreSQL) |

---

##  Running the Application

### Start the Streamlit App

```bash
# Make sure your virtual environment is active
.\venv\Scripts\activate   # Windows
source venv/bin/activate   # macOS/Linux

# Run the app
streamlit run frontend/app.py
```

The app will open at `http://localhost:8501`.

---

##  Usage Guide

### 1. Pipeline Runner Tab

1. **Upload a dataset** (CSV or JSON)
2. **Enter a problem statement** in plain English, e.g.:
   - *"Predict whether a student will get placed based on their academic scores"*
   - *"Forecast daily sales across stores using historical data"*
   - *"Detect anomalies in transaction data"*
3. **Choose mode:**
   - ** LLM Auto-Detect** — Gemini analyzes the dataset and determines task, target, and metrics
   - **Manual** — Manually select task type and target column
4. Click ** Run Pipeline**
5. The system will:
   - Detect leaky columns and drop them
   - Generate time-series features (if applicable)
   - Split data (70/15/15 — temporal for TS, random otherwise)
   - Train multiple models and select the best

### 2. Results Dashboard Tab

- View **metrics** for all trained models (accuracy, F1, RMSE, R², ROC AUC)
- See **feature importance** charts per model
- Compare models side-by-side

### 3. Inference Tab

| Mode | Description |
|------|-------------|
| ** Eval Inference** | Run the model on the held-out 15% eval set. Check "Auto-retrain" to trigger the 3-strategy improvement loop if performance is below threshold |
| ** Manual Input** | Enter comma-separated feature values to predict a single row |
| ** Batch Upload** | Upload a CSV to get predictions for all rows |
| ** Experiment History** | View all retraining attempts, strategies, and scores |

---

##  Power BI Integration

### Option A: Direct ODBC Connection (Recommended)

1. Install the [SQLite ODBC Driver](http://www.ch-werner.de/sqliteodbc/) (64-bit)
2. In Power BI Desktop → **Get Data → ODBC**
3. Connection string:
   ```
   Driver={SQLite3 ODBC Driver};Database=C:\path\to\AIDataAnalyticsEngine\analytics_engine.db;
   ```
4. Select tables and build dashboards

### Option B: CSV Export

```bash
python export_per_model.py
```

This creates per-model CSVs in the `exports/` folder:
- `metrics_<model_name>.csv` — Performance metrics per model
- `features_<model_name>.csv` — Feature importance per model
- `dataset_summary.csv` — Dataset metadata
- `pipeline_metadata.csv` — Pipeline configuration
- `model_runs.csv` — Retraining experiment history

In Power BI → **Get Data → Folder** → Point to the `exports/` directory.

---

##  Database Schema

The SQLite database (`analytics_engine.db`) contains 6 tables:

| Table | Description | Key Columns |
|-------|-------------|-------------|
| `dataset_summary` | Uploaded dataset metadata | rows, columns, metadata_json |
| `pipeline_metadata` | Pipeline config per run | task_objective, preprocessing_steps, selected_models |
| `model_metrics` | Performance metrics per model | accuracy, f1_score, rmse, r2_score, roc_auc, is_best_model |
| `feature_importance` | Feature importance scores | feature_name, importance_score (linked to model_metrics) |
| `predictions` | Individual prediction records | predicted_value, prediction_probability |
| `model_runs` | Retraining experiment log | strategy, attempt, validation_score, training_duration, improved |

---

##  How It Works

### End-to-End Pipeline Flow

```
1. User uploads dataset + describes goal
         │
2. Gemini LLM analyzes → task type, target, metric, leakage detection
         │
3. Dataset Analyzer profiles data → detects time-series, feature types
         │
4. Pipeline Planner selects preprocessing steps + candidate models
         │
5. [If time-series] Feature Generator adds lag/rolling/calendar features
         │
6. Data Split: 70% train / 15% validation / 15% eval
   (temporal for TS, random for non-TS)
         │
7. Train all candidate models on training set
         │
8. Evaluate on validation set → select best model (by LLM-chosen metric)
         │
9. Save best model + all metrics to database
         │
10. [Inference Tab] Evaluate on held-out eval set
         │
11. [If below threshold + auto-retrain enabled]
    → Retraining Controller (3-strategy loop)
    → Replace model if improved
```

### Retraining Strategy Details

```
Eval Score < Threshold?
    │
    ├─ Attempt 1: Hyperparameter Tuning
    │  └─ RandomizedSearchCV with predefined param grids
    │  └─ 20 iterations, 3-fold CV
    │
    ├─ Attempt 2: Alternative Models
    │  └─ Try model families not yet explored
    │  └─ LightGBM, GradientBoosting, etc.
    │
    └─ Attempt 3: Data Improvements
       └─ SMOTE for class imbalance
       └─ Remove features with >0.95 correlation
       └─ Retrain on improved data
```

---

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m "Add your feature"`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request
