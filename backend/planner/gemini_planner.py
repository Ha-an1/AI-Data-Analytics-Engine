import os
import json
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

MAX_RETRIES = 3
BASE_DELAY = 2  # seconds

class GeminiPlanner:
    """Uses Google Gemini to analyze a dataset sample + user problem statement
    and automatically determine the task objective and target column."""

    def analyze_with_llm(self, df_sample_json, column_names, user_problem_statement):
        """
        Sends a dataset sample and problem statement to Gemini.
        Returns: {"task": "classification|regression|...", "target": "column_name", "reasoning": "..."}
        """
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not set in .env file.")
            return None

        prompt = f"""You are an expert data scientist performing a thorough pre-modeling analysis. A user has uploaded a dataset and described what they want to achieve.

USER'S PROBLEM STATEMENT:
\"{user_problem_statement}\"

DATASET COLUMNS:
{json.dumps(column_names)}

DATASET SAMPLE (first 5 rows as JSON):
{df_sample_json}

You must perform the following analysis steps:

STEP 1 - TASK IDENTIFICATION:
Determine the most appropriate ML task type. Choose EXACTLY ONE from: classification, regression, clustering, anomaly_detection, time_series_forecasting, nlp_classification

STEP 2 - TARGET COLUMN SELECTION:
Identify the column the user wants to predict. If the task is unsupervised (clustering or anomaly_detection), set target to null.

STEP 3 - DATA LEAKAGE DETECTION (CRITICAL):
Carefully examine EVERY column and determine if it would cause data leakage if used as a feature. A column causes data leakage if:
- It is a DIRECT CONSEQUENCE of the target variable (e.g., "salary_package" only has values when "placed" = 1, so it leaks the answer)
- It contains information that would NOT be available at prediction time (e.g., "outcome_date" when predicting the outcome)
- It is essentially the same information as the target in a different form (e.g., "pass_fail_status" when predicting "exam_result")
- It is a unique identifier or index column with no predictive value (e.g., "student_id", "row_number")

For EACH column, ask yourself: "Would this column be available BEFORE we know the target value? Or does it only exist BECAUSE of the target?"

If a column's values are predominantly null/zero/empty for one class of the target and populated for the other, that is a strong leakage signal.

STEP 4 - METRIC SELECTION:
Determine the most appropriate PRIMARY evaluation metric for this specific problem. Consider the nature of the problem:
- Classification: accuracy, precision, recall, f1_score, roc_auc
- Regression: rmse, mae, r2_score
- Clustering: silhouette_score
Also determine secondary metrics and whether the primary metric should be maximized or minimized.

STEP 5 - REASONING:
Provide a brief overall reasoning for your choices.

Respond ONLY with valid JSON in this exact format, no markdown, no code fences:
{{"task": "...", "target": "...", "drop_columns": ["col1", "col2"], "drop_reasons": {{"col1": "reason", "col2": "reason"}}, "primary_metric": "f1_score", "secondary_metrics": ["precision", "recall"], "optimization_goal": "maximize", "reasoning": "..."}}"""

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048
            }
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                # Handle 429 rate limit with exponential backoff
                if response.status_code == 429:
                    delay = BASE_DELAY * (2 ** attempt)
                    logger.warning(f"Gemini rate limited (429). Retrying in {delay}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                    continue

                response.raise_for_status()

                result = response.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]

                # Clean any markdown fences if present
                text = text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text.rsplit("```", 1)[0]
                text = text.strip()

                parsed = json.loads(text)
                logger.info(f"Gemini LLM response: {parsed}")
                return parsed

            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY * (2 ** attempt)
                    logger.warning(f"Gemini request failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Gemini API request failed after {MAX_RETRIES} attempts: {e}")
                    return None
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                # Attempt to repair truncated JSON
                logger.warning(f"JSON parse failed, attempting repair: {e}")
                try:
                    repaired = self._repair_json(text)
                    if repaired:
                        return repaired
                except Exception:
                    pass
                logger.error(f"Failed to parse Gemini response even after repair: {e}")
                return None

        logger.error(f"Gemini API exhausted all {MAX_RETRIES} retries.")
        return None

    def _repair_json(self, text):
        """Attempts to repair truncated JSON from Gemini."""
        # Try to find the essential fields even in broken JSON
        import re

        task_match = re.search(r'"task"\s*:\s*"([^"]+)"', text)
        target_match = re.search(r'"target"\s*:\s*"([^"]+)"', text)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', text)

        # Extract drop_columns array
        drop_match = re.search(r'"drop_columns"\s*:\s*\[([^\]]*)\]', text)
        drop_cols = []
        if drop_match:
            drop_cols = [s.strip().strip('"') for s in drop_match.group(1).split(',') if s.strip().strip('"')]

        if task_match:
            # Extract primary_metric
            metric_match = re.search(r'"primary_metric"\s*:\s*"([^"]+)"', text)
            goal_match = re.search(r'"optimization_goal"\s*:\s*"([^"]+)"', text)
            
            result = {
                "task": task_match.group(1),
                "target": target_match.group(1) if target_match else None,
                "drop_columns": drop_cols,
                "drop_reasons": {},
                "primary_metric": metric_match.group(1) if metric_match else None,
                "secondary_metrics": [],
                "optimization_goal": goal_match.group(1) if goal_match else None,
                "reasoning": reasoning_match.group(1).rstrip('"') if reasoning_match else "Repaired from truncated response"
            }
            logger.info(f"Repaired Gemini response: {result}")
            return result
        return None
