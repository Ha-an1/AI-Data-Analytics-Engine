import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Default Configuration ──
DEFAULT_LAGS = [1, 7, 14, 30]
DEFAULT_ROLLING_WINDOWS = [7, 30]


class TimeSeriesFeatureGenerator:
    """
    Automatically generates temporal features for time-series regression:
      - Lag features (per entity group if group columns exist)
      - Rolling window statistics (mean, std)
      - Calendar features from the datetime column
    """

    def __init__(self, time_column, target_column, group_columns=None,
                 lags=None, rolling_windows=None):
        self.time_column = time_column
        self.target_column = target_column
        self.group_columns = group_columns or []
        self.lags = lags or DEFAULT_LAGS
        self.rolling_windows = rolling_windows or DEFAULT_ROLLING_WINDOWS

    def generate(self, df):
        """
        Full pipeline: sort → calendar features → lag features → rolling stats → drop NaN rows.
        Returns the transformed DataFrame.
        """
        df = df.copy()
        logger.info(f"[TS Features] Starting generation: time={self.time_column}, "
                     f"target={self.target_column}, groups={self.group_columns}")

        # 1. Parse and sort by time
        df = self._parse_and_sort(df)

        # 2. Calendar features from the datetime column
        df = self._add_calendar_features(df)

        # 3. Lag features
        df = self._add_lag_features(df)

        # 4. Rolling statistics
        df = self._add_rolling_features(df)

        # 5. Drop the original datetime column (models can't handle it)
        if self.time_column in df.columns:
            df = df.drop(columns=[self.time_column])

        # 6. Drop rows with NaN introduced by lags/rolling
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)
        if rows_before != rows_after:
            logger.info(f"[TS Features] Dropped {rows_before - rows_after} rows with insufficient history "
                         f"({rows_after} rows remaining)")

        logger.info(f"[TS Features] Generation complete. Final shape: {df.shape}")
        return df

    def _parse_and_sort(self, df):
        """Parse datetime column and sort chronologically."""
        try:
            df[self.time_column] = pd.to_datetime(df[self.time_column])
        except Exception as e:
            logger.warning(f"[TS Features] Could not parse '{self.time_column}' as datetime: {e}")
            return df

        if self.group_columns:
            sort_cols = self.group_columns + [self.time_column]
        else:
            sort_cols = [self.time_column]

        df = df.sort_values(sort_cols).reset_index(drop=True)
        logger.info(f"[TS Features] Sorted by {sort_cols}")
        return df

    def _add_calendar_features(self, df):
        """Extract calendar features from the datetime column."""
        if self.time_column not in df.columns:
            return df

        dt = df[self.time_column]
        if not pd.api.types.is_datetime64_any_dtype(dt):
            return df

        df["day_of_week"] = dt.dt.dayofweek
        df["day_of_month"] = dt.dt.day
        df["month"] = dt.dt.month
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        df["quarter"] = dt.dt.quarter
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

        logger.info("[TS Features] Added 6 calendar features")
        return df

    def _add_lag_features(self, df):
        """Generate lag features, per group if group columns exist."""
        target = self.target_column
        if target not in df.columns:
            logger.warning(f"[TS Features] Target column '{target}' not found, skipping lags")
            return df

        count = 0
        for lag in self.lags:
            col_name = f"lag_{lag}"
            if self.group_columns:
                df[col_name] = df.groupby(self.group_columns)[target].shift(lag)
            else:
                df[col_name] = df[target].shift(lag)
            count += 1

        logger.info(f"[TS Features] Added {count} lag features: {['lag_' + str(l) for l in self.lags]}")
        return df

    def _add_rolling_features(self, df):
        """Generate rolling mean and std, per group if group columns exist."""
        target = self.target_column
        if target not in df.columns:
            return df

        count = 0
        for window in self.rolling_windows:
            mean_col = f"rolling_mean_{window}"
            std_col = f"rolling_std_{window}"

            if self.group_columns:
                grouped = df.groupby(self.group_columns)[target]
                # shift(1) to avoid data leakage (don't include current row)
                df[mean_col] = grouped.transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                df[std_col] = grouped.transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
            else:
                shifted = df[target].shift(1)
                df[mean_col] = shifted.rolling(window, min_periods=1).mean()
                df[std_col] = shifted.rolling(window, min_periods=1).std()
            count += 2

        logger.info(f"[TS Features] Added {count} rolling features for windows {self.rolling_windows}")
        return df


def detect_time_series(df, target_column=None):
    """
    Detect whether a DataFrame is a time-series dataset.
    
    Returns:
        dict or None: Detection result with time_column and group_columns,
                      or None if not a time-series dataset.
    """
    # 1. Find datetime columns (already parsed or parseable strings)
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # 2. Check string columns that look like dates
    if not datetime_cols:
        date_keywords = ["date", "timestamp", "time", "datetime", "period", "day"]
        for col in df.columns:
            if any(kw in col.lower() for kw in date_keywords):
                try:
                    parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                    if parsed.notna().sum() / len(df) > 0.8:
                        datetime_cols.append(col)
                except Exception:
                    continue

    if not datetime_cols:
        return None

    time_col = datetime_cols[0]  # Use the first detected datetime column

    # 3. Check for a continuous numerical target that changes over time
    if target_column and target_column in df.columns:
        if df[target_column].dtype not in ["object", "category", "bool"]:
            nunique = df[target_column].nunique()
            if nunique < 10:
                return None  # Likely classification, not time-series regression
        else:
            return None

    # 4. Detect entity/group columns (low-cardinality categorical: IDs, store names)
    group_cols = []
    id_keywords = ["id", "store", "product", "region", "category", "entity", "group", "location", "site"]
    for col in df.columns:
        if col == time_col or col == target_column:
            continue
        if any(kw in col.lower() for kw in id_keywords):
            if df[col].nunique() < 500:  # Reasonable number of entities
                group_cols.append(col)

    logger.info(f"[TS Detection] Detected time-series: time_col={time_col}, groups={group_cols}")
    return {
        "is_time_series": True,
        "time_column": time_col,
        "group_columns": group_cols
    }
