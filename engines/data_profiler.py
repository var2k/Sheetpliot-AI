"""
SheetPilot AI - Data Profiler Engine
Performs comprehensive data health inspection, anomaly detection, type inference,
missingness profiling, outlier detection, and health score calculation.
"""

import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class DataProfilerEngine:
    """Calculates granular data health diagnostics and statistical summaries."""

    EMAIL_REGEX = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
    PHONE_REGEX = re.compile(r"^\+?[\d\s\-\(\)\.]{7,25}$")

    @classmethod
    def profile(cls, df: pd.DataFrame, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Runs full end-to-end dataset profiling."""
        if df is None or df.empty:
            return cls._empty_profile()

        work_df = df if sample_size is None or len(df) <= sample_size else df.sample(sample_size, random_state=42)

        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols if total_cols > 0 else 0

        # Null metrics
        null_counts = df.isna().sum()
        total_null_cells = int(null_counts.sum())
        null_percentage = round((total_null_cells / total_cells * 100), 2) if total_cells > 0 else 0.0

        # Duplicate metrics
        duplicate_rows_count = int(df.duplicated().sum())
        duplicate_percentage = round((duplicate_rows_count / total_rows * 100), 2) if total_rows > 0 else 0.0

        # Column-level profiles
        columns_profile = {}
        all_warnings: List[Dict[str, str]] = []
        outlier_total_count = 0

        for col in df.columns:
            series = df[col]
            col_prof = cls._profile_column(series, total_rows)
            columns_profile[str(col)] = col_prof

            if col_prof.get("outliers_count", 0) > 0:
                outlier_total_count += col_prof["outliers_count"]

            for warn in col_prof.get("warnings", []):
                all_warnings.append({"column": str(col), "warning": warn})

        # Calculate Overall Health Score (0-100)
        health_score, health_grade = cls._calculate_health_score(
            total_rows=total_rows,
            null_percentage=null_percentage,
            duplicate_percentage=duplicate_percentage,
            warning_count=len(all_warnings),
            columns_count=total_cols
        )

        return {
            "overview": {
                "total_rows": total_rows,
                "total_columns": total_cols,
                "total_cells": total_cells,
                "total_null_cells": total_null_cells,
                "null_percentage": null_percentage,
                "duplicate_rows": duplicate_rows_count,
                "duplicate_percentage": duplicate_percentage,
                "total_outliers": outlier_total_count,
                "health_score": health_score,
                "health_grade": health_grade,
                "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            },
            "columns": columns_profile,
            "warnings": all_warnings,
            "recommendations": cls._generate_recommendations(
                duplicate_rows=duplicate_rows_count,
                null_percentage=null_percentage,
                warnings=all_warnings,
                columns_profile=columns_profile
            )
        }

    @classmethod
    def _profile_column(cls, s: pd.Series, total_rows: int) -> Dict[str, Any]:
        """Profiles a single column."""
        null_count = int(s.isna().sum())
        null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        non_null_series = s.dropna()
        unique_count = int(non_null_series.nunique())
        unique_pct = round((unique_count / len(non_null_series) * 100), 2) if len(non_null_series) > 0 else 0.0

        inferred_type = cls._infer_semantic_type(non_null_series)
        warnings: List[str] = []

        if null_pct > 25.0:
            warnings.append(f"High missing rate ({null_pct}%)")

        col_dict: Dict[str, Any] = {
            "name": str(s.name),
            "dtype": str(s.dtype),
            "inferred_type": inferred_type,
            "null_count": null_count,
            "null_percentage": null_pct,
            "unique_count": unique_count,
            "unique_percentage": unique_pct,
            "sample_values": [str(x) for x in non_null_series.iloc[:4].tolist()],
            "stats": {},
            "outliers_count": 0,
            "warnings": warnings,
        }

        # Check for whitespace issues if string
        if inferred_type in ["text", "categorical", "email", "phone"] and len(non_null_series) > 0:
            str_vals = non_null_series.astype(str)
            has_whitespace = str_vals.str.strip().ne(str_vals).any()
            if has_whitespace:
                warnings.append("Contains un-trimmed leading/trailing whitespace")

        # Type-specific stats
        if inferred_type in ["integer", "float"]:
            numeric_s = pd.to_numeric(non_null_series, errors="coerce").dropna()
            if not numeric_s.empty:
                col_dict["stats"] = {
                    "min": round(float(numeric_s.min()), 2),
                    "max": round(float(numeric_s.max()), 2),
                    "mean": round(float(numeric_s.mean()), 2),
                    "median": round(float(numeric_s.median()), 2),
                    "std": round(float(numeric_s.std()), 2) if len(numeric_s) > 1 else 0.0,
                }
                # IQR outlier calculation
                q1 = numeric_s.quantile(0.25)
                q3 = numeric_s.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outliers = numeric_s[(numeric_s < q1 - 1.5 * iqr) | (numeric_s > q3 + 1.5 * iqr)]
                    col_dict["outliers_count"] = int(len(outliers))
                    if len(outliers) > 0:
                        warnings.append(f"{len(outliers)} statistical outliers detected (IQR)")

        elif inferred_type == "datetime":
            dt_s = pd.to_datetime(non_null_series, errors="coerce").dropna()
            if not dt_s.empty:
                col_dict["stats"] = {
                    "min_date": str(dt_s.min()),
                    "max_date": str(dt_s.max()),
                }

        elif inferred_type in ["categorical", "text"]:
            val_counts = non_null_series.value_counts()
            top_val = str(val_counts.index[0]) if not val_counts.empty else "N/A"
            top_freq = int(val_counts.iloc[0]) if not val_counts.empty else 0
            col_dict["stats"] = {
                "top_value": top_val,
                "top_frequency": top_freq,
                "top_percentage": round(top_freq / len(non_null_series) * 100, 1) if len(non_null_series) > 0 else 0,
            }

        return col_dict

    @classmethod
    def _infer_semantic_type(cls, non_null_series: pd.Series) -> str:
        """Infers high-level semantic type for data validation and cleaning."""
        if non_null_series.empty:
            return "empty"

        sample = non_null_series.iloc[:100]

        # Check numeric
        if pd.api.types.is_numeric_dtype(non_null_series):
            if pd.api.types.is_integer_dtype(non_null_series):
                return "integer"
            return "float"

        # Check boolean
        if pd.api.types.is_bool_dtype(non_null_series):
            return "boolean"

        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(non_null_series):
            return "datetime"

        # Check string representations
        sample_str = [str(x).strip() for x in sample]

        # Check boolean strings
        bool_set = {"true", "false", "yes", "no", "y", "n", "0", "1"}
        if all(x.lower() in bool_set for x in sample_str if x):
            return "boolean"

        # Check email pattern
        email_matches = sum(bool(cls.EMAIL_REGEX.match(x)) for x in sample_str)
        if len(sample_str) > 0 and (email_matches / len(sample_str)) >= 0.7:
            return "email"

        # Check phone pattern
        phone_matches = sum(bool(cls.PHONE_REGEX.match(x)) for x in sample_str)
        if len(sample_str) > 0 and (phone_matches / len(sample_str)) >= 0.7:
            return "phone"

        # Check if parseable as datetime
        parsed_dates = pd.to_datetime(sample, errors="coerce")
        if parsed_dates.notna().sum() / len(sample) >= 0.8:
            return "datetime"

        # Check if parseable as numeric
        parsed_nums = pd.to_numeric(sample_str, errors="coerce")
        if pd.Series(parsed_nums).notna().sum() / len(sample) >= 0.8:
            return "numeric"

        # Categorical vs Free Text
        unique_ratio = non_null_series.nunique() / len(non_null_series)
        if unique_ratio < 0.2:
            return "categorical"

        return "text"

    @classmethod
    def _calculate_health_score(
        cls,
        total_rows: int,
        null_percentage: float,
        duplicate_percentage: float,
        warning_count: int,
        columns_count: int
    ) -> tuple[int, str]:
        """Calculates a normalized 0-100 score reflecting data quality."""
        score = 100.0

        # Missing values penalty (up to 35 points)
        score -= min(35.0, null_percentage * 1.2)

        # Duplicates penalty (up to 25 points)
        score -= min(25.0, duplicate_percentage * 2.0)

        # Warnings penalty (up to 25 points)
        if columns_count > 0:
            warning_ratio = warning_count / max(1, columns_count)
            score -= min(25.0, warning_ratio * 10.0)

        final_score = max(0, min(100, int(round(score))))

        if final_score >= 90:
            grade = "A+"
        elif final_score >= 80:
            grade = "A"
        elif final_score >= 70:
            grade = "B"
        elif final_score >= 60:
            grade = "C"
        elif final_score >= 50:
            grade = "D"
        else:
            grade = "F"

        return final_score, grade

    @classmethod
    def _generate_recommendations(
        cls,
        duplicate_rows: int,
        null_percentage: float,
        warnings: List[Dict[str, str]],
        columns_profile: Dict[str, Any]
    ) -> List[str]:
        """Generates human-readable, actionable data cleaning suggestions."""
        recs = []
        if duplicate_rows > 0:
            recs.append(f"Remove {duplicate_rows} duplicate rows to ensure entity uniqueness.")

        for col_name, prof in columns_profile.items():
            for w in prof.get("warnings", []):
                if "whitespace" in w.lower():
                    recs.append(f"Trim extra whitespace from text column '{col_name}'.")
                elif "outliers" in w.lower():
                    recs.append(f"Review or clip {prof.get('outliers_count', 0)} numeric outliers in '{col_name}'.")
                elif "high missing" in w.lower():
                    recs.append(f"Impute or handle missing values ({prof.get('null_percentage')}% missing) in '{col_name}'.")

        if null_percentage > 5.0 and not recs:
            recs.append("Drop or impute columns with excessive null values.")

        if not recs:
            recs.append("Dataset is in excellent condition with high fidelity.")

        return recs[:8]

    @classmethod
    def _empty_profile(cls) -> Dict[str, Any]:
        return {
            "overview": {
                "total_rows": 0,
                "total_columns": 0,
                "total_cells": 0,
                "total_null_cells": 0,
                "null_percentage": 0.0,
                "duplicate_rows": 0,
                "duplicate_percentage": 0.0,
                "total_outliers": 0,
                "health_score": 100,
                "health_grade": "A+",
                "memory_mb": 0.0,
            },
            "columns": {},
            "warnings": [],
            "recommendations": ["Upload or select a dataset to view health diagnostics."]
        }
