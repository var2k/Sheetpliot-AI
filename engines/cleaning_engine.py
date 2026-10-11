"""
SheetPilot AI - Cleaning Engine
Robust, vectorized data cleaning, standardization, type casting,
deduplication, imputation, and outlier treatment functions.
"""

import re
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class CleaningEngine:
    """Executes atomic and composite data cleaning operations on DataFrames."""

    @classmethod
    def trim_whitespace(cls, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, int]:
        """Trims leading/trailing whitespace and collapses multiple internal spaces."""
        df_out = df.copy()
        target_cols = columns or [c for c in df_out.columns if pd.api.types.is_object_dtype(df_out[c]) or pd.api.types.is_string_dtype(df_out[c])]
        cells_affected = 0

        for col in target_cols:
            if col in df_out.columns:
                s = df_out[col].astype(str)
                cleaned = s.apply(lambda x: re.sub(r"\s+", " ", x.strip()) if pd.notna(x) and x != "nan" and x != "None" else x)
                # Count diff
                diff = (s != cleaned) & df_out[col].notna()
                cells_affected += int(diff.sum())
                # Restore actual NaNs
                df_out[col] = df_out[col].where(df_out[col].isna(), cleaned)

        return df_out, cells_affected

    @classmethod
    def standardize_case(
        cls,
        df: pd.DataFrame,
        columns: List[str],
        case_type: str = "title"
    ) -> Tuple[pd.DataFrame, int]:
        """Converts strings to title, lower, upper, or sentence case."""
        df_out = df.copy()
        cells_affected = 0

        for col in columns:
            if col in df_out.columns:
                s = df_out[col].astype(str)
                if case_type == "lower":
                    new_s = s.str.lower()
                elif case_type == "upper":
                    new_s = s.str.upper()
                elif case_type == "title":
                    new_s = s.str.title()
                elif case_type == "sentence":
                    new_s = s.apply(lambda x: x.capitalize() if isinstance(x, str) else x)
                else:
                    new_s = s

                diff = (s != new_s) & df_out[col].notna()
                cells_affected += int(diff.sum())
                df_out[col] = df_out[col].where(df_out[col].isna(), new_s)

        return df_out, cells_affected

    @classmethod
    def clean_phone_numbers(
        cls,
        df: pd.DataFrame,
        columns: List[str],
        phone_format: str = "dashed"  # 'dashed': 123-456-7890, 'e164': +11234567890, 'digits': 1234567890
    ) -> Tuple[pd.DataFrame, int]:
        """Extracts clean numeric sequences and standardizes phone formatting."""
        df_out = df.copy()
        cells_affected = 0

        def _format_phone(val: Any) -> Any:
            if pd.isna(val) or val is None or str(val).strip() in ("", "nan", "None"):
                return np.nan

            digits = re.sub(r"\D", "", str(val))
            if not digits:
                return val

            # Normalize 11-digit starting with 1 to 10-digit
            if len(digits) == 11 and digits.startswith("1"):
                digits = digits[1:]

            if len(digits) == 10:
                if phone_format == "dashed":
                    return f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
                elif phone_format == "e164":
                    return f"+1{digits}"
                elif phone_format == "parentheses":
                    return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
                else:
                    return digits
            return digits

        for col in columns:
            if col in df_out.columns:
                orig = df_out[col].copy()
                df_out[col] = df_out[col].apply(_format_phone)
                diff = (orig.astype(str) != df_out[col].astype(str)) & orig.notna()
                cells_affected += int(diff.sum())

        return df_out, cells_affected

    @classmethod
    def clean_emails(cls, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, int]:
        """Lowercases emails and trims spaces."""
        df_out = df.copy()
        cells_affected = 0

        for col in columns:
            if col in df_out.columns:
                s = df_out[col].astype(str).str.strip().str.lower()
                diff = (df_out[col].astype(str) != s) & df_out[col].notna()
                cells_affected += int(diff.sum())
                df_out[col] = df_out[col].where(df_out[col].isna(), s)

        return df_out, cells_affected

    @classmethod
    def standardize_dates(
        cls,
        df: pd.DataFrame,
        columns: List[str],
        target_format: str = "%Y-%m-%d"
    ) -> Tuple[pd.DataFrame, int]:
        """Parses heterogeneous date formats and formats to target string representation."""
        df_out = df.copy()
        cells_affected = 0

        for col in columns:
            if col in df_out.columns:
                orig = df_out[col].copy()
                parsed_dates = pd.to_datetime(df_out[col], errors="coerce")
                formatted = parsed_dates.dt.strftime(target_format)
                # Keep original if unparseable, otherwise use formatted
                df_out[col] = formatted.where(parsed_dates.notna(), orig)
                diff = (orig.astype(str) != df_out[col].astype(str)) & orig.notna()
                cells_affected += int(diff.sum())

        return df_out, cells_affected

    @classmethod
    def remove_duplicates(
        cls,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first"
    ) -> Tuple[pd.DataFrame, int]:
        """Removes duplicate rows based on all or specified columns."""
        orig_count = len(df)
        df_out = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        dropped_count = orig_count - len(df_out)
        return df_out, dropped_count

    @classmethod
    def fill_missing(
        cls,
        df: pd.DataFrame,
        column: str,
        method: str = "value",
        custom_val: Any = None
    ) -> Tuple[pd.DataFrame, int]:
        """Fills null values in a column using a constant, mean, median, mode, or ffill."""
        df_out = df.copy()
        if column not in df_out.columns:
            return df_out, 0

        null_count = int(df_out[column].isna().sum())
        if null_count == 0:
            return df_out, 0

        if method == "mean":
            val = pd.to_numeric(df_out[column], errors="coerce").mean()
            df_out[column] = df_out[column].fillna(round(val, 2))
        elif method == "median":
            val = pd.to_numeric(df_out[column], errors="coerce").median()
            df_out[column] = df_out[column].fillna(round(val, 2))
        elif method == "mode":
            mode_vals = df_out[column].mode()
            val = mode_vals.iloc[0] if not mode_vals.empty else ""
            df_out[column] = df_out[column].fillna(val)
        elif method == "ffill":
            df_out[column] = df_out[column].ffill()
        elif method == "bfill":
            df_out[column] = df_out[column].bfill()
        else:
            # Custom value
            df_out[column] = df_out[column].fillna(custom_val if custom_val is not None else "")

        return df_out, null_count

    @classmethod
    def drop_empty_rows(cls, df: pd.DataFrame, threshold: float = 0.9) -> Tuple[pd.DataFrame, int]:
        """Drops rows where missing cell percentage exceeds threshold or all are null."""
        orig_count = len(df)
        min_non_null = int(len(df.columns) * (1.0 - threshold))
        df_out = df.dropna(thresh=max(1, min_non_null)).reset_index(drop=True)
        dropped = orig_count - len(df_out)
        return df_out, dropped

    @classmethod
    def drop_columns(cls, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, int]:
        """Drops specified columns."""
        existing = [c for c in columns if c in df.columns]
        df_out = df.drop(columns=existing)
        return df_out, len(existing)

    @classmethod
    def rename_columns(cls, df: pd.DataFrame, rename_map: Dict[str, str]) -> Tuple[pd.DataFrame, int]:
        """Renames columns according to mapping."""
        valid_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df_out = df.rename(columns=valid_map)
        return df_out, len(valid_map)

    @classmethod
    def sanitize_column_names(cls, df: pd.DataFrame, style: str = "snake_case") -> Tuple[pd.DataFrame, int]:
        """Cleans and standardizes column headers."""
        df_out = df.copy()
        new_cols = []
        renamed_count = 0

        for c in df_out.columns:
            s = str(c).strip()
            if style == "snake_case":
                # Remove special chars, replace spaces with underscores, lower
                s = re.sub(r"[^\w\s]", "", s)
                s = re.sub(r"\s+", "_", s).lower()
            elif style == "camelCase":
                s = re.sub(r"[^\w\s]", "", s)
                words = s.split()
                s = words[0].lower() + "".join(w.capitalize() for w in words[1:]) if words else "col"
            elif style == "upper":
                s = re.sub(r"[^\w\s]", "", s)
                s = re.sub(r"\s+", "_", s).upper()
            else:
                s = re.sub(r"\s+", " ", s)

            if s != c:
                renamed_count += 1
            new_cols.append(s)

        df_out.columns = new_cols
        return df_out, renamed_count

    @classmethod
    def handle_outliers(
        cls,
        df: pd.DataFrame,
        column: str,
        method: str = "iqr_clip",
        factor: float = 1.5
    ) -> Tuple[pd.DataFrame, int]:
        """Detects and clips or removes statistical outliers in a numeric column."""
        df_out = df.copy()
        if column not in df_out.columns:
            return df_out, 0

        numeric_s = pd.to_numeric(df_out[column], errors="coerce")
        q1 = numeric_s.quantile(0.25)
        q3 = numeric_s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return df_out, 0

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        outlier_mask = (numeric_s < lower_bound) | (numeric_s > upper_bound)
        outlier_count = int(outlier_mask.sum())

        if outlier_count == 0:
            return df_out, 0

        if method == "iqr_clip":
            clipped_s = numeric_s.clip(lower=lower_bound, upper=upper_bound)
            df_out[column] = clipped_s
        elif method == "iqr_drop":
            df_out = df_out[~outlier_mask].reset_index(drop=True)

        return df_out, outlier_count

    @classmethod
    def convert_data_type(
        cls,
        df: pd.DataFrame,
        column: str,
        target_type: str
    ) -> Tuple[pd.DataFrame, int]:
        """Coerces column to target data type with clean sanitization."""
        df_out = df.copy()
        if column not in df_out.columns:
            return df_out, 0

        orig = df_out[column]
        converted = orig

        if target_type in ["numeric", "float"]:
            # Strip currency symbols and commas e.g. "$1,234.50"
            s = orig.astype(str).str.replace(r"[$,€£]", "", regex=True).str.replace(",", "", regex=False).str.strip()
            converted = pd.to_numeric(s, errors="coerce")
        elif target_type in ["integer", "int"]:
            s = orig.astype(str).str.replace(r"[$,€£]", "", regex=True).str.replace(",", "", regex=False).str.strip()
            converted = pd.to_numeric(s, errors="coerce").round().astype("Int64")
        elif target_type == "string":
            converted = orig.astype(str).replace("nan", np.nan)
        elif target_type == "datetime":
            converted = pd.to_datetime(orig, errors="coerce")
        elif target_type == "boolean":
            true_vals = {"true", "1", "yes", "y", "t"}
            converted = orig.astype(str).str.lower().str.strip().isin(true_vals)

        diff = orig.ne(converted) & orig.notna()
        df_out[column] = converted
        return df_out, int(diff.sum())

    @classmethod
    def split_column(
        cls,
        df: pd.DataFrame,
        column: str,
        delimiter: str = " ",
        new_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, int]:
        """Splits a single column into multiple columns by delimiter."""
        df_out = df.copy()
        if column not in df_out.columns:
            return df_out, 0

        split_df = df_out[column].astype(str).str.split(delimiter, n=1 if not new_cols or len(new_cols) <= 2 else len(new_cols)-1, expand=True)
        if new_cols:
            for i, name in enumerate(new_cols):
                if i < len(split_df.columns):
                    df_out[name] = split_df[i].str.strip()
                else:
                    df_out[name] = ""
        else:
            for i in range(len(split_df.columns)):
                df_out[f"{column}_{i+1}"] = split_df[i].str.strip()

        return df_out, len(df_out)

    @classmethod
    def merge_columns(
        cls,
        df: pd.DataFrame,
        columns: List[str],
        new_col_name: str,
        separator: str = " "
    ) -> Tuple[pd.DataFrame, int]:
        """Merges multiple columns into a new column."""
        df_out = df.copy()
        valid = [c for c in columns if c in df_out.columns]
        if not valid:
            return df_out, 0

        df_out[new_col_name] = df_out[valid].astype(str).replace("nan", "").agg(separator.join, axis=1).str.strip()
        return df_out, len(df_out)

    @classmethod
    def calculate_column(
        cls,
        df: pd.DataFrame,
        new_col_name: str,
        expression: str
    ) -> Tuple[pd.DataFrame, int]:
        """Evaluates a safe mathematical formula (e.g. 'revenue - cost')."""
        df_out = df.copy()
        # Evaluate using pandas.eval in local environment
        try:
            evaluated = df_out.eval(expression)
            df_out[new_col_name] = evaluated
            return df_out, len(df_out)
        except Exception:
            return df_out, 0

    @classmethod
    def filter_data(
        cls,
        df: pd.DataFrame,
        column: str,
        operator: str,
        value: Any
    ) -> Tuple[pd.DataFrame, int]:
        """Filters rows according to condition."""
        if column not in df.columns:
            return df, 0

        orig_len = len(df)
        s = df[column]

        if operator == ">":
            mask = pd.to_numeric(s, errors="coerce") > float(value)
        elif operator == ">=":
            mask = pd.to_numeric(s, errors="coerce") >= float(value)
        elif operator == "<":
            mask = pd.to_numeric(s, errors="coerce") < float(value)
        elif operator == "<=":
            mask = pd.to_numeric(s, errors="coerce") <= float(value)
        elif operator in ("==", "equals", "is"):
            mask = s.astype(str).str.lower() == str(value).lower()
        elif operator in ("!=", "not_equals", "is_not"):
            mask = s.astype(str).str.lower() != str(value).lower()
        elif operator == "contains":
            mask = s.astype(str).str.contains(str(value), case=False, na=False)
        elif operator == "not_contains":
            mask = ~s.astype(str).str.contains(str(value), case=False, na=False)
        elif operator == "is_null":
            mask = s.isna()
        elif operator == "not_null":
            mask = s.notna()
        else:
            mask = pd.Series([True] * len(df), index=df.index)

        df_out = df[mask].reset_index(drop=True)
        dropped = orig_len - len(df_out)
        return df_out, dropped
