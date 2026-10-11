"""
SheetPilot AI - Validation Engine
Applies strict business validation rules, format audits,
value-set constraints, and uniqueness checks.
"""

import re
from typing import Any, Dict, List, Optional
import pandas as pd


class ValidationEngine:
    """Evaluates user-configured or automatic schema validation rules on DataFrames."""

    COMMON_PATTERNS = {
        "email": r"^[\w\.-]+@[\w\.-]+\.\w+$",
        "phone_us": r"^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$",
        "zip_us": r"^\d{5}(-\d{4})?$",
        "alphanumeric": r"^[a-zA-Z0-9_]+$",
        "numeric": r"^-?\d+(\.\d+)?$",
    }

    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        rules: List[Dict[str, Any]],
        max_violations_sample: int = 50
    ) -> Dict[str, Any]:
        """
        Validates df against list of rule dicts:
        Each rule dict format:
        {
            'column': 'email',
            'rule_type': 'not_null' | 'unique' | 'regex' | 'range' | 'allowed_values' | 'dtype',
            'params': { ... }
        }
        """
        if df is None or df.empty:
            return {"passed": True, "violations_count": 0, "violations": []}

        all_violations: List[Dict[str, Any]] = []
        rule_summary: List[Dict[str, Any]] = []

        for rule in rules:
            col = rule.get("column")
            rule_type = rule.get("rule_type")
            params = rule.get("params", {})

            if col not in df.columns:
                rule_summary.append({
                    "column": col,
                    "rule": rule_type,
                    "status": "SKIPPED",
                    "reason": f"Column '{col}' not found in dataset",
                    "violations": 0
                })
                continue

            series = df[col]
            violating_indices: List[int] = []
            reason = ""

            if rule_type == "not_null":
                mask = series.isna() | (series.astype(str).str.strip() == "")
                violating_indices = series[mask].index.tolist()
                reason = "Value is missing or empty"

            elif rule_type == "unique":
                non_null_s = series.dropna()
                dupe_mask = non_null_s.duplicated(keep=False)
                violating_indices = non_null_s[dupe_mask].index.tolist()
                reason = "Duplicate value detected in unique field"

            elif rule_type == "regex":
                pattern_str = params.get("pattern", "")
                if pattern_str in cls.COMMON_PATTERNS:
                    pattern_str = cls.COMMON_PATTERNS[pattern_str]
                pattern = re.compile(pattern_str)
                non_null_s = series.dropna().astype(str).str.strip()
                mask = ~non_null_s.apply(lambda x: bool(pattern.match(x)))
                violating_indices = non_null_s[mask].index.tolist()
                reason = f"Value failed regex pattern match '{pattern_str}'"

            elif rule_type == "range":
                min_val = params.get("min")
                max_val = params.get("max")
                num_s = pd.to_numeric(series, errors="coerce")
                mask = pd.Series([False] * len(series), index=series.index)
                if min_val is not None:
                    mask |= (num_s < float(min_val))
                if max_val is not None:
                    mask |= (num_s > float(max_val))
                violating_indices = series[mask].index.tolist()
                reason = f"Numeric value out of range [{min_val}, {max_val}]"

            elif rule_type == "allowed_values":
                allowed = [str(x).lower().strip() for x in params.get("values", [])]
                non_null_s = series.dropna().astype(str).str.lower().str.strip()
                mask = ~non_null_s.isin(allowed)
                violating_indices = non_null_s[mask].index.tolist()
                reason = f"Value not in allowed set: {params.get('values')}"

            # Log violations
            violations_count = len(violating_indices)
            rule_summary.append({
                "column": col,
                "rule": rule_type,
                "status": "FAIL" if violations_count > 0 else "PASS",
                "violations": violations_count,
            })

            for idx in violating_indices:
                if len(all_violations) < max_violations_sample:
                    all_violations.append({
                        "row": int(idx) + 1,
                        "column": str(col),
                        "rule": rule_type,
                        "value": str(series.at[idx]),
                        "reason": reason,
                    })

        total_violations = sum(r["violations"] for r in rule_summary)
        return {
            "passed": total_violations == 0,
            "total_violations": total_violations,
            "rule_summary": rule_summary,
            "violations": all_violations,
        }

    @classmethod
    def auto_generate_rules(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Automatically infers recommended validation rules from dataset characteristics."""
        rules = []
        for col in df.columns:
            s = df[col].dropna()
            if s.empty:
                continue

            col_lower = str(col).lower()

            # ID / Unique
            if "id" in col_lower or "code" in col_lower or s.nunique() == len(df):
                rules.append({"column": col, "rule_type": "not_null", "params": {}})
                rules.append({"column": col, "rule_type": "unique", "params": {}})

            # Email
            elif "email" in col_lower or "mail" in col_lower:
                rules.append({"column": col, "rule_type": "not_null", "params": {}})
                rules.append({"column": col, "rule_type": "regex", "params": {"pattern": "email"}})

            # Phone
            elif "phone" in col_lower or "mobile" in col_lower or "tel" in col_lower:
                rules.append({"column": col, "rule_type": "regex", "params": {"pattern": "phone_us"}})

            # Numeric positivity / range
            elif pd.api.types.is_numeric_dtype(s):
                if (s >= 0).all():
                    rules.append({"column": col, "rule_type": "range", "params": {"min": 0}})

            # Low cardinality categorical
            elif s.nunique() <= 5 and len(s) > 20:
                rules.append({
                    "column": col,
                    "rule_type": "allowed_values",
                    "params": {"values": s.unique().tolist()}
                })

        return rules
