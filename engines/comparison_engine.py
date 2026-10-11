"""
SheetPilot AI - Comparison Engine
Calculates side-by-side Before vs. After diffs, cell-level modifications,
structural schema changes, and data health improvement metrics.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from engines.data_profiler import DataProfilerEngine


class ComparisonEngine:
    """Computes comprehensive deltas and audit diffs between raw and processed datasets."""

    @classmethod
    def compare(
        cls,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        max_sample_changes: int = 15
    ) -> Dict[str, Any]:
        """Compares before and after DataFrames to yield high-fidelity diff statistics."""
        if df_before is None or df_after is None:
            return {"error": "Missing input data for comparison"}

        rows_before = len(df_before)
        rows_after = len(df_after)
        rows_diff = rows_after - rows_before

        cols_before = list(df_before.columns)
        cols_after = list(df_after.columns)

        cols_added = [c for c in cols_after if c not in cols_before]
        cols_removed = [c for c in cols_before if c not in cols_after]
        common_cols = [c for c in cols_before if c in cols_after]

        # Null cell reduction
        nulls_before = int(df_before.isna().sum().sum())
        nulls_after = int(df_after.isna().sum().sum())
        nulls_diff = nulls_after - nulls_before

        # Cell-level changes for common rows & columns
        min_rows = min(rows_before, rows_after)
        cell_changes_count = 0
        sample_changes: List[Dict[str, Any]] = []

        if min_rows > 0 and common_cols:
            sub_before = df_before.iloc[:min_rows][common_cols]
            sub_after = df_after.iloc[:min_rows][common_cols]

            for col in common_cols:
                s_b = sub_before[col].astype(str)
                s_a = sub_after[col].astype(str)
                mask_diff = (s_b != s_a) & ~(sub_before[col].isna() & sub_after[col].isna())
                changed_in_col = int(mask_diff.sum())
                cell_changes_count += changed_in_col

                if changed_in_col > 0 and len(sample_changes) < max_sample_changes:
                    changed_indices = mask_diff[mask_diff].index[:3]
                    for idx in changed_indices:
                        if len(sample_changes) < max_sample_changes:
                            sample_changes.append({
                                "row": int(idx) + 1,
                                "column": str(col),
                                "before": str(sub_before.at[idx, col]),
                                "after": str(sub_after.at[idx, col]),
                            })

        # Profiles and Health Score delta
        prof_before = DataProfilerEngine.profile(df_before)
        prof_after = DataProfilerEngine.profile(df_after)

        score_before = prof_before["overview"]["health_score"]
        score_after = prof_after["overview"]["health_score"]
        score_diff = score_after - score_before

        return {
            "rows": {
                "before": rows_before,
                "after": rows_after,
                "diff": rows_diff,
                "pct_change": round((rows_diff / rows_before * 100), 2) if rows_before > 0 else 0,
            },
            "columns": {
                "before_count": len(cols_before),
                "after_count": len(cols_after),
                "added": cols_added,
                "removed": cols_removed,
                "common": common_cols,
            },
            "nulls": {
                "before": nulls_before,
                "after": nulls_after,
                "diff": nulls_diff,
            },
            "cell_changes": {
                "count": cell_changes_count,
                "samples": sample_changes,
            },
            "health_score": {
                "before": score_before,
                "after": score_after,
                "diff": score_diff,
                "grade_before": prof_before["overview"]["health_grade"],
                "grade_after": prof_after["overview"]["health_grade"],
            },
            "memory": {
                "before_mb": prof_before["overview"]["memory_mb"],
                "after_mb": prof_after["overview"]["memory_mb"],
            }
        }
