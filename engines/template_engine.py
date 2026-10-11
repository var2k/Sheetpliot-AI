"""
SheetPilot AI - Template Engine
Conforms and transforms arbitrary source DataFrames into standard enterprise target schemas.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class TemplateEngine:
    """Transforms and reshapes source tabular data to align with target template structures."""

    @classmethod
    def apply_template(
        cls,
        source_df: pd.DataFrame,
        target_columns: List[str],
        column_mapping: Dict[str, Optional[str]],
        keep_unmapped: bool = False,
        default_values: Optional[Dict[str, Any]] = None,
        type_conversions: Optional[Dict[str, str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Transforms source_df into target schema format according to column_mapping.
        target_columns: Ordered list of columns expected in target template.
        column_mapping: Dict mapping target_col_name -> source_col_name (or None)
        """
        if source_df is None or source_df.empty:
            empty_df = pd.DataFrame(columns=target_columns)
            return empty_df, {"mapped_count": 0, "missing_targets": target_columns}

        output_df = pd.DataFrame(index=source_df.index)
        defaults = default_values or {}
        mapped_sources = set()
        successfully_mapped = []
        missing_in_source = []

        # Populate target columns
        for target_col in target_columns:
            source_col = column_mapping.get(target_col)
            if source_col and source_col in source_df.columns:
                output_df[target_col] = source_df[source_col].copy()
                mapped_sources.add(source_col)
                successfully_mapped.append(target_col)
            else:
                # Target column not found in source: fill default or NaN
                def_val = defaults.get(target_col, np.nan)
                output_df[target_col] = def_val
                missing_in_source.append(target_col)

        # Append unmapped columns if requested
        unmapped_sources = [c for c in source_df.columns if c not in mapped_sources]
        if keep_unmapped:
            for unmapped_col in unmapped_sources:
                target_col_name = f"EXTRA_{unmapped_col}"
                output_df[target_col_name] = source_df[unmapped_col].copy()

        # Apply target type conversions if specified
        if type_conversions:
            for col, dtype_target in type_conversions.items():
                if col in output_df.columns:
                    try:
                        if dtype_target == "numeric":
                            output_df[col] = pd.to_numeric(output_df[col], errors="coerce")
                        elif dtype_target == "datetime":
                            output_df[col] = pd.to_datetime(output_df[col], errors="coerce")
                        elif dtype_target == "string":
                            output_df[col] = output_df[col].astype(str)
                    except Exception:
                        pass

        report = {
            "target_columns_total": len(target_columns),
            "successfully_mapped_count": len(successfully_mapped),
            "successfully_mapped_columns": successfully_mapped,
            "missing_in_source_count": len(missing_in_source),
            "missing_in_source_columns": missing_in_source,
            "unmapped_source_columns_count": len(unmapped_sources),
            "unmapped_source_columns": unmapped_sources,
            "kept_unmapped": keep_unmapped,
        }

        return output_df, report

    @classmethod
    def extract_template_schema(cls, template_df: pd.DataFrame) -> Dict[str, Any]:
        """Extracts column names, inferred dtypes, and sample values from a template workbook."""
        schema = {
            "columns": list(template_df.columns),
            "column_types": {str(c): str(template_df[c].dtype) for c in template_df.columns},
            "sample_row": {str(c): (template_df[c].dropna().iloc[0] if not template_df[c].dropna().empty else "") for c in template_df.columns}
        }
        return schema
