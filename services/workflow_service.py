"""
SheetPilot AI - Workflow Service
Central orchestration service coordinating data transformations,
undo/redo versioning, audit logging, and profiling.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from engines.audit_engine import AuditEngine
from engines.cleaning_engine import CleaningEngine
from engines.column_mapper import ColumnMapperEngine
from engines.comparison_engine import ComparisonEngine
from engines.data_profiler import DataProfilerEngine
from engines.template_engine import TemplateEngine


class WorkflowService:
    """Manages the full reactive data processing session."""

    def __init__(self):
        self.original_df: Optional[pd.DataFrame] = None
        self.active_df: Optional[pd.DataFrame] = None
        self.filename: str = ""
        self.sheet_name: str = ""
        self.history_stack: List[pd.DataFrame] = []
        self.redo_stack: List[pd.DataFrame] = []
        self.audit_engine = AuditEngine()

    def set_dataset(self, df: pd.DataFrame, filename: str, sheet_name: str = "Sheet1"):
        """Initializes or resets the active workspace with a new dataset."""
        self.original_df = df.copy()
        self.active_df = df.copy()
        self.filename = filename
        self.sheet_name = sheet_name
        self.history_stack = []
        self.redo_stack = []
        self.audit_engine.clear()
        self.audit_engine.log_action(
            action="dataset_loaded",
            summary=f"Loaded '{filename}' ({sheet_name}) with {len(df)} rows and {len(df.columns)} columns",
            rows_affected=len(df),
            cells_affected=len(df) * len(df.columns)
        )

    def is_loaded(self) -> bool:
        """Returns True if a valid dataset is currently loaded."""
        return self.active_df is not None and not self.active_df.empty

    def push_state(self):
        """Pushes current active DataFrame onto history stack before a modification."""
        if self.active_df is not None:
            self.history_stack.append(self.active_df.copy())
            self.redo_stack.clear()
            # Keep max 15 states in memory to prevent memory explosion
            if len(self.history_stack) > 15:
                self.history_stack.pop(0)

    def undo(self) -> bool:
        """Rolls back to the previous DataFrame state."""
        if not self.history_stack or self.active_df is None:
            return False

        self.redo_stack.append(self.active_df.copy())
        self.active_df = self.history_stack.pop()
        self.audit_engine.log_action(
            action="undo",
            summary="Reverted last transformation step",
            rows_affected=len(self.active_df)
        )
        return True

    def redo(self) -> bool:
        """Re-applies the previously undone state."""
        if not self.redo_stack or self.active_df is None:
            return False

        self.history_stack.append(self.active_df.copy())
        self.active_df = self.redo_stack.pop()
        self.audit_engine.log_action(
            action="redo",
            summary="Re-applied previously undone step",
            rows_affected=len(self.active_df)
        )
        return True

    def reset_to_original(self):
        """Resets active DataFrame back to original loaded state."""
        if self.original_df is not None:
            self.push_state()
            self.active_df = self.original_df.copy()
            self.audit_engine.log_action(
                action="reset_to_original",
                summary="Reset dataset back to original uploaded state",
                rows_affected=len(self.active_df)
            )

    def execute_action(
        self,
        action: str,
        params: Dict[str, Any],
        description: str
    ) -> Dict[str, Any]:
        """Executes a single atomic transformation action on the active dataset."""
        if self.active_df is None:
            return {"status": "ERROR", "message": "No dataset loaded"}

        self.push_state()
        work_df = self.active_df.copy()
        affected_cells = 0
        affected_rows = 0

        try:
            if action == "trim_whitespace":
                work_df, affected_cells = CleaningEngine.trim_whitespace(work_df, params.get("columns"))
            elif action == "standardize_case":
                work_df, affected_cells = CleaningEngine.standardize_case(
                    work_df, params.get("columns", []), params.get("case_type", "title")
                )
            elif action == "clean_phone_numbers":
                work_df, affected_cells = CleaningEngine.clean_phone_numbers(
                    work_df, params.get("columns", []), params.get("phone_format", "dashed")
                )
            elif action == "clean_emails":
                work_df, affected_cells = CleaningEngine.clean_emails(work_df, params.get("columns", []))
            elif action == "standardize_dates":
                work_df, affected_cells = CleaningEngine.standardize_dates(
                    work_df, params.get("columns", []), params.get("target_format", "%Y-%m-%d")
                )
            elif action == "remove_duplicates":
                work_df, affected_rows = CleaningEngine.remove_duplicates(
                    work_df, params.get("subset"), params.get("keep", "first")
                )
            elif action == "fill_missing":
                work_df, affected_cells = CleaningEngine.fill_missing(
                    work_df, params.get("column", ""), params.get("method", "value"), params.get("custom_val")
                )
            elif action == "drop_empty_rows":
                work_df, affected_rows = CleaningEngine.drop_empty_rows(work_df, params.get("threshold", 0.7))
            elif action == "drop_columns":
                work_df, affected_cells = CleaningEngine.drop_columns(work_df, params.get("columns", []))
            elif action == "rename_columns":
                work_df, affected_cells = CleaningEngine.rename_columns(work_df, params.get("rename_map", {}))
            elif action == "sanitize_column_names":
                work_df, affected_cells = CleaningEngine.sanitize_column_names(work_df, params.get("style", "snake_case"))
            elif action == "handle_outliers":
                work_df, affected_cells = CleaningEngine.handle_outliers(
                    work_df, params.get("column", ""), params.get("method", "iqr_clip"), params.get("factor", 1.5)
                )
            elif action == "convert_data_type":
                work_df, affected_cells = CleaningEngine.convert_data_type(
                    work_df, params.get("column", ""), params.get("target_type", "string")
                )
            elif action == "filter_data":
                work_df, affected_rows = CleaningEngine.filter_data(
                    work_df, params.get("column", ""), params.get("operator", "=="), params.get("value")
                )
            elif action == "calculate_column":
                work_df, affected_rows = CleaningEngine.calculate_column(
                    work_df, params.get("new_col_name", ""), params.get("expression", "")
                )
            elif action == "split_column":
                work_df, affected_rows = CleaningEngine.split_column(
                    work_df, params.get("column", ""), params.get("delimiter", " "), params.get("new_cols")
                )
            elif action == "merge_columns":
                work_df, affected_rows = CleaningEngine.merge_columns(
                    work_df, params.get("columns", []), params.get("new_col_name", ""), params.get("separator", " ")
                )
            else:
                return {"status": "ERROR", "message": f"Unknown action: {action}"}

            self.active_df = work_df
            entry = self.audit_engine.log_action(
                action=action,
                summary=description,
                parameters=params,
                rows_affected=affected_rows,
                cells_affected=affected_cells
            )

            return {
                "status": "SUCCESS",
                "message": description,
                "rows_affected": affected_rows,
                "cells_affected": affected_cells,
                "entry": entry
            }
        except Exception as e:
            # Revert push on exception
            if self.history_stack:
                self.active_df = self.history_stack.pop()
            return {"status": "ERROR", "message": str(e)}

    def apply_template_transformation(
        self,
        target_columns: List[str],
        column_mapping: Dict[str, Optional[str]],
        keep_unmapped: bool = False
    ) -> Dict[str, Any]:
        """Conforms active dataset into target schema format."""
        if self.active_df is None:
            return {"status": "ERROR", "message": "No active dataset"}

        self.push_state()
        new_df, report = TemplateEngine.apply_template(
            source_df=self.active_df,
            target_columns=target_columns,
            column_mapping=column_mapping,
            keep_unmapped=keep_unmapped
        )
        self.active_df = new_df

        self.audit_engine.log_action(
            action="apply_target_template",
            summary=f"Conformed dataset to target schema ({report['successfully_mapped_count']}/{len(target_columns)} columns mapped)",
            parameters={"target_columns": target_columns, "mapping": column_mapping},
            rows_affected=len(new_df)
        )

        return {"status": "SUCCESS", "report": report}

    def get_diff(self) -> Dict[str, Any]:
        """Calculates comparison diff against original dataset."""
        if self.original_df is None or self.active_df is None:
            return {}
        return ComparisonEngine.compare(self.original_df, self.active_df)

    def get_profile(self) -> Dict[str, Any]:
        """Profiles the active dataset."""
        if self.active_df is None:
            return DataProfilerEngine._empty_profile()
        return DataProfilerEngine.profile(self.active_df)
