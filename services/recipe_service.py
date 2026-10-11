"""
SheetPilot AI - Recipe Service
Manages reusable transformation recipes (macros) for automated, batch data pipelines.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from config.settings import WORKFLOW_DIR
from engines.cleaning_engine import CleaningEngine


class RecipeService:
    """Stores, loads, and executes sequential data transformation recipes."""

    @classmethod
    def save_recipe(
        cls,
        recipe_name: str,
        steps: List[Dict[str, Any]],
        description: str = "",
        target_template: Optional[str] = None
    ) -> Path:
        """Saves a pipeline recipe to the workflows directory."""
        clean_name = "".join(c for c in recipe_name if c.isalnum() or c in ("_", "-")).strip() or "recipe"
        file_path = WORKFLOW_DIR / f"{clean_name}.json"

        recipe_data = {
            "name": recipe_name,
            "description": description,
            "target_template": target_template,
            "steps": steps,
            "created_at": pd.Timestamp.now().isoformat(),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(recipe_data, f, indent=2)

        return file_path

    @classmethod
    def load_recipe(cls, recipe_name_or_path: str) -> Dict[str, Any]:
        """Loads a recipe by name or path."""
        p = Path(recipe_name_or_path)
        if not p.is_file():
            p = WORKFLOW_DIR / f"{recipe_name_or_path}.json"

        if not p.is_file():
            raise FileNotFoundError(f"Recipe '{recipe_name_or_path}' not found.")

        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def list_recipes(cls) -> List[Dict[str, Any]]:
        """Returns metadata of all saved recipes."""
        recipes = []
        for file in WORKFLOW_DIR.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    recipes.append({
                        "filename": file.name,
                        "name": data.get("name", file.stem),
                        "description": data.get("description", ""),
                        "step_count": len(data.get("steps", [])),
                        "created_at": data.get("created_at", ""),
                    })
            except Exception:
                continue
        return recipes

    @classmethod
    def execute_recipe(
        cls,
        df: pd.DataFrame,
        recipe_steps: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Executes a list of recipe steps sequentially on a DataFrame."""
        work_df = df.copy()
        execution_log = []

        for idx, step in enumerate(recipe_steps):
            action = step.get("action")
            params = step.get("params", {})
            desc = step.get("description", f"Step {idx+1}: {action}")

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

                execution_log.append({
                    "step": idx + 1,
                    "action": action,
                    "summary": desc,
                    "status": "SUCCESS",
                    "rows_affected": affected_rows,
                    "cells_affected": affected_cells,
                })
            except Exception as e:
                execution_log.append({
                    "step": idx + 1,
                    "action": action,
                    "summary": f"Failed: {str(e)}",
                    "status": "ERROR",
                    "rows_affected": 0,
                    "cells_affected": 0,
                })

        return work_df, execution_log
