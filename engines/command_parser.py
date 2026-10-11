"""
SheetPilot AI - Command Parser Engine
Translates human natural language instructions into structured, executable
data transformation pipelines using rule-based NLP and optional OpenAI LLM fallback.
"""

import json
import re
from typing import Any, Dict, List, Optional
from config.settings import OPENAI_API_KEY, OPENAI_MODEL


class CommandParserEngine:
    """Interprets natural language commands into standardized pipeline actions."""

    # Built-in quick command suggestions
    SUGGESTED_COMMANDS = [
        "Trim whitespace in all columns",
        "Remove all duplicate rows",
        "Sanitize column headers to snake_case",
        "Convert email column to lowercase",
        "Title case customer names",
        "Format phone numbers as (XXX) XXX-XXXX",
        "Standardize dates to YYYY-MM-DD",
        "Drop empty rows with >50% missing cells",
        "Fill missing values in status with 'Active'",
        "Filter where sales > 100",
        "Remove statistical outliers in price",
        "Calculate total_price = price * quantity"
    ]

    @classmethod
    def parse_command(
        cls,
        user_prompt: str,
        available_columns: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Parses a natural language instruction and returns a list of action dicts:
        [
            {
                "action": "trim_whitespace" | "remove_duplicates" | "standardize_case" | ...,
                "params": { ... },
                "description": "..."
            }
        ]
        """
        cleaned_prompt = user_prompt.strip()
        if not cleaned_prompt:
            return []

        # Try LLM if configured
        if OPENAI_API_KEY:
            try:
                llm_actions = cls._parse_with_openai(cleaned_prompt, available_columns)
                if llm_actions:
                    return llm_actions
            except Exception:
                pass  # Fallback to rule-based parser on any network/key issue

        # Rule-based NLP Parser
        return cls._parse_rule_based(cleaned_prompt, available_columns)

    @classmethod
    def _parse_rule_based(
        cls,
        prompt: str,
        columns: List[str]
    ) -> List[Dict[str, Any]]:
        """Regex and keyword-based natural language intent parser."""
        actions: List[Dict[str, Any]] = []
        p_lower = prompt.lower()
        col_lookup = {c.lower(): c for c in columns}

        # 1. Trim whitespace
        if "trim" in p_lower and ("whitespace" in p_lower or "space" in p_lower or "all" in p_lower):
            matched_cols = cls._extract_columns_from_prompt(prompt, columns)
            actions.append({
                "action": "trim_whitespace",
                "params": {"columns": matched_cols if matched_cols else None},
                "description": f"Trim whitespace in {', '.join(matched_cols) if matched_cols else 'all text columns'}"
            })

        # 2. Remove Duplicates
        elif "duplicate" in p_lower or "dedup" in p_lower:
            matched_cols = cls._extract_columns_from_prompt(prompt, columns)
            actions.append({
                "action": "remove_duplicates",
                "params": {"subset": matched_cols if matched_cols else None, "keep": "first"},
                "description": f"Remove duplicate rows{' based on ' + ', '.join(matched_cols) if matched_cols else ''}"
            })

        # 3. Sanitize column names / headers
        elif "sanitize" in p_lower or "clean column" in p_lower or "snake_case" in p_lower or "clean header" in p_lower:
            style = "snake_case"
            if "camel" in p_lower:
                style = "camelCase"
            elif "upper" in p_lower:
                style = "upper"
            actions.append({
                "action": "sanitize_column_names",
                "params": {"style": style},
                "description": f"Sanitize column headers to {style}"
            })

        # 4. Standardize Case (lower, upper, title)
        elif any(w in p_lower for w in ["lowercase", "uppercase", "title case", "titlecase", "lower case", "upper case"]):
            case_type = "title"
            if "lower" in p_lower:
                case_type = "lower"
            elif "upper" in p_lower:
                case_type = "upper"

            matched_cols = cls._extract_columns_from_prompt(prompt, columns)
            if not matched_cols:
                # Default to email or name if found
                for c in columns:
                    if case_type == "lower" and "email" in c.lower():
                        matched_cols.append(c)
                    elif case_type == "title" and ("name" in c.lower() or "company" in c.lower()):
                        matched_cols.append(c)

            if matched_cols:
                actions.append({
                    "action": "standardize_case",
                    "params": {"columns": matched_cols, "case_type": case_type},
                    "description": f"Convert {', '.join(matched_cols)} to {case_type} case"
                })

        # 5. Clean / Standardize Phone numbers
        elif "phone" in p_lower and ("format" in p_lower or "clean" in p_lower or "standardize" in p_lower):
            matched_cols = cls._extract_columns_from_prompt(prompt, columns)
            if not matched_cols:
                matched_cols = [c for c in columns if "phone" in c.lower() or "mobile" in c.lower() or "tel" in c.lower()]

            fmt = "dashed"
            if "e164" in p_lower or "+1" in p_lower:
                fmt = "e164"
            elif "parenthes" in p_lower:
                fmt = "parentheses"
            elif "digit" in p_lower:
                fmt = "digits"

            if matched_cols:
                actions.append({
                    "action": "clean_phone_numbers",
                    "params": {"columns": matched_cols, "phone_format": fmt},
                    "description": f"Standardize phone numbers in {', '.join(matched_cols)} to {fmt}"
                })

        # 6. Standardize Dates
        elif "date" in p_lower and ("standardize" in p_lower or "format" in p_lower or "convert" in p_lower):
            matched_cols = cls._extract_columns_from_prompt(prompt, columns)
            if not matched_cols:
                matched_cols = [c for c in columns if "date" in c.lower() or "time" in c.lower()]

            actions.append({
                "action": "standardize_dates",
                "params": {"columns": matched_cols, "target_format": "%Y-%m-%d"},
                "description": f"Standardize date format in {', '.join(matched_cols)} to YYYY-MM-DD"
            })

        # 7. Drop empty rows
        elif "drop empty" in p_lower or "remove empty row" in p_lower or "drop blank" in p_lower:
            actions.append({
                "action": "drop_empty_rows",
                "params": {"threshold": 0.7},
                "description": "Drop empty or mostly blank rows"
            })

        # 8. Drop columns
        elif p_lower.startswith("drop column") or p_lower.startswith("remove column") or "delete column" in p_lower:
            matched_cols = cls._extract_columns_from_prompt(prompt, columns)
            if matched_cols:
                actions.append({
                    "action": "drop_columns",
                    "params": {"columns": matched_cols},
                    "description": f"Drop columns: {', '.join(matched_cols)}"
                })

        # 9. Fill missing / Impute
        elif "fill" in p_lower or "impute" in p_lower or "replace missing" in p_lower:
            matched_cols = cls._extract_columns_from_prompt(prompt, columns)
            method = "value"
            custom_val = ""

            if "mean" in p_lower:
                method = "mean"
            elif "median" in p_lower:
                method = "median"
            elif "mode" in p_lower:
                method = "mode"
            else:
                # Extract quoted value: 'Active' or "N/A"
                match_val = re.search(r"['\"]([^'\"]+)['\"]", prompt)
                if match_val:
                    custom_val = match_val.group(1)

            col = matched_cols[0] if matched_cols else (columns[0] if columns else "")
            if col:
                actions.append({
                    "action": "fill_missing",
                    "params": {"column": col, "method": method, "custom_val": custom_val},
                    "description": f"Fill missing values in '{col}' using {method if method != 'value' else f'value \"{custom_val}\"'}"
                })

        # 10. Filter data
        elif p_lower.startswith("filter") or "keep where" in p_lower or "keep only" in p_lower:
            matched_cols = cls._extract_columns_from_prompt(prompt, columns)
            col = matched_cols[0] if matched_cols else None

            # Check operators
            op = "=="
            val = ""
            if ">=" in prompt:
                op = ">="
                val = prompt.split(">=")[-1].strip().split()[0]
            elif "<=" in prompt:
                op = "<="
                val = prompt.split("<=")[-1].strip().split()[0]
            elif ">" in prompt:
                op = ">"
                val = prompt.split(">")[-1].strip().split()[0]
            elif "<" in prompt:
                op = "<"
                val = prompt.split("<")[-1].strip().split()[0]
            elif "contains" in p_lower:
                op = "contains"
                val = prompt.split("contains")[-1].strip().strip("'\"").split()[0]
            elif "is" in p_lower:
                op = "=="
                val = prompt.split("is")[-1].strip().strip("'\"").split()[0]

            if col and val:
                actions.append({
                    "action": "filter_data",
                    "params": {"column": col, "operator": op, "value": val},
                    "description": f"Filter rows where {col} {op} {val}"
                })

        # 11. Calculate new column
        elif "calculate" in p_lower or "add column" in p_lower or "=" in prompt:
            match_calc = re.search(r"(?:calculate|add column)\s+([\w_]+)\s*=\s*(.+)", prompt, re.IGNORECASE)
            if match_calc:
                new_col = match_calc.group(1).strip()
                expr = match_calc.group(2).strip()
                actions.append({
                    "action": "calculate_column",
                    "params": {"new_col_name": new_col, "expression": expr},
                    "description": f"Calculate column '{new_col}' = {expr}"
                })

        # 12. Handle outliers
        elif "outlier" in p_lower:
            matched_cols = cls._extract_columns_from_prompt(prompt, columns)
            method = "iqr_clip" if "clip" in p_lower else "iqr_drop"
            for col in matched_cols:
                actions.append({
                    "action": "handle_outliers",
                    "params": {"column": col, "method": method, "factor": 1.5},
                    "description": f"Handle outliers in '{col}' ({method})"
                })

        return actions

    @classmethod
    def _extract_columns_from_prompt(cls, prompt: str, columns: List[str]) -> List[str]:
        """Identifies column names mentioned in the user prompt."""
        found = []
        p_lower = prompt.lower()
        for c in columns:
            c_clean = str(c).lower().strip()
            # Match whole word or exact name
            pattern = r"\b" + re.escape(c_clean) + r"\b"
            if re.search(pattern, p_lower) or c_clean in p_lower:
                found.append(c)
        return found

    @classmethod
    def _parse_with_openai(cls, prompt: str, columns: List[str]) -> List[Dict[str, Any]]:
        """Invokes OpenAI LLM to parse complex transformation requests."""
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        system_prompt = f"""You are a Data Transformation Compiler for SheetPilot AI.
Available columns: {columns}
Allowed actions:
- trim_whitespace (params: columns)
- standardize_case (params: columns, case_type ['title', 'lower', 'upper'])
- clean_phone_numbers (params: columns, phone_format ['dashed', 'e164', 'parentheses', 'digits'])
- clean_emails (params: columns)
- standardize_dates (params: columns, target_format)
- remove_duplicates (params: subset, keep)
- fill_missing (params: column, method ['value', 'mean', 'median', 'mode'], custom_val)
- drop_empty_rows (params: threshold)
- drop_columns (params: columns)
- sanitize_column_names (params: style ['snake_case', 'camelCase', 'upper'])
- filter_data (params: column, operator ['>', '<', '==', '!=', 'contains'], value)
- calculate_column (params: new_col_name, expression)

Output valid JSON only matching:
[{{"action": "...", "params": {{...}}, "description": "..."}}]
"""
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        # Clean JSON markdown if wrapped
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n", "", content)
            content = re.sub(r"\n```$", "", content)

        return json.loads(content)
