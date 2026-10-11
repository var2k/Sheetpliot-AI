"""
SheetPilot AI - Audit Engine
Maintains an immutable, chronologically ordered action ledger
tracking all user actions, pipeline steps, and data transformations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


class AuditEngine:
    """Audit ledger tracking transformation history for compliance and reproducibility."""

    def __init__(self, initial_entries: Optional[List[Dict[str, Any]]] = None):
        self.entries: List[Dict[str, Any]] = initial_entries or []

    def log_action(
        self,
        action: str,
        summary: str,
        parameters: Optional[Dict[str, Any]] = None,
        rows_affected: int = 0,
        cells_affected: int = 0,
        status: str = "SUCCESS"
    ) -> Dict[str, Any]:
        """Appends a new immutable action entry to the ledger."""
        entry = {
            "step": len(self.entries) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "summary": summary,
            "parameters": parameters or {},
            "rows_affected": rows_affected,
            "cells_affected": cells_affected,
            "status": status,
        }
        self.entries.append(entry)
        return entry

    def get_entries(self) -> List[Dict[str, Any]]:
        """Returns all logged audit actions."""
        return list(self.entries)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the audit log into a clean pandas DataFrame."""
        if not self.entries:
            return pd.DataFrame(columns=["Step", "Timestamp", "Action", "Summary", "Rows Affected", "Cells Affected", "Status"])

        rows = []
        for e in self.entries:
            rows.append({
                "Step": e["step"],
                "Timestamp": e["timestamp"],
                "Action": e["action"],
                "Summary": e["summary"],
                "Rows Affected": e["rows_affected"],
                "Cells Affected": e["cells_affected"],
                "Status": e["status"],
            })
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        """Formats audit ledger as a clean markdown table."""
        if not self.entries:
            return "_No transformation actions recorded yet._"

        lines = [
            "| Step | Timestamp | Action | Summary | Impact |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]
        for e in self.entries:
            impact = f"{e['rows_affected']} rows, {e['cells_affected']} cells"
            lines.append(f"| {e['step']} | {e['timestamp']} | **{e['action']}** | {e['summary']} | {impact} |")

        return "\n".join(lines)

    def clear(self):
        """Resets the audit ledger."""
        self.entries = []
