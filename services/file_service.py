"""
SheetPilot AI - File Service
Handles file saving, temporary caching, and professional multi-sheet Excel,
CSV, and JSON generation with enterprise styling.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from config.settings import OUTPUT_DIR, UPLOAD_DIR


class FileService:
    """Manages disk I/O, temporary uploads, and professional styled file exports."""

    @classmethod
    def save_upload(cls, uploaded_file, custom_name: Optional[str] = None) -> Path:
        """Saves an uploaded file to the local uploads directory."""
        filename = custom_name or getattr(uploaded_file, "name", "uploaded_file.xlsx")
        target_path = UPLOAD_DIR / filename

        if hasattr(uploaded_file, "getbuffer"):
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        elif hasattr(uploaded_file, "read"):
            with open(target_path, "wb") as f:
                f.write(uploaded_file.read())

        return target_path

    @classmethod
    def export_excel(
        cls,
        df: pd.DataFrame,
        audit_df: Optional[pd.DataFrame] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        output_filename: str = "SheetPilot_Cleaned_Export.xlsx"
    ) -> bytes:
        """
        Exports a multi-sheet, beautifully formatted Excel workbook:
        Sheet 1: Cleaned Data (with zebra striping, navy bold header, auto-filter, auto-widths)
        Sheet 2: Audit Trail
        Sheet 3: Data Quality Profile
        """
        output_stream = io.BytesIO()

        with pd.ExcelWriter(output_stream, engine="xlsxwriter") as writer:
            workbook = writer.book

            # Formats
            header_format = workbook.add_format({
                "bold": True,
                "text_wrap": False,
                "valign": "vcenter",
                "fg_color": "#1E293B",  # Slate 800 Navy
                "font_color": "#FFFFFF",
                "border": 1,
                "border_color": "#334155",
                "font_name": "Calibri",
                "font_size": 11,
            })
            zebra_format = workbook.add_format({
                "fg_color": "#F8FAFC",
                "border": 1,
                "border_color": "#E2E8F0",
                "font_name": "Calibri",
                "font_size": 10,
            })
            normal_format = workbook.add_format({
                "border": 1,
                "border_color": "#E2E8F0",
                "font_name": "Calibri",
                "font_size": 10,
            })

            # 1. Sheet: Cleaned Data
            sheet_name = "Cleaned Data"
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
            worksheet = writer.sheets[sheet_name]

            # Header formatting & Auto-filter
            for col_idx, col_name in enumerate(df.columns):
                worksheet.write(0, col_idx, str(col_name), header_format)

            worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

            # Auto-adjust column widths
            for i, col in enumerate(df.columns):
                max_len = max(
                    df[col].astype(str).map(len).max() if len(df) > 0 else 0,
                    len(str(col))
                )
                col_width = min(50, max(max_len + 3, 12))
                worksheet.set_column(i, i, col_width)

            # Zebra striping for data rows
            for row_idx in range(1, len(df) + 1):
                row_fmt = zebra_format if (row_idx % 2 == 0) else normal_format
                for col_idx in range(len(df.columns)):
                    val = df.iat[row_idx - 1, col_idx]
                    if pd.notna(val):
                        worksheet.write(row_idx, col_idx, val, row_fmt)
                    else:
                        worksheet.write_blank(row_idx, col_idx, None, row_fmt)

            # 2. Sheet: Audit Trail
            if audit_df is not None and not audit_df.empty:
                audit_sheet_name = "Audit Trail"
                audit_df.to_excel(writer, sheet_name=audit_sheet_name, index=False)
                audit_ws = writer.sheets[audit_sheet_name]
                for col_idx, col_name in enumerate(audit_df.columns):
                    audit_ws.write(0, col_idx, str(col_name), header_format)
                for i, col in enumerate(audit_df.columns):
                    max_len = max(audit_df[col].astype(str).map(len).max() if len(audit_df) > 0 else 0, len(str(col)))
                    audit_ws.set_column(i, i, min(50, max(max_len + 3, 12)))

            # 3. Sheet: Data Quality Profile
            if profile_data and "columns" in profile_data:
                prof_sheet_name = "Quality Profile"
                cols_prof = profile_data["columns"]
                summary_rows = []
                for cname, cdata in cols_prof.items():
                    summary_rows.append({
                        "Column": cname,
                        "Type": cdata.get("inferred_type"),
                        "Missing Count": cdata.get("null_count"),
                        "Missing %": cdata.get("null_percentage"),
                        "Unique Count": cdata.get("unique_count"),
                        "Outliers": cdata.get("outliers_count", 0),
                    })
                prof_df = pd.DataFrame(summary_rows)
                prof_df.to_excel(writer, sheet_name=prof_sheet_name, index=False)
                prof_ws = writer.sheets[prof_sheet_name]
                for col_idx, col_name in enumerate(prof_df.columns):
                    prof_ws.write(0, col_idx, str(col_name), header_format)
                for i, col in enumerate(prof_df.columns):
                    prof_ws.set_column(i, i, 18)

        output_stream.seek(0)
        return output_stream.getvalue()

    @classmethod
    def export_csv(cls, df: pd.DataFrame) -> bytes:
        """Exports dataframe as UTF-8-SIG CSV (preserves special characters in Excel)."""
        return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    @classmethod
    def export_json(cls, df: pd.DataFrame) -> str:
        """Exports dataframe as formatted JSON records."""
        return df.to_json(orient="records", indent=2, date_format="iso")
