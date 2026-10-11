"""
SheetPilot AI - File Reader Engine
Reads Excel (.xlsx, .xls) and CSV/TSV files with automated encoding detection,
delimiter detection, sheet inspection, and header sniffing.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd


class FileReaderEngine:
    """Robust tabular data ingestion engine."""

    COMMON_ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
    COMMON_DELIMITERS = [",", "\t", ";", "|"]

    @staticmethod
    def get_sheet_names(file_source: Union[str, Path, io.BytesIO, bytes], filename: str = "") -> List[str]:
        """Return list of sheet names for Excel files, or a default name for CSVs."""
        name = str(filename).lower()
        if name.endswith(".csv") or name.endswith(".tsv"):
            return ["Sheet1"]

        try:
            excel_file = pd.ExcelFile(file_source)
            return excel_file.sheet_names
        except Exception:
            # Fallback if not an Excel file or unreadable
            return ["Sheet1"]

    @classmethod
    def read_file(
        cls,
        file_source: Union[str, Path, io.BytesIO, bytes],
        filename: str,
        sheet_name: Optional[str] = None,
        header_row: Optional[int] = None,
        max_rows: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Reads tabular data from a file path or in-memory byte buffer.
        Returns (DataFrame, metadata_dict).
        """
        filename_lower = filename.lower()
        df: Optional[pd.DataFrame] = None
        detected_encoding: Optional[str] = None
        detected_delimiter: Optional[str] = None
        sheets_list: List[str] = []

        if filename_lower.endswith((".xlsx", ".xls")):
            excel_file = pd.ExcelFile(file_source)
            sheets_list = excel_file.sheet_names
            target_sheet = sheet_name if sheet_name in sheets_list else sheets_list[0]

            # Read with potential header detection
            if header_row is not None:
                df = pd.read_excel(excel_file, sheet_name=target_sheet, header=header_row, nrows=max_rows)
            else:
                # Read raw first to detect header
                raw_df = pd.read_excel(excel_file, sheet_name=target_sheet, header=None, nrows=20)
                detected_header = cls._detect_header_row(raw_df)
                df = pd.read_excel(excel_file, sheet_name=target_sheet, header=detected_header, nrows=max_rows)

        elif filename_lower.endswith((".csv", ".tsv", ".txt")):
            sheets_list = ["Sheet1"]
            target_sheet = "Sheet1"
            # Read content bytes for encoding & delimiter sniffing
            content_bytes: bytes = b""
            if isinstance(file_source, (str, Path)):
                with open(file_source, "rb") as f:
                    content_bytes = f.read()
            elif isinstance(file_source, io.BytesIO):
                file_source.seek(0)
                content_bytes = file_source.read()
                file_source.seek(0)
            elif isinstance(file_source, bytes):
                content_bytes = file_source

            # Try common encodings
            text_content: Optional[str] = None
            for enc in cls.COMMON_ENCODINGS:
                try:
                    text_content = content_bytes.decode(enc)
                    detected_encoding = enc
                    break
                except UnicodeDecodeError:
                    continue

            if text_content is None:
                detected_encoding = "latin1"
                text_content = content_bytes.decode("latin1", errors="replace")

            # Detect delimiter
            sample_lines = "\n".join(text_content.splitlines()[:15])
            detected_delimiter = cls._detect_delimiter(sample_lines)

            # Detect header row
            if header_row is None:
                raw_df = pd.read_csv(
                    io.StringIO(text_content),
                    sep=detected_delimiter,
                    header=None,
                    nrows=20,
                    on_bad_lines="skip"
                )
                header_row = cls._detect_header_row(raw_df)

            df = pd.read_csv(
                io.StringIO(text_content),
                sep=detected_delimiter,
                header=header_row,
                nrows=max_rows,
                on_bad_lines="skip"
            )
        else:
            raise ValueError(f"Unsupported file format for '{filename}'. Supported: .xlsx, .xls, .csv, .tsv")

        if df is None:
            raise ValueError(f"Could not parse tabular data from {filename}")

        # Post-clean columns: ensure strings and remove completely unnamed empty tails
        df.columns = [str(c).strip() if not pd.isna(c) else f"Column_{i+1}" for i, c in enumerate(df.columns)]

        memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

        metadata = {
            "filename": filename,
            "sheet_name": target_sheet,
            "available_sheets": sheets_list,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "memory_mb": memory_mb,
            "encoding": detected_encoding,
            "delimiter": detected_delimiter,
        }

        return df, metadata

    @classmethod
    def _detect_delimiter(cls, sample_text: str) -> str:
        """Heuristically detect the CSV delimiter based on frequency in top lines."""
        lines = [line for line in sample_text.splitlines() if line.strip()][:5]
        if not lines:
            return ","

        counts = {delim: sum(line.count(delim) for line in lines) for delim in cls.COMMON_DELIMITERS}
        best_delim = max(counts, key=counts.get)
        return best_delim if counts[best_delim] > 0 else ","

    @classmethod
    def _detect_header_row(cls, sample_df: pd.DataFrame) -> int:
        """
        Inspect sample rows to find the first non-empty, predominantly string row
        that likely represents column headers.
        """
        if sample_df.empty:
            return 0

        for idx, row in sample_df.iterrows():
            non_null = row.dropna()
            # If row has reasonable number of non-null values
            if len(non_null) >= max(1, int(len(row) * 0.4)):
                # If majority are string-like and not purely numbers
                string_count = sum(isinstance(v, str) and not v.strip().replace(".", "", 1).isdigit() for v in non_null)
                if string_count / len(non_null) >= 0.5:
                    return int(idx)

        return 0
