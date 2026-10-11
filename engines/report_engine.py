"""
SheetPilot AI - Report Engine
Generates standalone, beautifully styled HTML Data Quality & Audit Reports
and Markdown executive summaries.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


class ReportEngine:
    """Creates downloadable interactive HTML reports and markdown summaries."""

    @classmethod
    def generate_html_report(
        cls,
        filename: str,
        profile_data: Dict[str, Any],
        comparison_data: Optional[Dict[str, Any]] = None,
        audit_entries: Optional[List[Dict[str, Any]]] = None,
        validation_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Renders an executive-ready, standalone HTML audit report."""
        overview = profile_data.get("overview", {})
        cols_profile = profile_data.get("columns", {})
        recs = profile_data.get("recommendations", [])
        warnings = profile_data.get("warnings", [])
        audit_list = audit_entries or []

        score = overview.get("health_score", 100)
        grade = overview.get("health_grade", "A+")
        score_color = "#10B981" if score >= 80 else "#F59E0B" if score >= 60 else "#EF4444"

        # Generate column table rows
        col_rows_html = ""
        for col_name, cinfo in cols_profile.items():
            null_badge = f'<span class="badge badge-warning">{cinfo.get("null_percentage")}%</span>' if cinfo.get("null_percentage", 0) > 0 else '<span class="badge badge-success">0%</span>'
            col_rows_html += f"""
            <tr>
                <td><strong>{col_name}</strong></td>
                <td><span class="badge badge-info">{cinfo.get("inferred_type")}</span></td>
                <td>{null_badge} ({cinfo.get("null_count")})</td>
                <td>{cinfo.get("unique_count")} ({cinfo.get("unique_percentage")}%)</td>
                <td>{cinfo.get("outliers_count", 0)}</td>
                <td><small>{", ".join(cinfo.get("sample_values", []))}</small></td>
            </tr>
            """

        # Generate audit rows
        audit_rows_html = ""
        for a in audit_list:
            audit_rows_html += f"""
            <tr>
                <td>#{a.get('step')}</td>
                <td><small>{a.get('timestamp')}</small></td>
                <td><strong>{a.get('action')}</strong></td>
                <td>{a.get('summary')}</td>
                <td><span class="badge badge-info">{a.get('rows_affected')} rows</span></td>
            </tr>
            """
        if not audit_list:
            audit_rows_html = "<tr><td colspan='5' style='text-align:center; color:#94A3B8;'>No transformation operations recorded.</td></tr>"

        # Comparison Section HTML
        comp_html = ""
        if comparison_data and "health_score" in comparison_data:
            c_score = comparison_data["health_score"]
            c_rows = comparison_data["rows"]
            c_cells = comparison_data.get("cell_changes", {})
            comp_html = f"""
            <div class="card">
                <h2>📈 Transformation Impact & Diff</h2>
                <div class="grid grid-3">
                    <div class="metric-box">
                        <div class="metric-label">Health Score Delta</div>
                        <div class="metric-value" style="color: {'#10B981' if c_score.get('diff', 0) >= 0 else '#EF4444'};">
                            {c_score.get('before')} &rarr; {c_score.get('after')} ({'+' if c_score.get('diff', 0) >= 0 else ''}{c_score.get('diff')} pts)
                        </div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Rows Modified</div>
                        <div class="metric-value">{c_rows.get('before')} &rarr; {c_rows.get('after')} ({c_rows.get('diff')})</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Total Cell Values Altered</div>
                        <div class="metric-value">{c_cells.get('count', 0)}</div>
                    </div>
                </div>
            </div>
            """

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SheetPilot AI - Data Health & Audit Report ({filename})</title>
    <style>
        :root {{
            --primary: #4F46E5;
            --bg: #0F172A;
            --card-bg: #1E293B;
            --text-main: #F8FAFC;
            --text-muted: #94A3B8;
            --border: #334155;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 30px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            color: #FFFFFF;
        }}
        .badge-logo {{
            background: linear-gradient(135deg, #4F46E5, #06B6D4);
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 14px;
        }}
        .grid {{
            display: grid;
            gap: 20px;
            margin-bottom: 25px;
        }}
        .grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
        .grid-3 {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        .card h2 {{
            margin-top: 0;
            font-size: 18px;
            color: #E2E8F0;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
            margin-bottom: 18px;
        }}
        .metric-box {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border);
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-label {{
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 26px;
            font-weight: 700;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            text-align: left;
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: var(--text-muted);
            background: rgba(15, 23, 42, 0.4);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.05em;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-success {{ background: rgba(16, 185, 129, 0.2); color: #10B981; }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: #F59E0B; }}
        .badge-info {{ background: rgba(79, 70, 229, 0.2); color: #818CF8; }}
        .recs-list {{
            margin: 0;
            padding-left: 20px;
        }}
        .recs-list li {{
            margin-bottom: 8px;
            color: #CBD5E1;
        }}
        .footer {{
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 40px;
            border-top: 1px solid var(--border);
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <span class="badge-logo">⚡ SheetPilot AI</span>
                <h1 style="margin-top: 8px;">Data Quality & Transformation Audit</h1>
                <div style="color: var(--text-muted); font-size: 13px;">Target File: <strong>{filename}</strong> | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            </div>
            <div style="text-align: right;">
                <div class="metric-label">Health Score</div>
                <div style="font-size: 42px; font-weight: 800; color: {score_color};">{score} <span style="font-size: 20px;">({grade})</span></div>
            </div>
        </div>

        <div class="grid grid-4">
            <div class="metric-box">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{overview.get("total_rows", 0):,}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Columns</div>
                <div class="metric-value">{overview.get("total_columns", 0)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Missing Cells</div>
                <div class="metric-value" style="color: {'#EF4444' if overview.get('null_percentage', 0) > 10 else '#10B981'};">
                    {overview.get("null_percentage", 0)}%
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Duplicate Rows</div>
                <div class="metric-value" style="color: {'#EF4444' if overview.get('duplicate_rows', 0) > 0 else '#10B981'};">
                    {overview.get("duplicate_rows", 0)}
                </div>
            </div>
        </div>

        {comp_html}

        <div class="card">
            <h2>📊 Column Health & Type Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column Name</th>
                        <th>Type</th>
                        <th>Missingness</th>
                        <th>Distinct Values</th>
                        <th>Outliers</th>
                        <th>Sample Values</th>
                    </tr>
                </thead>
                <tbody>
                    {col_rows_html}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>📜 Reproducible Transformation Audit Ledger</h2>
            <table>
                <thead>
                    <tr>
                        <th>Step</th>
                        <th>Timestamp</th>
                        <th>Operation</th>
                        <th>Summary</th>
                        <th>Rows Affected</th>
                    </tr>
                </thead>
                <tbody>
                    {audit_rows_html}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>💡 AI Quality Recommendations</h2>
            <ul class="recs-list">
                {"".join(f"<li>{r}</li>" for r in recs)}
            </ul>
        </div>

        <div class="footer">
            Generated autonomously by SheetPilot AI &bull; Enterprise Excel & Data Wrangling Engine
        </div>
    </div>
</body>
</html>
"""
        return html_template
