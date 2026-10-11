"""
SheetPilot AI - Column Mapper Engine
Intelligent column matching and alignment using exact, normalized,
synonym-based semantic, and fuzzy string similarity.
"""

from difflib import SequenceMatcher
import re
from typing import Dict, List, Optional, Tuple


class ColumnMapperEngine:
    """Matches arbitrary source spreadsheet columns against target template schemas."""

    # Built-in business synonym dictionaries
    SYNONYM_GROUPS = {
        "email": ["email", "e_mail", "email_address", "mail", "contact_email", "user_email"],
        "phone": ["phone", "tel", "telephone", "mobile", "cell", "contact_number", "phone_number", "contact_no"],
        "first_name": ["first_name", "fname", "first", "given_name", "forename"],
        "last_name": ["last_name", "lname", "last", "surname", "family_name"],
        "full_name": ["name", "full_name", "customer_name", "client_name", "contact_name", "employee_name"],
        "company": ["company", "organization", "org", "employer", "account_name", "firm", "business"],
        "address": ["address", "street", "street_address", "addr", "location", "address_1"],
        "city": ["city", "town", "municipality"],
        "state": ["state", "province", "region"],
        "zip_code": ["zip", "zip_code", "zipcode", "postal_code", "postal", "postcode"],
        "country": ["country", "nation"],
        "date": ["date", "created_date", "created_at", "signup_date", "transaction_date", "timestamp", "order_date"],
        "amount": ["amount", "revenue", "price", "total", "sales", "balance", "cost", "unit_price", "total_amount"],
        "status": ["status", "stage", "state", "account_status", "active_status"],
        "sku": ["sku", "product_id", "item_code", "part_number", "product_code"],
        "quantity": ["qty", "quantity", "count", "stock", "stock_qty", "units", "inventory"],
        "description": ["description", "desc", "notes", "memo", "details", "comments"],
        "id": ["id", "identifier", "code", "customer_id", "account_id", "order_id", "emp_id"],
    }

    @classmethod
    def _normalize_name(cls, name: str) -> str:
        """Strips symbols, punctuation, and lowers for comparison."""
        s = str(name).lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        return re.sub(r"\s+", "_", s)

    @classmethod
    def calculate_similarity(cls, str1: str, str2: str) -> float:
        """Calculates normalized Levenshtein-based similarity ratio (0.0 to 1.0)."""
        return SequenceMatcher(None, cls._normalize_name(str1), cls._normalize_name(str2)).ratio()

    @classmethod
    def match_columns(
        cls,
        source_columns: List[str],
        target_columns: List[str],
        similarity_threshold: float = 0.65
    ) -> Dict[str, Dict]:
        """
        Maps source columns to each target column.
        Returns a dict mapping target_column -> {
            'source_col': matched_col_or_None,
            'confidence': int (0-100),
            'match_type': 'exact' | 'normalized' | 'synonym' | 'fuzzy' | 'unmapped'
        }
        """
        mapping_result = {}
        available_sources = list(source_columns)

        for target in target_columns:
            target_norm = cls._normalize_name(target)
            best_match: Optional[str] = None
            best_score: float = 0.0
            match_type: str = "unmapped"

            # 1. Exact Match
            for src in available_sources:
                if src == target:
                    best_match = src
                    best_score = 1.0
                    match_type = "exact"
                    break

            # 2. Normalized Match
            if not best_match:
                for src in available_sources:
                    if cls._normalize_name(src) == target_norm:
                        best_match = src
                        best_score = 0.95
                        match_type = "normalized"
                        break

            # 3. Synonym Group Match
            if not best_match:
                # Find group for target
                target_group = None
                for group_key, aliases in cls.SYNONYM_GROUPS.items():
                    if target_norm == group_key or target_norm in aliases:
                        target_group = aliases
                        break

                if target_group:
                    for src in available_sources:
                        src_norm = cls._normalize_name(src)
                        if src_norm in target_group:
                            best_match = src
                            best_score = 0.90
                            match_type = "synonym"
                            break

            # 4. Fuzzy Similarity Match
            if not best_match:
                for src in available_sources:
                    score = cls.calculate_similarity(src, target)
                    if score > best_score and score >= similarity_threshold:
                        best_score = score
                        best_match = src
                        match_type = "fuzzy"

            # Record result
            confidence_pct = int(round(best_score * 100))
            mapping_result[target] = {
                "source_col": best_match,
                "confidence": confidence_pct if best_match else 0,
                "match_type": match_type if best_match else "unmapped",
            }

            # If strong match found, don't remove to allow flexibility, or remove if strict 1-to-1:
            # We keep it in available_sources so user can review/remap if desired.

        return mapping_result

    @classmethod
    def suggest_alternatives(
        cls,
        target_column: str,
        source_columns: List[str]
    ) -> List[Tuple[str, int]]:
        """Returns ranked list of candidate source columns with confidence scores."""
        scores = []
        for src in source_columns:
            score = cls.calculate_similarity(src, target_column)
            scores.append((src, int(round(score * 100))))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
