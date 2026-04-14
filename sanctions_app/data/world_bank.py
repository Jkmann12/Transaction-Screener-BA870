"""World Bank Governance Indicator fetcher.

Fetches country-level governance scores for risk assessment.
Falls back to hardcoded scores when API is unavailable.
"""
import requests
from typing import Dict, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import GOVERNANCE_RISK_SCORES, SANCTIONED_COUNTRIES, WORLD_BANK_API


# World Bank governance indicators (control of corruption, rule of law, govt effectiveness)
GOVERNANCE_INDICATORS = [
    'CC.EST',   # Control of Corruption
    'RL.EST',   # Rule of Law
    'GE.EST',   # Government Effectiveness
]


def get_governance_scores(country_code: str) -> float:
    """
    Fetch World Bank governance indicators for a country.

    Returns a composite risk score (0-1, higher = riskier).
    Tries World Bank API first; falls back to hardcoded scores.

    Args:
        country_code: ISO-2 country code

    Returns:
        Risk score between 0 (safe) and 1 (very risky)
    """
    # Check hardcoded scores first for sanctioned countries
    if country_code in SANCTIONED_COUNTRIES:
        return GOVERNANCE_RISK_SCORES.get(country_code, 0.9)

    try:
        scores = []
        for indicator in GOVERNANCE_INDICATORS:
            url = f"{WORLD_BANK_API}/country/{country_code}/indicator/{indicator}"
            params = {'format': 'json', 'mrv': 1, 'per_page': 1}
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and data[1] and data[1][0].get('value') is not None:
                    # WB scores range roughly -2.5 to 2.5; normalize to 0-1 (inverted)
                    raw = float(data[1][0]['value'])
                    normalized = 1 - ((raw + 2.5) / 5.0)
                    normalized = max(0.0, min(1.0, normalized))
                    scores.append(normalized)

        if scores:
            return sum(scores) / len(scores)
    except Exception:
        pass

    return GOVERNANCE_RISK_SCORES.get(country_code, 0.5)


def get_all_governance_scores(country_codes: list) -> Dict[str, float]:
    """
    Return governance risk scores for a list of countries.

    Args:
        country_codes: List of ISO-2 country codes

    Returns:
        Dict mapping country_code -> risk_score
    """
    # Use hardcoded scores (fast, offline-capable)
    return {code: GOVERNANCE_RISK_SCORES.get(code, 0.5) for code in country_codes}
