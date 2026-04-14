"""Entity name fuzzy matching against sanctions lists.

Uses rapidfuzz token_sort_ratio to handle name ordering differences
(e.g., "IRAN BANK MELLI" vs "BANK MELLI IRAN").
"""
import pandas as pd
from typing import Optional
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


def match_entity(name: str, sanctions_df: pd.DataFrame, threshold: int = 80) -> dict:
    """
    Match a counterparty name against the sanctions entity list.

    Returns the best match name, score (0-1), and source list.
    Uses token_sort_ratio to handle name ordering differences.

    Args:
        name: Counterparty name to check
        sanctions_df: DataFrame with 'entity_name' and 'source' columns
        threshold: Minimum match score (0-100) to consider a match

    Returns:
        Dict with 'matched_entity', 'match_score', 'source'
    """
    if not RAPIDFUZZ_AVAILABLE or sanctions_df.empty or not name or not isinstance(name, str):
        return {'matched_entity': None, 'match_score': 0.0, 'source': None}

    entity_names = sanctions_df['entity_name'].dropna().tolist()
    if not entity_names:
        return {'matched_entity': None, 'match_score': 0.0, 'source': None}

    result = process.extractOne(
        name, entity_names, scorer=fuzz.token_sort_ratio
    )
    if result and result[1] >= threshold:
        matched_name = result[0]
        source_matches = sanctions_df.loc[sanctions_df['entity_name'] == matched_name, 'source']
        source = source_matches.iloc[0] if not source_matches.empty else 'Unknown'
        return {
            'matched_entity': matched_name,
            'match_score': result[1] / 100.0,
            'source': source
        }
    elif result:
        return {'matched_entity': None, 'match_score': result[1] / 100.0, 'source': None}
    return {'matched_entity': None, 'match_score': 0.0, 'source': None}


def batch_match_entities(names: list, sanctions_df: pd.DataFrame, threshold: int = 80) -> list:
    """
    Apply match_entity to a list of counterparty names.

    Args:
        names: List of counterparty names
        sanctions_df: Sanctions entity DataFrame
        threshold: Minimum match score

    Returns:
        List of match result dicts
    """
    return [match_entity(name, sanctions_df, threshold) for name in names]
