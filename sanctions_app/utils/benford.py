"""Benford's Law analysis utilities for fraud detection.

Benford's Law states that in naturally occurring financial data the leading digit d
appears with probability log10(1 + 1/d). Deviations may indicate fabricated or
manipulated transaction data.
"""
import numpy as np
from scipy.stats import chisquare
from typing import Optional

BENFORD_EXPECTED = {d: np.log10(1 + 1/d) for d in range(1, 10)}


def first_digit(n: float) -> int:
    """Extract the leading digit of a number."""
    n = abs(float(n))
    if n == 0:
        return 0
    while n >= 10:
        n /= 10
    while n < 1:
        n *= 10
    return int(n)


def benford_test(amounts) -> Optional[dict]:
    """
    Compare the first-digit distribution of a list of amounts
    against the expected Benford distribution.

    Returns observed frequencies, expected frequencies, chi2 stat, and p-value.
    Returns None if insufficient data (< 10 observations).
    """
    digits = [first_digit(a) for a in amounts if a > 0]
    n = len(digits)
    if n < 10:
        return None

    observed = np.array([digits.count(d) for d in range(1, 10)], dtype=float)
    expected = np.array([BENFORD_EXPECTED[d] * n for d in range(1, 10)], dtype=float)

    chi2, p_value = chisquare(observed, f_exp=expected)
    return {
        'observed': observed,
        'expected': expected,
        'chi2': float(chi2),
        'p_value': float(p_value),
        'n': n
    }


def get_benford_deviation(amounts) -> float:
    """
    Return the chi-squared statistic for Benford conformance.
    Returns 0.0 if insufficient data.
    Used as a feature in the ML model.
    """
    result = benford_test(amounts)
    if result is None:
        return 0.0
    return result['chi2']
