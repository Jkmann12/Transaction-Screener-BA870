"""Constants for the Sanctions Detection App."""
import numpy as np

# Sanctioned country ISO-2 codes (OFAC/UN)
SANCTIONED_COUNTRIES = [
    'IR', 'KP', 'SY', 'CU', 'RU', 'BY', 'MM', 'VE', 'SD', 'LY',
    'SO', 'ZW', 'CF', 'SS', 'YE', 'HT', 'IQ', 'AF', 'ML'
]

# Broader high-risk countries
HIGH_RISK_COUNTRIES = [
    'NG', 'PK', 'BD', 'UA', 'KZ', 'UZ', 'TJ', 'TM', 'AZ', 'GE', 'AM', 'KG',
    'MX', 'CO', 'PE', 'EC', 'TZ', 'KE', 'ET', 'GH', 'SN', 'CM'
]

# Country to primary currency mapping
COUNTRY_CURRENCIES = {
    'US': 'USD', 'GB': 'GBP', 'EU': 'EUR', 'DE': 'EUR', 'FR': 'EUR',
    'IT': 'EUR', 'ES': 'EUR', 'NL': 'EUR', 'BE': 'EUR', 'AT': 'EUR',
    'CH': 'CHF', 'JP': 'JPY', 'CN': 'CNY', 'HK': 'HKD', 'SG': 'SGD',
    'AU': 'AUD', 'CA': 'CAD', 'NZ': 'NZD', 'IN': 'INR', 'BR': 'BRL',
    'MX': 'MXN', 'ZA': 'ZAR', 'NG': 'NGN', 'KE': 'KES', 'EG': 'EGP',
    'AE': 'AED', 'SA': 'SAR', 'QA': 'QAR', 'KW': 'KWD', 'TR': 'TRY',
    'RU': 'RUB', 'UA': 'UAH', 'PL': 'PLN', 'CZ': 'CZK', 'HU': 'HUF',
    'SE': 'SEK', 'NO': 'NOK', 'DK': 'DKK', 'KR': 'KRW', 'TH': 'THB',
    'MY': 'MYR', 'ID': 'IDR', 'PH': 'PHP', 'VN': 'VND', 'PK': 'PKR',
    'BD': 'BDT', 'IR': 'IRR', 'IQ': 'IQD', 'SY': 'SYP', 'KP': 'KPW',
    'CU': 'CUP', 'VE': 'VES', 'BY': 'BYR', 'SD': 'SDG', 'LY': 'LYD',
    'AF': 'AFN', 'MM': 'MMK', 'IL': 'ILS', 'AR': 'ARS', 'CL': 'CLP',
    'CO': 'COP', 'PE': 'PEN', 'EC': 'USD'
}

# Benford's Law expected probabilities
BENFORD_EXPECTED = {d: np.log10(1 + 1/d) for d in range(1, 10)}

# Feature names for ML models (order matters)
FEATURE_NAMES = [
    'sanctions_list_match_score',
    'country_risk_score',
    'sanctioned_country_flag',
    'transaction_amount_log',
    'currency_mismatch',
    'amount_to_revenue_ratio',
    'benford_deviation',
    'transaction_frequency',
    'high_risk_geo_interaction'
]

# Default risk thresholds (analogous to Altman Z-Score zones)
DEFAULT_HIGH_THRESHOLD = 0.7   # High Risk (Z < 1.80 distress zone)
DEFAULT_GREY_THRESHOLD = 0.4   # Grey Zone (1.80 < Z < 2.99)

# Data source URLs
OFAC_SDN_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"
WORLD_BANK_API = "https://api.worldbank.org/v2"

# Hardcoded governance risk scores (0-1, higher = riskier) for offline fallback
GOVERNANCE_RISK_SCORES = {
    'US': 0.1, 'GB': 0.1, 'DE': 0.1, 'FR': 0.15, 'JP': 0.1,
    'CA': 0.1, 'AU': 0.1, 'CH': 0.05, 'NL': 0.1, 'SE': 0.05,
    'NO': 0.05, 'DK': 0.05, 'NZ': 0.05, 'SG': 0.1, 'HK': 0.2,
    'KR': 0.2, 'IL': 0.25, 'ES': 0.2, 'IT': 0.25, 'PT': 0.2,
    'GR': 0.3, 'PL': 0.25, 'CZ': 0.25, 'HU': 0.35, 'TR': 0.55,
    'CN': 0.5, 'IN': 0.45, 'BR': 0.5, 'MX': 0.55, 'AR': 0.55,
    'CL': 0.3, 'CO': 0.5, 'PE': 0.45, 'ZA': 0.5, 'NG': 0.7,
    'KE': 0.55, 'EG': 0.6, 'SA': 0.5, 'AE': 0.3, 'QA': 0.3,
    'UA': 0.6, 'RU': 0.85, 'BY': 0.9, 'KZ': 0.65, 'UZ': 0.7,
    'PK': 0.7, 'BD': 0.65, 'ID': 0.5, 'PH': 0.55, 'MY': 0.4,
    'TH': 0.45, 'VN': 0.55, 'MM': 0.8, 'AF': 0.95, 'IQ': 0.9,
    'SY': 0.95, 'IR': 0.95, 'KP': 0.99, 'CU': 0.85, 'VE': 0.85,
    'SD': 0.9, 'LY': 0.9, 'SO': 0.95, 'YE': 0.9, 'ML': 0.8,
    'CF': 0.9, 'SS': 0.9, 'ZW': 0.8, 'HT': 0.8
}
