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
    # North America & Caribbean
    'US': 'USD', 'CA': 'CAD', 'MX': 'MXN', 'PA': 'USD', 'CR': 'CRC',
    'GT': 'GTQ', 'HN': 'HNL', 'SV': 'USD', 'NI': 'NIO', 'DO': 'DOP',
    'JM': 'JMD', 'TT': 'TTD', 'CU': 'CUP', 'HT': 'HTG',
    # South America
    'BR': 'BRL', 'AR': 'ARS', 'CL': 'CLP', 'CO': 'COP', 'PE': 'PEN',
    'VE': 'VES', 'EC': 'USD', 'BO': 'BOB', 'PY': 'PYG', 'UY': 'UYU',
    # Western Europe
    'GB': 'GBP', 'DE': 'EUR', 'FR': 'EUR', 'IT': 'EUR', 'ES': 'EUR',
    'NL': 'EUR', 'BE': 'EUR', 'AT': 'EUR', 'CH': 'CHF', 'SE': 'SEK',
    'NO': 'NOK', 'DK': 'DKK', 'FI': 'EUR', 'IE': 'EUR', 'PT': 'EUR',
    'LU': 'EUR', 'MT': 'EUR', 'IS': 'ISK', 'EU': 'EUR',
    # Central & Eastern Europe
    'PL': 'PLN', 'CZ': 'CZK', 'HU': 'HUF', 'RO': 'RON', 'BG': 'BGN',
    'HR': 'EUR', 'SK': 'EUR', 'SI': 'EUR', 'EE': 'EUR', 'LV': 'EUR',
    'LT': 'EUR', 'GR': 'EUR', 'CY': 'EUR', 'RS': 'RSD', 'AL': 'ALL',
    'BA': 'BAM', 'MK': 'MKD', 'MD': 'MDL', 'UA': 'UAH',
    # Middle East
    'AE': 'AED', 'SA': 'SAR', 'QA': 'QAR', 'KW': 'KWD', 'BH': 'BHD',
    'OM': 'OMR', 'JO': 'JOD', 'LB': 'LBP', 'IL': 'ILS', 'TR': 'TRY',
    'IR': 'IRR', 'IQ': 'IQD', 'SY': 'SYP', 'YE': 'YER',
    # South & Southeast Asia
    'JP': 'JPY', 'CN': 'CNY', 'HK': 'HKD', 'SG': 'SGD', 'KR': 'KRW',
    'TW': 'TWD', 'IN': 'INR', 'PK': 'PKR', 'BD': 'BDT', 'LK': 'LKR',
    'NP': 'NPR', 'MY': 'MYR', 'TH': 'THB', 'ID': 'IDR', 'PH': 'PHP',
    'VN': 'VND', 'KH': 'KHR', 'MM': 'MMK', 'MN': 'MNT',
    # Central Asia
    'RU': 'RUB', 'KZ': 'KZT', 'UZ': 'UZS', 'AZ': 'AZN', 'GE': 'GEL',
    'AM': 'AMD', 'KG': 'KGS', 'TJ': 'TJS', 'TM': 'TMT',
    # Oceania
    'AU': 'AUD', 'NZ': 'NZD',
    # Africa
    'ZA': 'ZAR', 'NG': 'NGN', 'KE': 'KES', 'EG': 'EGP', 'GH': 'GHS',
    'TZ': 'TZS', 'ET': 'ETB', 'SN': 'XOF', 'CI': 'XOF', 'CM': 'XAF',
    'DZ': 'DZD', 'MA': 'MAD', 'TN': 'TND', 'AO': 'AOA', 'MZ': 'MZN',
    'ZM': 'ZMW', 'ZW': 'ZWL', 'UG': 'UGX', 'RW': 'RWF', 'BW': 'BWP',
    'NA': 'NAD', 'MU': 'MUR', 'MG': 'MGA', 'CD': 'CDF',
    # Sanctioned / high-risk
    'KP': 'KPW', 'BY': 'BYR', 'SD': 'SDG', 'LY': 'LYD', 'AF': 'AFN',
    'SO': 'SOS', 'SS': 'SSP', 'CF': 'XAF', 'ML': 'XOF',
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
    # Very low risk — stable, transparent governance
    'CH': 0.05, 'NO': 0.05, 'DK': 0.05, 'SE': 0.05, 'FI': 0.05,
    'NZ': 0.05, 'LU': 0.08,
    'US': 0.1, 'GB': 0.1, 'DE': 0.1, 'FR': 0.15, 'JP': 0.1,
    'CA': 0.1, 'AU': 0.1, 'NL': 0.1, 'SG': 0.1, 'IS': 0.08,
    'IE': 0.1, 'AT': 0.1, 'MT': 0.15,
    # Low-moderate risk
    'KR': 0.2, 'HK': 0.2, 'EE': 0.15, 'LV': 0.2, 'LT': 0.2,
    'SK': 0.2, 'SI': 0.2, 'HR': 0.25, 'CY': 0.2, 'TW': 0.15,
    'ES': 0.2, 'PT': 0.2, 'BE': 0.15, 'IL': 0.25,
    # Moderate risk
    'CL': 0.3, 'UY': 0.3, 'BW': 0.3, 'MU': 0.25, 'AE': 0.3,
    'QA': 0.3, 'OM': 0.35, 'GR': 0.3, 'IT': 0.25, 'PL': 0.25,
    'CZ': 0.25, 'HU': 0.35, 'RO': 0.35, 'BG': 0.35, 'MY': 0.4,
    'NA': 0.4, 'TN': 0.45, 'JO': 0.4, 'TT': 0.4, 'MK': 0.45,
    'RS': 0.45, 'BA': 0.5, 'AL': 0.5, 'MD': 0.6,
    # Moderate-high risk
    'CN': 0.5, 'IN': 0.45, 'BR': 0.5, 'MX': 0.55, 'AR': 0.55,
    'ZA': 0.5, 'SA': 0.5, 'TH': 0.45, 'ID': 0.5, 'PH': 0.55,
    'VN': 0.55, 'KE': 0.55, 'RW': 0.4, 'GH': 0.5, 'SN': 0.55,
    'MA': 0.5, 'DZ': 0.6, 'EG': 0.6, 'CO': 0.5, 'PE': 0.45,
    'EC': 0.5, 'DO': 0.5, 'PA': 0.45, 'CR': 0.3, 'GT': 0.55,
    'HN': 0.6, 'SV': 0.55, 'NI': 0.6, 'JM': 0.5, 'BO': 0.55,
    'PY': 0.55, 'TR': 0.55, 'LB': 0.75, 'BH': 0.4, 'KW': 0.35,
    'LK': 0.55, 'NP': 0.6, 'KH': 0.65, 'MN': 0.55, 'CI': 0.6,
    'CM': 0.65, 'AO': 0.65, 'TZ': 0.55, 'ET': 0.65, 'UG': 0.65,
    'ZM': 0.6, 'MZ': 0.65, 'MG': 0.7, 'CD': 0.85, 'NG': 0.7,
    # High risk
    'UA': 0.6, 'AZ': 0.6, 'GE': 0.45, 'AM': 0.5, 'KZ': 0.65,
    'UZ': 0.7, 'KG': 0.65, 'TJ': 0.75, 'TM': 0.8,
    'PK': 0.7, 'BD': 0.65, 'HT': 0.8, 'ZW': 0.8, 'ML': 0.8,
    # Very high / sanctioned
    'RU': 0.85, 'BY': 0.9, 'MM': 0.8, 'AF': 0.95, 'IQ': 0.9,
    'SY': 0.95, 'IR': 0.95, 'KP': 0.99, 'CU': 0.85, 'VE': 0.85,
    'SD': 0.9, 'LY': 0.9, 'SO': 0.95, 'YE': 0.9,
    'CF': 0.9, 'SS': 0.9,
}
