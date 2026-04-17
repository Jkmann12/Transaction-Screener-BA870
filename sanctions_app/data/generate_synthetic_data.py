"""Synthetic transaction data generator.

Generates realistic international wire transfers following FATF typology patterns.
Transaction amounts follow Benford's Law for legitimate transactions,
with deliberate violations for suspicious transactions.
"""
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import SANCTIONED_COUNTRIES, COUNTRY_CURRENCIES, BENFORD_EXPECTED


# Fake company names for synthetic data
LEGITIMATE_COMPANIES = [
    'Global Trade Partners LLC', 'Pacific Rim Exports Inc', 'Atlantic Commerce Group',
    'Northern Star Trading', 'Meridian Financial Services', 'Apex International Corp',
    'Sterling Commerce Ltd', 'Continental Imports SA', 'Horizon Global Trade',
    'Summit Financial Group', 'Cascade Trading Company', 'Alpine Commercial Bank',
    'Riverdale Exports Ltd', 'Oakwood International', 'Greenfield Trading Co',
    'Westport Financial', 'Eastgate Commerce', 'Northfield Industries',
    'Southgate Exports', 'Midland Trading Group', 'Premier Finance Ltd',
    'Atlantic Bridge Corp', 'Pacific Gateway Inc', 'Global Nexus Trading',
    'InterTrade Solutions', 'CrossBorder Finance', 'Worldwide Commerce Ltd',
    'Universal Trading Co', 'International Exchange Corp', 'Global Dynamics Inc',
    'TechTrade International', 'DigitalFinance Corp', 'SmartTrade Solutions',
    'InnovatePay Ltd', 'FutureTrade Inc', 'NexGen Commerce',
    'MetroExports LLC', 'CityFinance Group', 'UrbanTrade Partners',
    'Coastal Commerce Corp', 'Harbor Finance Ltd', 'Bay Area Trading',
    'Mountain Commerce Inc', 'Valley Finance Group', 'Desert Trade Corp',
    'Prairie Financial Services', 'Forest Commerce Ltd', 'Island Trading Co',
    'Peninsula Finance Inc', 'Lakeside Commerce Group', 'Riverside Trading Partners',
    'Quantum Finance Corp', 'Synergy Trade Ltd', 'Nexus Commerce Inc',
    'Vector Financial Group', 'Matrix Trade Solutions', 'Vertex Commerce Corp',
    'Omega Finance Ltd', 'Alpha Trading Co', 'Beta Commerce Inc',
    'Gamma Financial Services', 'Delta Trade Group', 'Epsilon Exports Ltd',
    'Sigma Commerce Corp', 'Theta Finance Inc', 'Lambda Trading Ltd',
    'Kappa Commerce Group', 'Phi Financial Services', 'Chi Trade Partners',
    'Psi Commerce Corp', 'Mu Finance Ltd', 'Nu Trading Inc',
    'Xi Commerce Group', 'Upsilon Financial', 'Tau Trade Corp',
    'Rho Commerce Ltd', 'Pi Trading Solutions', 'Eta Finance Inc',
    'Iota Commerce Partners', 'Zeta Trade Group', 'Euro Commerce SA',
    'Asian Pacific Trade', 'Latin America Commerce', 'Nordic Finance Ltd',
    'Eastern European Trade', 'Western Commerce Corp', 'Southern Finance Inc'
]

SANCTIONED_ENTITIES = [
    'Tehran Trading Co', 'Iranian Export Corp', 'Pyongyang Finance Ltd',
    'Damascus Commercial Bank', 'Minsk State Enterprise', 'Havana Import Export',
    'Caracas Finance Group', 'Khartoum Trading Co', 'Tripoli Commerce Ltd',
    'Syrian Trade Corp', 'IRGC Trading Unit', 'NK Mining Corp',
    'Iranian Oil Services', 'Belarusian State Trader', 'Cuban Commerce SA',
    'Venezuelan State Import', 'Sudan Development Corp', 'Myanmar Resources Ltd',
    'Baghdad Finance Co', 'Kabul Trade Partners', 'Bamako Commerce Ltd',
    'Tehran Financial Services', 'Pyongyang Export Bureau', 'Damascus Finance Corp',
    'Havana State Trade', 'Caracas Oil Partners', 'Khartoum Finance Ltd'
]

# Countries weighted for normal vs sanctioned transactions
NORMAL_COUNTRIES = ['US', 'GB', 'DE', 'FR', 'JP', 'CA', 'AU', 'CH', 'SG', 'HK',
                    'CN', 'IN', 'BR', 'MX', 'KR', 'NL', 'SE', 'NO', 'DK', 'AE',
                    'SA', 'QA', 'TR', 'ZA', 'NG', 'EG', 'IL', 'PL', 'CZ', 'AR']
_raw_weights = np.array([0.12, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.04, 0.04, 0.04,
                         0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02,
                         0.02, 0.01, 0.02, 0.02, 0.02, 0.01, 0.02, 0.01, 0.01, 0.02])
NORMAL_WEIGHTS = _raw_weights / _raw_weights.sum()  # normalize to exactly 1.0

SANCTIONED_TX_COUNTRIES = ['IR', 'KP', 'SY', 'CU', 'RU', 'BY']

# All countries shown on the Risk Dashboard choropleth map — used to guarantee coverage
ALL_MAP_COUNTRIES = [
    # North America & Caribbean
    'US', 'CA', 'MX', 'GT', 'HN', 'SV', 'NI', 'CR', 'PA',
    'CU', 'DO', 'JM', 'TT', 'HT',
    # South America
    'BR', 'AR', 'CL', 'CO', 'PE', 'VE', 'EC', 'BO', 'PY', 'UY',
    # Western Europe
    'GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT', 'CH', 'SE',
    'NO', 'DK', 'FI', 'IE', 'PT', 'LU', 'MT', 'IS',
    # Central & Eastern Europe
    'PL', 'CZ', 'HU', 'RO', 'BG', 'HR', 'SK', 'SI', 'EE', 'LV',
    'LT', 'GR', 'CY', 'RS', 'AL', 'BA', 'MK', 'MD', 'UA', 'BY',
    # Middle East
    'AE', 'SA', 'QA', 'KW', 'BH', 'OM', 'JO', 'LB', 'IL', 'TR',
    'IR', 'IQ', 'SY', 'YE',
    # Asia-Pacific
    'JP', 'CN', 'HK', 'SG', 'KR', 'TW', 'IN', 'PK', 'BD', 'LK',
    'NP', 'MY', 'TH', 'ID', 'PH', 'VN', 'KH', 'MM', 'MN', 'AU', 'NZ',
    # Central Asia & Caucasus
    'RU', 'KZ', 'UZ', 'AZ', 'GE', 'AM', 'KG', 'TJ', 'TM', 'AF',
    # Africa
    'EG', 'DZ', 'MA', 'TN', 'LY', 'SD', 'NG', 'GH', 'SN', 'CI',
    'CM', 'ML', 'CF', 'KE', 'TZ', 'ET', 'UG', 'RW', 'SO', 'SS',
    'ZA', 'ZW', 'ZM', 'MZ', 'BW', 'NA', 'AO', 'MG', 'MU', 'CD',
    'KP',
]


def generate_benford_amount(min_val: float = 1000, max_val: float = 10_000_000) -> float:
    """Generate a random amount following Benford's Law distribution."""
    benford_probs = np.array([BENFORD_EXPECTED[d] for d in range(1, 10)], dtype=np.float64)
    benford_probs = benford_probs / benford_probs.sum()
    benford_probs[-1] += 1.0 - benford_probs.sum()  # fix any residual floating point error
    first_d = np.random.choice(range(1, 10), p=benford_probs)

    # Generate the magnitude (number of digits)
    log_min = np.log10(min_val)
    log_max = np.log10(max_val)
    log_val = np.random.uniform(np.log10(first_d), np.log10(first_d + 1))
    magnitude = np.random.randint(int(log_min), int(log_max) + 1)
    amount = (log_val + magnitude - len(str(first_d)) + 1)
    amount = 10 ** np.random.uniform(log_min, log_max)

    # Adjust to have correct first digit
    while int(str(f"{amount:.0f}")[0]) != first_d:
        amount = 10 ** np.random.uniform(log_min, log_max)

    return round(amount, 2)


def generate_suspicious_amount(min_val: float = 1000, max_val: float = 10_000_000) -> float:
    """Generate suspicious amount over-representing digits 5-9 (Benford violation)."""
    first_d = np.random.choice(range(5, 10))  # Over-represent 5-9
    log_min = np.log10(min_val)
    log_max = np.log10(max_val)
    amount = 10 ** np.random.uniform(log_min, log_max)
    while int(str(f"{amount:.0f}")[0]) != first_d:
        amount = 10 ** np.random.uniform(log_min, log_max)
    # FATF pattern: round numbers common in suspicious transactions
    if np.random.random() < 0.3:
        magnitude = 10 ** np.random.randint(3, 7)
        amount = round(amount / magnitude) * magnitude
    return round(amount, 2)


def generate_data(n: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate n synthetic wire transfer records.

    ~85% legitimate (Benford-conforming), ~15% suspicious (labeled is_sanctioned=1).
    Suspicious transactions: Benford violations, sanctioned countries/entities,
    currency mismatches, following FATF typology patterns.

    Args:
        n: Number of records to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with schema: transaction_id, date, sender_name, sender_country,
        receiver_name, receiver_country, amount, currency, is_sanctioned
    """
    np.random.seed(random_state)
    random.seed(random_state)

    n_suspicious = int(n * 0.15)
    n_normal = n - n_suspicious

    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    records = []

    # Generate normal transactions
    for i in range(n_normal):
        sender_country = np.random.choice(NORMAL_COUNTRIES, p=NORMAL_WEIGHTS)
        receiver_country = np.random.choice(NORMAL_COUNTRIES, p=NORMAL_WEIGHTS)
        amount = generate_benford_amount()
        currency = COUNTRY_CURRENCIES.get(receiver_country, 'USD')
        date = start_date + timedelta(days=np.random.randint(0, date_range))

        records.append({
            'transaction_id': f'TXN{i+1:07d}',
            'date': date.strftime('%Y-%m-%d'),
            'sender_name': random.choice(LEGITIMATE_COMPANIES),
            'sender_country': sender_country,
            'receiver_name': random.choice(LEGITIMATE_COMPANIES),
            'receiver_country': receiver_country,
            'amount': amount,
            'currency': currency,
            'is_sanctioned': 0
        })

    # Generate suspicious transactions
    for i in range(n_suspicious):
        # Mix of patterns from FATF typologies
        pattern = np.random.choice(['sanctioned_entity', 'sanctioned_country', 'both'], p=[0.3, 0.3, 0.4])
        sender_country = np.random.choice(NORMAL_COUNTRIES, p=NORMAL_WEIGHTS)

        if pattern in ('sanctioned_country', 'both'):
            receiver_country = np.random.choice(SANCTIONED_TX_COUNTRIES)
        else:
            receiver_country = np.random.choice(NORMAL_COUNTRIES, p=NORMAL_WEIGHTS)

        if pattern in ('sanctioned_entity', 'both'):
            receiver_name = random.choice(SANCTIONED_ENTITIES)
        else:
            receiver_name = random.choice(LEGITIMATE_COMPANIES)

        amount = generate_suspicious_amount()
        # Currency mismatch: use USD even when receiver is not in USD country (FATF pattern)
        currency = 'USD' if np.random.random() < 0.6 else COUNTRY_CURRENCIES.get(receiver_country, 'USD')
        date = start_date + timedelta(days=np.random.randint(0, date_range))

        records.append({
            'transaction_id': f'TXN{n_normal + i + 1:07d}',
            'date': date.strftime('%Y-%m-%d'),
            'sender_name': random.choice(LEGITIMATE_COMPANIES),
            'sender_country': sender_country,
            'receiver_name': receiver_name,
            'receiver_country': receiver_country,
            'amount': amount,
            'currency': currency,
            'is_sanctioned': 1
        })

    df = pd.DataFrame(records)

    # Guarantee at least MIN_PER_COUNTRY transactions for every map country
    MIN_PER_COUNTRY = 3
    covered = set(df['receiver_country'].unique())
    top_up = []
    for country in ALL_MAP_COUNTRIES:
        if country in covered:
            continue
        is_sanctioned = 1 if country in SANCTIONED_COUNTRIES else 0
        for j in range(MIN_PER_COUNTRY):
            sender_country = np.random.choice(NORMAL_COUNTRIES, p=NORMAL_WEIGHTS)
            if is_sanctioned:
                amount = generate_suspicious_amount()
                receiver_name = random.choice(SANCTIONED_ENTITIES)
                currency = 'USD' if np.random.random() < 0.6 else COUNTRY_CURRENCIES.get(country, 'USD')
            else:
                amount = generate_benford_amount()
                receiver_name = random.choice(LEGITIMATE_COMPANIES)
                currency = COUNTRY_CURRENCIES.get(country, 'USD')
            date = start_date + timedelta(days=np.random.randint(0, date_range))
            top_up.append({
                'transaction_id': f'TXN_CU{len(top_up)+1:05d}',
                'date': date.strftime('%Y-%m-%d'),
                'sender_name': random.choice(LEGITIMATE_COMPANIES),
                'sender_country': sender_country,
                'receiver_name': receiver_name,
                'receiver_country': country,
                'amount': amount,
                'currency': currency,
                'is_sanctioned': is_sanctioned,
            })

    if top_up:
        df = pd.concat([df, pd.DataFrame(top_up)], ignore_index=True)

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


if __name__ == '__main__':
    print("Generating synthetic transaction data...")
    df = generate_data(n=10000)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_transactions.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions ({df['is_sanctioned'].sum()} suspicious)")
    print(f"Saved to {output_path}")

    # Validate Benford's Law
    from utils.benford import benford_test
    legit = df[df['is_sanctioned'] == 0]['amount']
    result = benford_test(legit.tolist())
    if result:
        print(f"\nBenford test on legitimate transactions: chi2={result['chi2']:.2f}, p={result['p_value']:.4f}")
        print("Observed first digits:", result['observed'])
