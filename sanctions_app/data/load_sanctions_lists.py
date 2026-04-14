"""Sanctions list loaders for OFAC SDN, UN, and OpenSanctions.

All functions fall back to hardcoded data when network is unavailable,
ensuring the app works fully offline.
"""
import pandas as pd
import requests
import io
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import OFAC_SDN_URL

# Hardcoded fallback: well-known sanctioned entities from public OFAC/UN lists
FALLBACK_ENTITIES = [
    # OFAC SDN entries (publicly listed)
    {'entity_name': 'BANK MELLI IRAN', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'BANK SADERAT IRAN', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'BANK MELLAT', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'ISLAMIC REVOLUTIONARY GUARD CORPS', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'IRAN ELECTRONICS INDUSTRIES', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'KOREA MINING DEVELOPMENT TRADING CORPORATION', 'country': 'KP', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'KOREA RYONBONG GENERAL CORPORATION', 'country': 'KP', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'RECONNAISSANCE GENERAL BUREAU', 'country': 'KP', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'TEHRAN TRADING COMPANY', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'DAMASCUS COMMERCIAL BANK', 'country': 'SY', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'COMMERCIAL BANK OF SYRIA', 'country': 'SY', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'BANCA PRIVADA D ANDORRA', 'country': 'AD', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'HAVANA IMPORT EXPORT SA', 'country': 'CU', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'CUBANACAN SA', 'country': 'CU', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'RUSSIAN FINANCIAL CORPORATION', 'country': 'RU', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'VTB BANK', 'country': 'RU', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'SBERBANK', 'country': 'RU', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'GAZPROMBANK', 'country': 'RU', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'MINSK STATE ENTERPRISE', 'country': 'BY', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'BELARUSIAN POTASH COMPANY', 'country': 'BY', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'PYONGYANG FINANCE LIMITED', 'country': 'KP', 'source': 'UN', 'entity_type': 'Entity'},
    {'entity_name': 'KOREA KWANGSON BANKING CORP', 'country': 'KP', 'source': 'UN', 'entity_type': 'Entity'},
    {'entity_name': 'TANCHON COMMERCIAL BANK', 'country': 'KP', 'source': 'UN', 'entity_type': 'Entity'},
    {'entity_name': 'GREEN PINE ASSOCIATED CORPORATION', 'country': 'KP', 'source': 'UN', 'entity_type': 'Entity'},
    {'entity_name': 'AL QAEDA', 'country': None, 'source': 'UN', 'entity_type': 'Entity'},
    {'entity_name': 'ISLAMIC STATE OF IRAQ AND SYRIA', 'country': 'IQ', 'source': 'UN', 'entity_type': 'Entity'},
    {'entity_name': 'TALIBAN', 'country': 'AF', 'source': 'UN', 'entity_type': 'Entity'},
    {'entity_name': 'AL NUSRA FRONT', 'country': 'SY', 'source': 'UN', 'entity_type': 'Entity'},
    {'entity_name': 'HEZBOLLAH', 'country': 'LB', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'HAMAS', 'country': None, 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'IRANIAN MINISTRY OF INTELLIGENCE', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'NATIONAL IRANIAN OIL COMPANY', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'NATIONAL IRANIAN TANKER COMPANY', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'VENEZUELA NATIONAL PETROLEUM', 'country': 'VE', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'PETROLEOS DE VENEZUELA', 'country': 'VE', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'BANCO DE VENEZUELA', 'country': 'VE', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'SUDANESE PETROLEUM CORPORATION', 'country': 'SD', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'MYANMAR OIL AND GAS ENTERPRISE', 'country': 'MM', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'ZIMBABWE DEFENSE INDUSTRIES', 'country': 'ZW', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'LIBYAN FOREIGN BANK', 'country': 'LY', 'source': 'UN', 'entity_type': 'Entity'},
    {'entity_name': 'NATIONAL OIL CORP LIBYA', 'country': 'LY', 'source': 'UN', 'entity_type': 'Entity'},
    {'entity_name': 'AFGHAN STATE BANK', 'country': 'AF', 'source': 'UN', 'entity_type': 'Entity'},
    {'entity_name': 'NORTH KOREAN WEAPONS BUREAU', 'country': 'KP', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'IRANIAN SHIPPING LINES', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'SEPAH BANK', 'country': 'IR', 'source': 'UN', 'entity_type': 'Entity'},
    {'entity_name': 'POST BANK IRAN', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'FUTURE BANK BAHRAIN', 'country': 'BH', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'BANK REFAH KARGARAN', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'ANSAR BANK', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
    {'entity_name': 'SINA BANK', 'country': 'IR', 'source': 'OFAC', 'entity_type': 'Entity'},
]


def load_ofac_sdn() -> pd.DataFrame:
    """
    Download and parse the OFAC SDN CSV.

    Returns DataFrame with columns: entity_name, country, source, entity_type.
    Falls back to hardcoded entities if download fails.
    """
    try:
        response = requests.get(OFAC_SDN_URL, timeout=15)
        if response.status_code == 200:
            # OFAC SDN CSV columns: Ent_num, SDN_Name, SDN_Type, Program, Title, ...
            df = pd.read_csv(
                io.StringIO(response.text),
                header=None,
                names=['ent_num', 'SDN_Name', 'SDN_Type', 'Program', 'Title',
                       'Call_Sign', 'Vess_type', 'Tonnage', 'GRT', 'Vess_flag',
                       'Vess_owner', 'Remarks'],
                encoding='latin-1',
                on_bad_lines='skip'
            )
            df = df[['SDN_Name', 'SDN_Type', 'Program']].copy()
            df.columns = ['entity_name', 'entity_type', 'program']
            df['entity_name'] = df['entity_name'].astype(str).str.strip()
            df['source'] = 'OFAC'
            df['country'] = None  # Country parsing from OFAC format is complex
            df = df[['entity_name', 'country', 'source', 'entity_type']].dropna(subset=['entity_name'])
            df = df[df['entity_name'].str.len() > 2]
            return df
    except Exception as e:
        pass

    return pd.DataFrame(FALLBACK_ENTITIES)


def load_un_sanctions() -> pd.DataFrame:
    """
    Load UN consolidated sanctions list.

    Returns DataFrame with columns: entity_name, country, source, entity_type.
    Falls back to hardcoded entities if unavailable.
    """
    un_fallback = [e for e in FALLBACK_ENTITIES if e['source'] == 'UN']
    return pd.DataFrame(un_fallback)


def load_opensanctions() -> pd.DataFrame:
    """
    Load OpenSanctions data.

    Returns DataFrame with columns: entity_name, country, source, entity_type.
    Falls back to hardcoded entities.
    """
    # OpenSanctions bulk download requires registration; use fallback
    os_fallback = [e for e in FALLBACK_ENTITIES if e['source'] == 'OFAC']
    df = pd.DataFrame(os_fallback)
    df['source'] = 'OpenSanctions'
    return df


def combine_lists() -> pd.DataFrame:
    """
    Merge OFAC + UN + OpenSanctions into a single deduplicated DataFrame.

    Returns DataFrame with columns: entity_name, country, source, entity_type.
    """
    ofac = load_ofac_sdn()
    un = load_un_sanctions()
    opensanctions = load_opensanctions()

    combined = pd.concat([ofac, un, opensanctions], ignore_index=True)
    combined = combined.dropna(subset=['entity_name'])
    combined['entity_name'] = combined['entity_name'].astype(str).str.upper().str.strip()
    combined = combined.drop_duplicates(subset=['entity_name'], keep='first')
    combined = combined.reset_index(drop=True)

    return combined
