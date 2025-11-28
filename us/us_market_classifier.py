"""
US Stock Market State Classifier

Classifies stocks into 31 market states (A-F groups + OTHER) based on 11 conditions.
Based on Korean model structure with US market characteristics.

Key Features:
- 31 predefined market states organized in 6 groups (A-F)
- 11-condition classification logic
- 24-hour caching support
- Batch processing for multiple stocks
- Classification statistics

Architecture:
- Class-based structure (matches Korean model pattern)
- Priority-based classification (F → E → A → B → C → D → OTHER)
- Detailed state information with representative stocks
- Factor weights per market state

Usage:
    classifier = USMarketClassifier()
    market_state = classifier.classify(conditions)

    # Batch processing
    results = classifier.classify_batch(conditions_dict)

    # Statistics
    stats = classifier.get_classification_stats(results)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import Counter

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===========================
# Market State Definitions
# ===========================

# Market State IDs (31 states + OTHER)
MARKET_STATES = {
    # A Group: NASDAQ Tech (7)
    'A1': 'A1-NASDAQ-Mega-Tech-HighGrowth',
    'A2': 'A2-NASDAQ-Large-Tech-HyperGrowth',
    'A3': 'A3-NASDAQ-Large-Tech-Profitable',
    'A4': 'A4-NASDAQ-Mid-Tech-HighGrowth',
    'A5': 'A5-NASDAQ-Small-Tech-Emerging',
    'A6': 'A6-NASDAQ-Large-CommServices-Growth',
    'A7': 'A7-NASDAQ-Mega-Tech-ModerateGrowth',

    # B Group: NYSE Value (8)
    'B1': 'B1-NYSE-Mega-Financials-Value',
    'B2': 'B2-NYSE-Large-FinancialServices-Quality',
    'B3': 'B3-NYSE-Mega-Energy-Defensive',
    'B4': 'B4-NYSE-Large-Industrials-Cyclical',
    'B5': 'B5-NYSE-Mega-ConsumerDefensive-Stable',
    'B6': 'B6-NYSE-Large-ConsumerCyclical-Growth',
    'B7': 'B7-NYSE-Large-Materials-Value',
    'B8': 'B8-NYSE-Mega-Utilities-Income',

    # C Group: Healthcare/Biotech (4)
    'C1': 'C1-NYSE-Mega-Healthcare-Quality',
    'C2': 'C2-NASDAQ-Large-Biotech-Growth',
    'C3': 'C3-NASDAQ-Mid-Biotech-Speculative',
    'C4': 'C4-NYSE-Large-HealthcareServices-Stable',

    # D Group: Mid/Small Growth (5)
    'D1': 'D1-NASDAQ-Mid-Growth-Profitable',
    'D2': 'D2-NASDAQ-Small-Growth-Unprofitable',
    'D3': 'D3-NYSE-Mid-Industrial-Recovery',
    'D4': 'D4-NASDAQ-Mid-ConsumerCyclical-Disruptor',
    'D5': 'D5-NYSE-Small-RealEstate-Value',

    # E Group: Defensive (4)
    'E1': 'E1-Utilities-Dividend',
    'E2': 'E2-ConsumerDefensive-Recession',
    'E3': 'E3-Healthcare-Defensive',
    'E4': 'E4-Telecom-Yield',

    # F Group: Special Situations (3)
    'F1': 'F1-HighVolatility-Momentum',
    'F2': 'F2-Micro-Speculative',
    'F3': 'F3-StrongAnalyst-Upgrade',

    # Other
    'OTHER': 'OTHER'
}


# Market State Factor Weights
MARKET_STATE_FACTOR_WEIGHTS = {
    # A Group: NASDAQ Tech (7)
    'A1-NASDAQ-Mega-Tech-HighGrowth': {
        'value': 0.10, 'quality': 0.30, 'momentum': 0.30, 'growth': 0.30
    },
    'A2-NASDAQ-Large-Tech-HyperGrowth': {
        'value': 0.05, 'quality': 0.20, 'momentum': 0.40, 'growth': 0.35
    },
    'A3-NASDAQ-Large-Tech-Profitable': {
        'value': 0.25, 'quality': 0.40, 'momentum': 0.20, 'growth': 0.15
    },
    'A4-NASDAQ-Mid-Tech-HighGrowth': {
        'value': 0.15, 'quality': 0.25, 'momentum': 0.30, 'growth': 0.30
    },
    'A5-NASDAQ-Small-Tech-Emerging': {
        'value': 0.10, 'quality': 0.15, 'momentum': 0.35, 'growth': 0.40
    },
    'A6-NASDAQ-Large-CommServices-Growth': {
        'value': 0.15, 'quality': 0.30, 'momentum': 0.30, 'growth': 0.25
    },
    'A7-NASDAQ-Mega-Tech-ModerateGrowth': {
        'value': 0.15, 'quality': 0.35, 'momentum': 0.25, 'growth': 0.25
    },

    # B Group: NYSE Value (8)
    'B1-NYSE-Mega-Financials-Value': {
        'value': 0.40, 'quality': 0.30, 'momentum': 0.15, 'growth': 0.15
    },
    'B2-NYSE-Large-FinancialServices-Quality': {
        'value': 0.30, 'quality': 0.40, 'momentum': 0.20, 'growth': 0.10
    },
    'B3-NYSE-Mega-Energy-Defensive': {
        'value': 0.45, 'quality': 0.25, 'momentum': 0.15, 'growth': 0.15
    },
    'B4-NYSE-Large-Industrials-Cyclical': {
        'value': 0.30, 'quality': 0.30, 'momentum': 0.25, 'growth': 0.15
    },
    'B5-NYSE-Mega-ConsumerDefensive-Stable': {
        'value': 0.35, 'quality': 0.40, 'momentum': 0.10, 'growth': 0.15
    },
    'B6-NYSE-Large-ConsumerCyclical-Growth': {
        'value': 0.20, 'quality': 0.30, 'momentum': 0.25, 'growth': 0.25
    },
    'B7-NYSE-Large-Materials-Value': {
        'value': 0.35, 'quality': 0.25, 'momentum': 0.20, 'growth': 0.20
    },
    'B8-NYSE-Mega-Utilities-Income': {
        'value': 0.45, 'quality': 0.35, 'momentum': 0.05, 'growth': 0.15
    },

    # C Group: Healthcare/Biotech (4)
    'C1-NYSE-Mega-Healthcare-Quality': {
        'value': 0.20, 'quality': 0.40, 'momentum': 0.20, 'growth': 0.20
    },
    'C2-NASDAQ-Large-Biotech-Growth': {
        'value': 0.15, 'quality': 0.30, 'momentum': 0.30, 'growth': 0.25
    },
    'C3-NASDAQ-Mid-Biotech-Speculative': {
        'value': 0.05, 'quality': 0.10, 'momentum': 0.45, 'growth': 0.40
    },
    'C4-NYSE-Large-HealthcareServices-Stable': {
        'value': 0.30, 'quality': 0.35, 'momentum': 0.15, 'growth': 0.20
    },

    # D Group: Mid/Small Growth (5)
    'D1-NASDAQ-Mid-Growth-Profitable': {
        'value': 0.15, 'quality': 0.30, 'momentum': 0.30, 'growth': 0.25
    },
    'D2-NASDAQ-Small-Growth-Unprofitable': {
        'value': 0.05, 'quality': 0.10, 'momentum': 0.45, 'growth': 0.40
    },
    'D3-NYSE-Mid-Industrial-Recovery': {
        'value': 0.35, 'quality': 0.20, 'momentum': 0.25, 'growth': 0.20
    },
    'D4-NASDAQ-Mid-ConsumerCyclical-Disruptor': {
        'value': 0.10, 'quality': 0.25, 'momentum': 0.35, 'growth': 0.30
    },
    'D5-NYSE-Small-RealEstate-Value': {
        'value': 0.40, 'quality': 0.30, 'momentum': 0.10, 'growth': 0.20
    },

    # E Group: Defensive (4)
    'E1-Utilities-Dividend': {
        'value': 0.50, 'quality': 0.35, 'momentum': 0.05, 'growth': 0.10
    },
    'E2-ConsumerDefensive-Recession': {
        'value': 0.40, 'quality': 0.45, 'momentum': 0.05, 'growth': 0.10
    },
    'E3-Healthcare-Defensive': {
        'value': 0.30, 'quality': 0.45, 'momentum': 0.10, 'growth': 0.15
    },
    'E4-Telecom-Yield': {
        'value': 0.45, 'quality': 0.30, 'momentum': 0.10, 'growth': 0.15
    },

    # F Group: Special Situations (3)
    'F1-HighVolatility-Momentum': {
        'value': 0.05, 'quality': 0.10, 'momentum': 0.60, 'growth': 0.25
    },
    'F2-Micro-Speculative': {
        'value': 0.20, 'quality': 0.15, 'momentum': 0.40, 'growth': 0.25
    },
    'F3-StrongAnalyst-Upgrade': {
        'value': 0.10, 'quality': 0.20, 'momentum': 0.45, 'growth': 0.25
    },

    # Other
    'OTHER': {
        'value': 0.25, 'quality': 0.25, 'momentum': 0.25, 'growth': 0.25
    }
}


# Market State Information
MARKET_STATE_INFO = {
    # A Group
    'A1-NASDAQ-Mega-Tech-HighGrowth': {
        'group': 'A',
        'description': 'NASDAQ Mega-cap Tech High Growth',
        'representative_stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'characteristics': 'Market cap $200B+, 15%+ annual growth, high profitability'
    },
    'A2-NASDAQ-Large-Tech-HyperGrowth': {
        'group': 'A',
        'description': 'NASDAQ Large-cap Tech Hyper Growth',
        'representative_stocks': ['NVDA', 'AMD', 'TSLA', 'MSTR'],
        'characteristics': '30%+ growth, high volatility, AI/cloud/semiconductor'
    },
    'A3-NASDAQ-Large-Tech-Profitable': {
        'group': 'A',
        'description': 'NASDAQ Large-cap Tech Profitable',
        'representative_stocks': ['ORCL', 'INTC', 'CSCO', 'IBM'],
        'characteristics': 'Mature tech, stable profitability, dividends'
    },
    'A4-NASDAQ-Mid-Tech-HighGrowth': {
        'group': 'A',
        'description': 'NASDAQ Mid-cap Tech High Growth',
        'representative_stocks': ['SNOW', 'CRWD', 'PANW', 'NOW'],
        'characteristics': '$2B-$10B, SaaS/cybersecurity, high growth potential'
    },
    'A5-NASDAQ-Small-Tech-Emerging': {
        'group': 'A',
        'description': 'NASDAQ Small-cap Tech Emerging',
        'representative_stocks': ['Small emerging tech companies'],
        'characteristics': '$300M-$2B, high risk/return, low liquidity'
    },
    'A6-NASDAQ-Large-CommServices-Growth': {
        'group': 'A',
        'description': 'NASDAQ Large Communication Services Growth',
        'representative_stocks': ['META', 'GOOGL', 'NFLX'],
        'characteristics': 'Platform business, advertising revenue, network effects'
    },
    'A7-NASDAQ-Mega-Tech-ModerateGrowth': {
        'group': 'A',
        'description': 'NASDAQ Mega-cap Tech Moderate Growth',
        'representative_stocks': ['AAPL', 'CSCO', 'ADBE', 'QCOM'],
        'characteristics': 'Mature mega-cap tech, 5-15% growth, high profitability, stable cash flow, dividend payers'
    },

    # B Group
    'B1-NYSE-Mega-Financials-Value': {
        'group': 'B',
        'description': 'NYSE Mega Financials Value',
        'representative_stocks': ['JPM', 'BAC', 'WFC', 'C'],
        'characteristics': 'Large banks, interest rate sensitive, high dividends'
    },
    'B2-NYSE-Large-FinancialServices-Quality': {
        'group': 'B',
        'description': 'NYSE Large Financial Services Quality',
        'representative_stocks': ['BLK', 'GS', 'MS', 'SPGI'],
        'characteristics': 'Asset managers, investment banks, high profitability'
    },
    'B3-NYSE-Mega-Energy-Defensive': {
        'group': 'B',
        'description': 'NYSE Mega Energy Defensive',
        'representative_stocks': ['XOM', 'CVX', 'COP'],
        'characteristics': 'Oil/gas majors, inflation hedge, high dividends'
    },
    'B4-NYSE-Large-Industrials-Cyclical': {
        'group': 'B',
        'description': 'NYSE Large Industrials Cyclical',
        'representative_stocks': ['CAT', 'BA', 'GE', 'HON'],
        'characteristics': 'Industrial conglomerates, economic cycle sensitive'
    },
    'B5-NYSE-Mega-ConsumerDefensive-Stable': {
        'group': 'B',
        'description': 'NYSE Mega Consumer Defensive Stable',
        'representative_stocks': ['WMT', 'PG', 'KO', 'PEP'],
        'characteristics': 'Consumer staples, recession-resistant, low volatility'
    },
    'B6-NYSE-Large-ConsumerCyclical-Growth': {
        'group': 'B',
        'description': 'NYSE Large Consumer Cyclical Growth',
        'representative_stocks': ['HD', 'NKE', 'SBUX', 'MCD'],
        'characteristics': 'Discretionary goods, economic expansion beneficiary'
    },
    'B7-NYSE-Large-Materials-Value': {
        'group': 'B',
        'description': 'NYSE Large Materials Value',
        'representative_stocks': ['LIN', 'APD', 'FCX'],
        'characteristics': 'Materials/chemicals, commodity price sensitive'
    },
    'B8-NYSE-Mega-Utilities-Income': {
        'group': 'B',
        'description': 'NYSE Mega Utilities Income',
        'representative_stocks': ['NEE', 'DUK', 'SO'],
        'characteristics': 'Utilities, lowest volatility, high dividends'
    },

    # C Group
    'C1-NYSE-Mega-Healthcare-Quality': {
        'group': 'C',
        'description': 'NYSE Mega Healthcare Quality',
        'representative_stocks': ['UNH', 'JNJ', 'LLY', 'ABBV'],
        'characteristics': 'Large pharma/healthcare, stable growth, high margins'
    },
    'C2-NASDAQ-Large-Biotech-Growth': {
        'group': 'C',
        'description': 'NASDAQ Large Biotech Growth',
        'representative_stocks': ['AMGN', 'GILD', 'REGN', 'VRTX'],
        'characteristics': 'Large biotech, drug pipeline, FDA approval risk'
    },
    'C3-NASDAQ-Mid-Biotech-Speculative': {
        'group': 'C',
        'description': 'NASDAQ Mid Biotech Speculative',
        'representative_stocks': ['Mid-cap clinical stage biotech'],
        'characteristics': 'Clinical stage, very high risk/return, unprofitable'
    },
    'C4-NYSE-Large-HealthcareServices-Stable': {
        'group': 'C',
        'description': 'NYSE Large Healthcare Services Stable',
        'representative_stocks': ['CVS', 'CI', 'HUM'],
        'characteristics': 'Health insurance/services, stable revenue, defensive'
    },

    # D Group
    'D1-NASDAQ-Mid-Growth-Profitable': {
        'group': 'D',
        'description': 'NASDAQ Mid Growth Profitable',
        'representative_stocks': ['Various mid-cap growth stocks'],
        'characteristics': 'Mid-cap high growth, profitable, expansion stage'
    },
    'D2-NASDAQ-Small-Growth-Unprofitable': {
        'group': 'D',
        'description': 'NASDAQ Small Growth Unprofitable',
        'representative_stocks': ['Small growth stocks (unprofitable)'],
        'characteristics': 'Small-cap high growth, unprofitable, high risk'
    },
    'D3-NYSE-Mid-Industrial-Recovery': {
        'group': 'D',
        'description': 'NYSE Mid Industrial Recovery',
        'representative_stocks': ['Mid-cap industrials in recovery'],
        'characteristics': 'Mid-cap industrials, economic cycle beneficiary'
    },
    'D4-NASDAQ-Mid-ConsumerCyclical-Disruptor': {
        'group': 'D',
        'description': 'NASDAQ Mid Consumer Cyclical Disruptor',
        'representative_stocks': ['UBER', 'DASH', 'ABNB'],
        'characteristics': 'Platform business, industry disruptors, network effects'
    },
    'D5-NYSE-Small-RealEstate-Value': {
        'group': 'D',
        'description': 'NYSE Small Real Estate Value',
        'representative_stocks': ['Small REITs'],
        'characteristics': 'Small REITs, high dividend yield, interest rate sensitive'
    },

    # E Group
    'E1-Utilities-Dividend': {
        'group': 'E',
        'description': 'Utilities Dividend',
        'representative_stocks': ['Utility companies'],
        'characteristics': 'Recession defense, 4-5% dividends, lowest volatility'
    },
    'E2-ConsumerDefensive-Recession': {
        'group': 'E',
        'description': 'Consumer Defensive Recession',
        'representative_stocks': ['Consumer staples'],
        'characteristics': 'Best recession defense, stable demand, low beta'
    },
    'E3-Healthcare-Defensive': {
        'group': 'E',
        'description': 'Healthcare Defensive',
        'representative_stocks': ['Large pharma (defensive)'],
        'characteristics': 'Healthcare defense, inelastic demand, high margins'
    },
    'E4-Telecom-Yield': {
        'group': 'E',
        'description': 'Telecom Yield',
        'representative_stocks': ['VZ', 'T'],
        'characteristics': 'Telecom majors, 5-7% dividends, mature industry'
    },

    # F Group
    'F1-HighVolatility-Momentum': {
        'group': 'F',
        'description': 'High Volatility Momentum',
        'representative_stocks': ['High volatility momentum stocks'],
        'characteristics': 'Extreme volatility, short-term trading, news-sensitive'
    },
    'F2-Micro-Speculative': {
        'group': 'F',
        'description': 'Micro Cap Speculative',
        'representative_stocks': ['Micro-cap stocks'],
        'characteristics': 'Micro-cap $50M-$300M, very high risk, low liquidity'
    },
    'F3-StrongAnalyst-Upgrade': {
        'group': 'F',
        'description': 'Strong Analyst Upgrade',
        'representative_stocks': ['Stocks with strong analyst upgrades'],
        'characteristics': 'Strong positive signals, short-term momentum, event-driven'
    },

    # Other
    'OTHER': {
        'group': 'OTHER',
        'description': 'Other / Unclassified',
        'representative_stocks': [],
        'characteristics': 'Does not match any specific state'
    }
}


# ===========================
# USMarketClassifier Class
# ===========================

class USMarketClassifier:
    """
    US Stock Market State Classifier

    Classifies stocks into 31 market states based on 11 conditions:
    1. Exchange (NYSE/NASDAQ)
    2. Market Cap (MEGA/LARGE/MEDIUM/SMALL/MICRO)
    3. Liquidity (HIGH/MEDIUM/LOW)
    4. Economic Cycle (EXPANSION/RECOVERY/SLOWDOWN/RECESSION/NEUTRAL)
    5. Sector (11 GICS sectors)
    6. Volatility (VERY_HIGH/HIGH/MEDIUM/LOW)
    7. Growth Profile (HYPER_GROWTH/HIGH_GROWTH/MODERATE_GROWTH/SLOW_GROWTH/NEGATIVE_GROWTH)
    8. Profitability (HIGH_PROFIT/PROFITABLE/BREAKEVEN/LOSS_MAKING)
    9. Options Positioning (BULLISH/NEUTRAL/BEARISH)
    10. Analyst Momentum (STRONG_UPGRADE/UPGRADE/NEUTRAL/DOWNGRADE/STRONG_DOWNGRADE)
    11. Market Sentiment (OVERHEATED/GREED/NEUTRAL/FEAR/PANIC)

    Market States:
    - A Group (7): NASDAQ Tech
    - B Group (8): NYSE Value
    - C Group (4): Healthcare/Biotech
    - D Group (5): Mid/Small Growth
    - E Group (4): Defensive
    - F Group (3): Special Situations
    - OTHER: Unclassified

    Classification Priority: F → E → A → B → C → D → OTHER
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize US market classifier

        Args:
            cache_dir: Directory for cache file (default: us/)
        """
        if cache_dir is None:
            cache_dir = os.path.dirname(__file__)

        self.cache_file = os.path.join(cache_dir, 'us_market_classification_cache.json')
        self.cache_duration = 86400  # 24 hours in seconds

        logger.info("USMarketClassifier initialized")

    def classify(self, conditions: Dict) -> str:
        """
        Classify stock into one of 31 market states based on 11 conditions

        Priority:
        1. F Group (Special Situations) - highest priority
        2. E Group (Defensive) - recession/defensive scenarios
        3. A, B, C, D Groups (Core) - main classification logic
        4. OTHER (Unclassified)

        Parameters:
            conditions: Dictionary containing all 11 condition values

        Returns:
            str: Market state ID (e.g., 'A1-NASDAQ-Mega-Tech-HighGrowth')
        """

        # Extract condition values
        exchange = conditions.get('exchange')
        market_cap_category = conditions.get('market_cap_category')
        sector = conditions.get('sector')
        growth_profile = conditions.get('growth_profile')
        profitability = conditions.get('profitability')
        volatility = conditions.get('volatility')
        economic_cycle = conditions.get('economic_cycle')
        liquidity_level = conditions.get('liquidity_level')
        options_positioning = conditions.get('options_positioning')
        analyst_momentum = conditions.get('analyst_momentum')
        market_sentiment = conditions.get('market_sentiment')

        logger.debug(f"Classifying: Exchange={exchange}, Cap={market_cap_category}, Sector={sector}, Growth={growth_profile}")

        # ===== PRIORITY 1: F GROUP (Special Situations) =====

        # F2: Micro-cap (unconditional)
        if market_cap_category == 'MICRO_CAP':
            logger.info("Classified as F2: Micro-cap speculative")
            return MARKET_STATES['F2']

        # F1: Very high volatility + momentum
        if volatility == 'VERY_HIGH' and options_positioning in ['BULLISH', 'BEARISH']:
            logger.info("Classified as F1: High volatility momentum")
            return MARKET_STATES['F1']

        # F3: Strong analyst upgrade + bullish signals (exclude MEGA_CAP)
        if (analyst_momentum == 'STRONG_POSITIVE' and
            market_sentiment == 'BULLISH' and
            options_positioning == 'BULLISH' and
            market_cap_category in ['LARGE_CAP', 'MID_CAP', 'SMALL_CAP']):
            logger.info("Classified as F3: Strong analyst upgrade")
            return MARKET_STATES['F3']

        # ===== PRIORITY 2: E GROUP (Defensive) =====

        # E1: Utilities in recession/tightening
        if (sector == 'UTILITIES' and
            economic_cycle in ['RECESSION_RISK', 'TIGHTENING'] and
            volatility == 'LOW'):
            logger.info("Classified as E1: Utilities dividend")
            return MARKET_STATES['E1']

        # E2: Consumer defensive in recession
        if (sector == 'CONSUMER DEFENSIVE' and
            economic_cycle == 'RECESSION_RISK' and
            market_cap_category == 'MEGA_CAP'):
            logger.info("Classified as E2: Consumer defensive recession")
            return MARKET_STATES['E2']

        # E3: Healthcare defensive in recession
        if (sector == 'HEALTHCARE' and
            economic_cycle in ['RECESSION_RISK', 'TIGHTENING'] and
            market_cap_category in ['LARGE_CAP', 'MEGA_CAP'] and
            profitability == 'HIGHLY_PROFITABLE'):
            logger.info("Classified as E3: Healthcare defensive")
            return MARKET_STATES['E3']

        # E4: Telecom yield
        if (sector == 'COMMUNICATION SERVICES' and
            exchange == 'NYSE' and
            growth_profile in ['SLOW_GROWTH', 'DECLINING'] and
            market_cap_category == 'LARGE_CAP'):
            logger.info("Classified as E4: Telecom yield")
            return MARKET_STATES['E4']

        # ===== PRIORITY 3: A GROUP (NASDAQ Tech) =====

        if exchange == 'NASDAQ' and sector == 'TECHNOLOGY':
            # A1: NASDAQ Mega Tech High Growth
            if (market_cap_category == 'MEGA_CAP' and
                growth_profile in ['HIGH_GROWTH', 'HYPER_GROWTH'] and
                profitability in ['HIGHLY_PROFITABLE', 'PROFITABLE']):
                logger.info("Classified as A1: NASDAQ Mega Tech High Growth")
                return MARKET_STATES['A1']

            # A2: NASDAQ Large Tech Hyper Growth
            if (market_cap_category == 'LARGE_CAP' and
                growth_profile == 'HYPER_GROWTH' and
                volatility in ['HIGH', 'VERY_HIGH']):
                logger.info("Classified as A2: NASDAQ Large Tech Hyper Growth")
                return MARKET_STATES['A2']

            # A3: NASDAQ Large Tech Profitable
            if (market_cap_category == 'LARGE_CAP' and
                growth_profile in ['MODERATE_GROWTH', 'SLOW_GROWTH'] and
                profitability == 'HIGHLY_PROFITABLE'):
                logger.info("Classified as A3: NASDAQ Large Tech Profitable")
                return MARKET_STATES['A3']

            # A4: NASDAQ Mid Tech High Growth
            if (market_cap_category == 'MID_CAP' and
                growth_profile == 'HIGH_GROWTH' and
                profitability in ['PROFITABLE', 'MODERATELY_PROFITABLE']):
                logger.info("Classified as A4: NASDAQ Mid Tech High Growth")
                return MARKET_STATES['A4']

            # A5: NASDAQ Small Tech Emerging
            if (market_cap_category == 'SMALL_CAP' and
                growth_profile in ['HYPER_GROWTH', 'HIGH_GROWTH']):
                logger.info("Classified as A5: NASDAQ Small Tech Emerging")
                return MARKET_STATES['A5']

            # A7: NASDAQ Mega Tech Moderate Growth
            if (market_cap_category == 'MEGA_CAP' and
                growth_profile == 'MODERATE_GROWTH' and
                profitability == 'HIGHLY_PROFITABLE'):
                logger.info("Classified as A7: NASDAQ Mega Tech Moderate Growth")
                return MARKET_STATES['A7']

        # A6: NASDAQ Communication Services Growth
        if (exchange == 'NASDAQ' and
            sector == 'COMMUNICATION SERVICES' and
            market_cap_category in ['LARGE_CAP', 'MEGA_CAP'] and
            growth_profile == 'HIGH_GROWTH'):
            logger.info("Classified as A6: NASDAQ Communication Services Growth")
            return MARKET_STATES['A6']

        # ===== PRIORITY 4: B GROUP (NYSE Value) =====

        if exchange == 'NYSE':
            # B1: NYSE Mega Financials Value
            if (sector in ['FINANCIALS', 'FINANCIAL SERVICES'] and
                market_cap_category in ['MEGA_CAP', 'LARGE_CAP'] and
                growth_profile in ['SLOW_GROWTH', 'MODERATE_GROWTH', 'DECLINING']):
                logger.info("Classified as B1: NYSE Mega Financials Value")
                return MARKET_STATES['B1']

            # B2: NYSE Large Financial Services Quality
            if (sector == 'FINANCIAL SERVICES' and
                market_cap_category in ['LARGE_CAP', 'MEGA_CAP'] and
                profitability in ['HIGHLY_PROFITABLE', 'PROFITABLE']):
                logger.info("Classified as B2: NYSE Large Financial Services Quality")
                return MARKET_STATES['B2']

            # B3: NYSE Mega Energy Defensive
            if (sector == 'ENERGY' and
                market_cap_category in ['MEGA_CAP', 'LARGE_CAP'] and
                growth_profile in ['SLOW_GROWTH', 'DECLINING']):
                logger.info("Classified as B3: NYSE Mega Energy Defensive")
                return MARKET_STATES['B3']

            # B4: NYSE Large Industrials Cyclical
            if (sector == 'INDUSTRIALS' and
                market_cap_category == 'LARGE_CAP' and
                economic_cycle in ['EXPANSION', 'NEUTRAL']):
                logger.info("Classified as B4: NYSE Large Industrials Cyclical")
                return MARKET_STATES['B4']

            # B5: NYSE Mega Consumer Defensive Stable
            if (sector == 'CONSUMER DEFENSIVE' and
                market_cap_category == 'MEGA_CAP' and
                profitability == 'HIGHLY_PROFITABLE'):
                logger.info("Classified as B5: NYSE Mega Consumer Defensive Stable")
                return MARKET_STATES['B5']

            # B6: NYSE Large Consumer Cyclical Growth
            if (sector == 'CONSUMER CYCLICAL' and
                market_cap_category in ['LARGE_CAP', 'MEGA_CAP'] and
                growth_profile in ['MODERATE_GROWTH', 'HIGH_GROWTH']):
                logger.info("Classified as B6: NYSE Large Consumer Cyclical Growth")
                return MARKET_STATES['B6']

            # B7: NYSE Large Materials Value
            if (sector == 'BASIC MATERIALS' and
                market_cap_category == 'LARGE_CAP'):
                logger.info("Classified as B7: NYSE Large Materials Value")
                return MARKET_STATES['B7']

            # B8: NYSE Mega Utilities Income
            if (sector == 'UTILITIES' and
                market_cap_category in ['MEGA_CAP', 'LARGE_CAP'] and
                volatility == 'LOW'):
                logger.info("Classified as B8: NYSE Mega Utilities Income")
                return MARKET_STATES['B8']

        # ===== PRIORITY 5: C GROUP (Healthcare) =====

        if sector == 'HEALTHCARE':
            # C1: NYSE Mega Healthcare Quality
            if (exchange == 'NYSE' and
                market_cap_category == 'MEGA_CAP' and
                profitability == 'HIGHLY_PROFITABLE'):
                logger.info("Classified as C1: NYSE Mega Healthcare Quality")
                return MARKET_STATES['C1']

            # C2: NASDAQ Large Biotech Growth
            if (exchange == 'NASDAQ' and
                market_cap_category == 'LARGE_CAP' and
                growth_profile == 'HIGH_GROWTH'):
                logger.info("Classified as C2: NASDAQ Large Biotech Growth")
                return MARKET_STATES['C2']

            # C3: NASDAQ Mid Biotech Speculative
            if (exchange == 'NASDAQ' and
                market_cap_category == 'MID_CAP' and
                profitability == 'UNPROFITABLE'):
                logger.info("Classified as C3: NASDAQ Mid Biotech Speculative")
                return MARKET_STATES['C3']

            # C4: NYSE Large Healthcare Services Stable
            if (exchange == 'NYSE' and
                market_cap_category == 'LARGE_CAP' and
                profitability in ['PROFITABLE', 'MODERATELY_PROFITABLE']):
                logger.info("Classified as C4: NYSE Large Healthcare Services Stable")
                return MARKET_STATES['C4']

        # ===== PRIORITY 6: D GROUP (Mid/Small Growth) =====

        # D1: NASDAQ Mid Growth Profitable
        if (exchange == 'NASDAQ' and
            market_cap_category == 'MID_CAP' and
            growth_profile == 'HIGH_GROWTH' and
            profitability == 'PROFITABLE'):
            logger.info("Classified as D1: NASDAQ Mid Growth Profitable")
            return MARKET_STATES['D1']

        # D2: NASDAQ Small Growth Unprofitable
        if (exchange == 'NASDAQ' and
            market_cap_category == 'SMALL_CAP' and
            growth_profile in ['HYPER_GROWTH', 'HIGH_GROWTH'] and
            profitability == 'UNPROFITABLE'):
            logger.info("Classified as D2: NASDAQ Small Growth Unprofitable")
            return MARKET_STATES['D2']

        # D3: NYSE Mid Industrial Recovery
        if (exchange == 'NYSE' and
            market_cap_category == 'MID_CAP' and
            sector in ['INDUSTRIALS', 'BASIC MATERIALS'] and
            economic_cycle in ['EXPANSION', 'EASING']):
            logger.info("Classified as D3: NYSE Mid Industrial Recovery")
            return MARKET_STATES['D3']

        # D4: NASDAQ Mid Consumer Cyclical Disruptor
        if (exchange == 'NASDAQ' and
            market_cap_category in ['MID_CAP', 'LARGE_CAP'] and
            sector in ['CONSUMER CYCLICAL', 'COMMUNICATION SERVICES'] and
            growth_profile == 'HIGH_GROWTH'):
            logger.info("Classified as D4: NASDAQ Mid Consumer Cyclical Disruptor")
            return MARKET_STATES['D4']

        # D5: Small Real Estate Value
        if (market_cap_category in ['SMALL_CAP', 'MID_CAP'] and
            sector == 'REAL ESTATE' and
            growth_profile == 'SLOW_GROWTH'):
            logger.info("Classified as D5: Small Real Estate Value")
            return MARKET_STATES['D5']

        # ===== DEFAULT: OTHER =====
        logger.info("Classified as OTHER: No specific state matched")
        return MARKET_STATES['OTHER']

    def load_cache(self) -> Optional[Dict]:
        """
        Load classification cache if valid (within 24 hours)

        Returns:
            Cached classifications or None
        """
        if not os.path.exists(self.cache_file):
            logger.info("No cache file found")
            return None

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Check expiry
            expiry_time = datetime.fromisoformat(cache_data['expiry'])
            if datetime.now() > expiry_time:
                logger.info("Cache expired")
                return None

            logger.info(f"Cache loaded: {len(cache_data['classifications'])} classifications")
            return cache_data['classifications']

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None

    def save_cache(self, classifications: Dict):
        """
        Save classifications to cache file

        Args:
            classifications: Dict of {symbol: classification_name}
        """
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'expiry': (datetime.now() + timedelta(seconds=self.cache_duration)).isoformat(),
                'classifications': classifications
            }

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Cache saved: {len(classifications)} classifications")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def classify_batch(self, conditions_dict: Dict[str, Dict]) -> Dict[str, str]:
        """
        Classify multiple stocks at once

        Args:
            conditions_dict: Dict of {symbol: conditions}

        Returns:
            Dict of {symbol: classification_name}
        """
        # Try to load cache
        cached = self.load_cache()
        if cached:
            logger.info("Using cached classifications")
            return cached

        # Classify all stocks
        results = {}
        for symbol, conditions in conditions_dict.items():
            results[symbol] = self.classify(conditions)

        # Save to cache
        self.save_cache(results)

        return results

    def get_classification_stats(self, classifications: Dict[str, str]) -> Dict:
        """
        Get statistics about classifications

        Args:
            classifications: Dict of {symbol: classification_name}

        Returns:
            Statistics dict with counts and coverage information
        """
        counts = Counter(classifications.values())
        total = len(classifications)

        stats = {
            'total_stocks': total,
            'classification_counts': dict(counts),
            'coverage': {
                'classified': total - counts.get('OTHER', 0),
                'others': counts.get('OTHER', 0),
                'coverage_rate': (total - counts.get('OTHER', 0)) / total * 100 if total > 0 else 0
            },
            'group_distribution': self._get_group_distribution(classifications)
        }

        return stats

    def _get_group_distribution(self, classifications: Dict[str, str]) -> Dict:
        """
        Get distribution by group (A, B, C, D, E, F, OTHER)

        Args:
            classifications: Dict of {symbol: classification_name}

        Returns:
            Dict of {group: count}
        """
        group_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'OTHER': 0}

        for state in classifications.values():
            if state == 'OTHER':
                group_counts['OTHER'] += 1
            elif state.startswith('A'):
                group_counts['A'] += 1
            elif state.startswith('B'):
                group_counts['B'] += 1
            elif state.startswith('C'):
                group_counts['C'] += 1
            elif state.startswith('D'):
                group_counts['D'] += 1
            elif state.startswith('E'):
                group_counts['E'] += 1
            elif state.startswith('F'):
                group_counts['F'] += 1

        return group_counts


# ===========================
# Helper Functions
# ===========================

def get_state_factor_weights(market_state: str) -> Dict:
    """
    Get factor weights for a given market state

    Parameters:
        market_state: Market state ID

    Returns:
        dict: Factor weights {'value': float, 'quality': float, 'momentum': float, 'growth': float}
    """
    return MARKET_STATE_FACTOR_WEIGHTS.get(market_state, {
        'value': 0.25,
        'quality': 0.25,
        'momentum': 0.25,
        'growth': 0.25
    })


def get_state_info(market_state: str) -> Dict:
    """
    Get detailed information about a market state

    Parameters:
        market_state: Market state ID

    Returns:
        dict: State information including group, description, representative stocks, characteristics
    """
    return MARKET_STATE_INFO.get(market_state, {
        'group': 'UNKNOWN',
        'description': 'Unknown state',
        'representative_stocks': [],
        'characteristics': 'N/A'
    })


def get_all_states_by_group(group: str) -> List[str]:
    """
    Get all market states belonging to a specific group

    Parameters:
        group: Group ID ('A', 'B', 'C', 'D', 'E', 'F', 'OTHER')

    Returns:
        list: List of market state IDs in the group
    """
    return [state_id for state_id, info in MARKET_STATE_INFO.items()
            if info['group'] == group]


def format_state_summary(market_state: str, conditions: Dict) -> str:
    """
    Format a human-readable summary of the market state classification

    Parameters:
        market_state: Market state ID
        conditions: Condition values used for classification

    Returns:
        str: Formatted summary string
    """
    info = get_state_info(market_state)
    weights = get_state_factor_weights(market_state)

    summary = f"\n{'='*70}\n"
    summary += f"Market State Classification\n"
    summary += f"{'='*70}\n\n"
    summary += f"State: {market_state}\n"
    summary += f"Group: {info['group']}\n"
    summary += f"Description: {info['description']}\n"
    summary += f"Representative Stocks: {', '.join(info['representative_stocks'][:5])}\n"
    summary += f"Characteristics: {info['characteristics']}\n\n"

    summary += f"[Key Conditions]\n"
    summary += f"Exchange: {conditions.get('exchange', 'N/A')}\n"
    summary += f"Market Cap: {conditions.get('market_cap_category', 'N/A')}\n"
    summary += f"Sector: {conditions.get('sector', 'N/A')}\n"
    summary += f"Growth Profile: {conditions.get('growth_profile', 'N/A')}\n"
    summary += f"Profitability: {conditions.get('profitability', 'N/A')}\n"
    summary += f"Economic Cycle: {conditions.get('economic_cycle', 'N/A')}\n\n"

    summary += f"[State Factor Weights]\n"
    summary += f"Value:    {weights['value']*100:6.2f}%\n"
    summary += f"Quality:  {weights['quality']*100:6.2f}%\n"
    summary += f"Momentum: {weights['momentum']*100:6.2f}%\n"
    summary += f"Growth:   {weights['growth']*100:6.2f}%\n"
    summary += f"{'='*70}\n"

    return summary


# ===========================
# Main execution for testing
# ===========================

if __name__ == "__main__":
    # Test classification with sample conditions
    print("\nUS Market State Classifier - Test Mode\n")
    print("="*70)

    classifier = USMarketClassifier()

    # Test case 1: AAPL-like stock (NASDAQ Mega Tech)
    test_conditions_1 = {
        'exchange': 'NASDAQ',
        'market_cap_category': 'MEGA_CAP',
        'sector': 'TECHNOLOGY',
        'growth_profile': 'HIGH_GROWTH',
        'profitability': 'HIGHLY_PROFITABLE',
        'volatility': 'MEDIUM',
        'economic_cycle': 'NEUTRAL',
        'liquidity_level': 'VERY_HIGH',
        'options_positioning': 'BULLISH',
        'analyst_momentum': 'POSITIVE',
        'market_sentiment': 'BULLISH'
    }

    state_1 = classifier.classify(test_conditions_1)
    print(format_state_summary(state_1, test_conditions_1))

    # Test case 2: JPM-like stock (NYSE Mega Financials)
    test_conditions_2 = {
        'exchange': 'NYSE',
        'market_cap_category': 'MEGA_CAP',
        'sector': 'FINANCIALS',
        'growth_profile': 'MODERATE_GROWTH',
        'profitability': 'PROFITABLE',
        'volatility': 'MEDIUM',
        'economic_cycle': 'NEUTRAL',
        'liquidity_level': 'VERY_HIGH',
        'options_positioning': 'NEUTRAL',
        'analyst_momentum': 'NEUTRAL',
        'market_sentiment': 'NEUTRAL'
    }

    state_2 = classifier.classify(test_conditions_2)
    print(format_state_summary(state_2, test_conditions_2))

    # Test batch classification
    print("\n" + "="*70)
    print("Batch Classification Test")
    print("="*70)

    batch_conditions = {
        'AAPL': test_conditions_1,
        'JPM': test_conditions_2
    }

    batch_results = classifier.classify_batch(batch_conditions)
    print(f"\nBatch results: {batch_results}")

    # Test statistics
    stats = classifier.get_classification_stats(batch_results)
    print(f"\nClassification Statistics:")
    print(f"Total stocks: {stats['total_stocks']}")
    print(f"Coverage rate: {stats['coverage']['coverage_rate']:.2f}%")
    print(f"Group distribution: {stats['group_distribution']}")

    # Display group summary
    print("\n" + "="*70)
    print("Market State Groups Summary")
    print("="*70)
    for group in ['A', 'B', 'C', 'D', 'E', 'F']:
        states = get_all_states_by_group(group)
        print(f"\nGroup {group}: {len(states)} states")
        for state in states:
            print(f"  - {state}")
