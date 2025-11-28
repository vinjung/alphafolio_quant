"""
US Market Environment Classifier

Classifies overall market conditions into 5 environments:
- BULL: Strong uptrend with low volatility
- BEAR: Downtrend with high volatility
- SIDEWAYS: Range-bound with moderate volatility
- HIGH_VOLATILITY: VIX spike (overrides other conditions)
- NORMAL: Default condition

This classification is used in combination with 31 market states to create
the integrated weighting system (32 states × 5 environments).

Architecture:
- Class-based structure (matches US factor calculator pattern)
- db_manager pattern for database access
- Optional caching for performance
- Mock data support for testing

Usage:
    # Real database
    classifier = USMarketEnvironmentClassifier(db_manager, analysis_date)
    environment = await classifier.classify()

    # Get history
    history = await classifier.get_history(days=30)

    # Mock data (testing)
    classifier = USMarketEnvironmentClassifier(None, use_mock=True)
    environment = await classifier.classify()

File: us/us_market_environment_classifier.py
"""

import os
import json
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# Market Environment Definitions
# ========================================================================

MARKET_ENVIRONMENTS = {
    'BULL': {
        'description': 'Bull Market - Strong uptrend with low volatility',
        'characteristics': 'MA200 rising >5%, VIX <20, Price >5% above MA200',
        'typical_vix': '< 20',
        'trend': 'Upward'
    },
    'BEAR': {
        'description': 'Bear Market - Downtrend with high volatility',
        'characteristics': 'MA200 declining >5%, VIX >30, Price <5% below MA200',
        'typical_vix': '> 30',
        'trend': 'Downward'
    },
    'SIDEWAYS': {
        'description': 'Sideways Market - Range-bound with moderate volatility',
        'characteristics': 'MA200 flat ±2%, Price within ±5% of MA200, VIX 15-25',
        'typical_vix': '15-25',
        'trend': 'Flat'
    },
    'HIGH_VOLATILITY': {
        'description': 'High Volatility - Elevated fear/uncertainty',
        'characteristics': 'VIX >25 (overrides other conditions)',
        'typical_vix': '> 25',
        'trend': 'Unstable'
    },
    'NORMAL': {
        'description': 'Normal Market - No strong directional signals',
        'characteristics': 'Mixed signals, moderate conditions',
        'typical_vix': '15-25',
        'trend': 'Neutral'
    }
}


# ========================================================================
# USMarketEnvironmentClassifier Class
# ========================================================================

class USMarketEnvironmentClassifier:
    """
    US Market Environment Classifier

    Classifies overall US market environment based on SPY/VIX data into 5 categories:
    1. BEAR (Downtrend + high VIX + weak breadth)
    2. BULL (Uptrend + low VIX + strong breadth)
    3. HIGH_VOLATILITY (VIX >25 without clear trend)
    4. SIDEWAYS (Range-bound + moderate VIX)
    5. NORMAL (Default fallback)

    Classification Priority: BEAR → BULL → HIGH_VOLATILITY → SIDEWAYS → NORMAL

    Note: Specific market conditions (BEAR/BULL) are checked before general
    volatility to properly identify trending markets with high volatility.

    Args:
        db_manager: AsyncDatabaseManager instance (None if use_mock=True)
        analysis_date: Date to classify (default: latest available)
        use_mock: Use mock data for testing (default: False)
        cache_dir: Directory for cache files (default: ./cache)
    """

    def __init__(
        self,
        db_manager=None,
        analysis_date: Optional[datetime] = None,
        use_mock: bool = False,
        cache_dir: str = None
    ):
        """Initialize market environment classifier"""
        self.db_manager = db_manager
        self.analysis_date = analysis_date
        self.use_mock = use_mock
        self.logger = logging.getLogger(__name__)

        # Cache configuration
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), 'cache')
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, 'market_environment_cache.json')
        self.cache_duration = 3600  # 1 hour

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Validate configuration
        if not use_mock and db_manager is None:
            raise ValueError("db_manager is required when use_mock=False")

    async def execute_query(self, query: str, *params):
        """
        Execute query using db_manager

        Args:
            query: SQL query string
            *params: Query parameters

        Returns:
            Query results
        """
        if self.use_mock:
            raise RuntimeError("Cannot execute real queries in mock mode")

        try:
            result = await self.db_manager.execute_query(query, *params)
            return result
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise

    async def classify(self) -> str:
        """
        Classify current US market environment based on SPY/VIX data

        Classification logic (priority order):
        1. BEAR: Downtrend + high VIX + price below MA200
        2. BULL: Uptrend + low VIX + price above MA200
        3. HIGH_VOLATILITY: VIX > 25 (without clear trend)
        4. SIDEWAYS: Range-bound + moderate VIX
        5. NORMAL: Default fallback

        Returns:
            Market environment: 'BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOLATILITY', or 'NORMAL'
        """
        try:
            # Check cache first
            cached_env = self._load_from_cache()
            if cached_env:
                self.logger.info(f"Using cached environment: {cached_env}")
                return cached_env

            # Get market data (real or mock)
            if self.use_mock:
                market_data = self._get_mock_market_data()
            else:
                market_data = await self._get_spy_vix_data()

            if not market_data:
                self.logger.warning("Market data not available, defaulting to NORMAL")
                return 'NORMAL'

            # Extract metrics
            close = market_data['close']
            ma_200 = market_data['ma_200']
            ma_50 = market_data['ma_50']
            volatility_30d = market_data['volatility_30d']
            distance_from_ma200 = market_data['distance_from_ma200']
            ma_200_trend_pct = market_data['ma_200_trend_pct']
            vix = market_data['vix']

            # Classification logic (priority order)
            # Reordered: Specific conditions (BEAR/BULL) before general HIGH_VOLATILITY

            # 1. BEAR Market (check before HIGH_VOLATILITY)
            # Strong bear criteria: Downtrend + high VIX + price below MA200
            if (ma_200_trend_pct < -0.05 and  # MA200 declining > 5% over 3 months
                vix > 30 and  # High fear
                distance_from_ma200 < -0.05):  # Price > 5% below MA200
                self.logger.info(f"Market Environment: BEAR (strong) (MA200 trend={ma_200_trend_pct:.2%}, VIX={vix:.2f})")
                environment = 'BEAR'
                self._save_to_cache(environment)
                return environment

            # Weaker bear criteria
            if (ma_200_trend_pct < -0.02 and
                distance_from_ma200 < -0.08 and
                vix > 25):
                self.logger.info(f"Market Environment: BEAR (weak) (dist from MA200={distance_from_ma200:.2%}, VIX={vix:.2f})")
                environment = 'BEAR'
                self._save_to_cache(environment)
                return environment

            # 2. BULL Market
            # Strong bull criteria: Uptrend + low VIX + price above MA200
            if (ma_200_trend_pct > 0.05 and  # MA200 rising > 5% over 3 months
                vix < 20 and  # Low fear
                distance_from_ma200 > 0.05):  # Price > 5% above MA200
                self.logger.info(f"Market Environment: BULL (MA200 trend={ma_200_trend_pct:.2%}, VIX={vix:.2f})")
                environment = 'BULL'
                self._save_to_cache(environment)
                return environment

            # Weaker bull criteria
            if (ma_200_trend_pct > 0.02 and
                distance_from_ma200 > 0.03 and
                vix < 18 and
                ma_50 > ma_200):  # Golden cross
                self.logger.info(f"Market Environment: BULL (weak) (dist from MA200={distance_from_ma200:.2%}, VIX={vix:.2f})")
                environment = 'BULL'
                self._save_to_cache(environment)
                return environment

            # 3. HIGH_VOLATILITY (general volatility spike without clear trend)
            if vix > 25:
                self.logger.info(f"Market Environment: HIGH_VOLATILITY (VIX={vix:.2f})")
                environment = 'HIGH_VOLATILITY'
                self._save_to_cache(environment)
                return environment

            # 4. SIDEWAYS Market
            # Criteria: Flat trend + price oscillating around MA200
            if (abs(ma_200_trend_pct) < 0.02 and  # MA200 flat (±2% over 3 months)
                abs(distance_from_ma200) < 0.05 and  # Price within ±5% of MA200
                15 < vix < 25):  # Moderate volatility
                self.logger.info(f"Market Environment: SIDEWAYS (MA200 trend={ma_200_trend_pct:.2%}, dist={distance_from_ma200:.2%})")
                environment = 'SIDEWAYS'
                self._save_to_cache(environment)
                return environment

            # 5. NORMAL (default)
            self.logger.info(f"Market Environment: NORMAL (no strong signals) (VIX={vix:.2f}, MA200 trend={ma_200_trend_pct:.2%})")
            environment = 'NORMAL'
            self._save_to_cache(environment)
            return environment

        except Exception as e:
            self.logger.error(f"Error classifying market environment: {e}")
            self.logger.warning("Defaulting to NORMAL due to error")
            return 'NORMAL'

    async def _get_spy_vix_data(self) -> Optional[Dict]:
        """
        Get SPY and VIX data from database

        Returns:
            Dict with market metrics or None
        """
        try:
            # Get SPY price data with moving averages
            spy_data = await self.execute_query("""
                WITH recent_prices AS (
                    SELECT
                        date,
                        close,
                        high,
                        low,
                        volume,
                        AVG(close) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) as ma_200,
                        AVG(close) OVER (ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as ma_50,
                        STDDEV(close) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) * SQRT(252) as volatility_30d
                    FROM us_daily
                    WHERE symbol = 'SPY'
                        AND ($1::DATE IS NULL OR date <= $1::DATE)
                    ORDER BY date DESC
                    LIMIT 1
                ),
                ma_200_trend AS (
                    SELECT
                        AVG(close) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND 140 PRECEDING) as ma_200_3m_ago
                    FROM us_daily
                    WHERE symbol = 'SPY'
                        AND ($1::DATE IS NULL OR date <= $1::DATE)
                    ORDER BY date DESC
                    LIMIT 1
                )
                SELECT
                    p.close,
                    p.ma_200,
                    p.ma_50,
                    p.volatility_30d,
                    t.ma_200_3m_ago,
                    (p.close - p.ma_200)::FLOAT / p.ma_200 as distance_from_ma200,
                    (p.ma_200 - t.ma_200_3m_ago)::FLOAT / t.ma_200_3m_ago as ma_200_trend_pct
                FROM recent_prices p
                CROSS JOIN ma_200_trend t
            """, self.analysis_date)

            if not spy_data or spy_data[0]['close'] is None:
                self.logger.warning("SPY data not available")
                return None

            spy_row = spy_data[0]

            # Extract SPY metrics
            close = float(spy_row['close'])
            ma_200 = float(spy_row['ma_200']) if spy_row['ma_200'] else close
            ma_50 = float(spy_row['ma_50']) if spy_row['ma_50'] else close
            volatility_30d = float(spy_row['volatility_30d']) if spy_row['volatility_30d'] else 0.15
            distance_from_ma200 = float(spy_row['distance_from_ma200']) if spy_row['distance_from_ma200'] else 0.0
            ma_200_trend_pct = float(spy_row['ma_200_trend_pct']) if spy_row['ma_200_trend_pct'] else 0.0

            # Get VIX if available
            try:
                vix_data = await self.execute_query("""
                    SELECT close::FLOAT as vix
                    FROM us_daily
                    WHERE symbol = 'VIX'
                        AND ($1::DATE IS NULL OR date <= $1::DATE)
                    ORDER BY date DESC
                    LIMIT 1
                """, self.analysis_date)
                vix = float(vix_data[0]['vix']) if vix_data and vix_data[0]['vix'] else None
            except:
                vix = None

            # If VIX not available, estimate from SPY volatility
            if vix is None:
                vix = volatility_30d * 100  # Rough approximation
                self.logger.info(f"VIX not available, estimated from SPY volatility: {vix:.2f}")

            return {
                'close': close,
                'ma_200': ma_200,
                'ma_50': ma_50,
                'volatility_30d': volatility_30d,
                'distance_from_ma200': distance_from_ma200,
                'ma_200_trend_pct': ma_200_trend_pct,
                'vix': vix
            }

        except Exception as e:
            self.logger.error(f"Error getting SPY/VIX data: {e}")
            return None

    def _get_mock_market_data(self) -> Dict:
        """
        Get mock market data for testing

        Returns:
            Dict with mock market metrics
        """
        # Simulate current BULL market conditions (as of late 2024)
        return {
            'close': 580.0,
            'ma_200': 520.0,
            'ma_50': 565.0,
            'volatility_30d': 0.12,
            'distance_from_ma200': 0.115,  # 11.5% above MA200
            'ma_200_trend_pct': 0.08,  # 8% growth over 3 months
            'vix': 14.5
        }

    async def get_history(self, days: int = 30) -> Dict[str, str]:
        """
        Get historical market environment classifications

        Args:
            days: Number of trading days to look back

        Returns:
            Dict with {date: environment}
        """
        if self.use_mock:
            return self._get_mock_history(days)

        try:
            # Get list of trading days
            dates_data = await self.execute_query("""
                SELECT DISTINCT date
                FROM us_daily
                WHERE symbol = 'SPY'
                    AND date >= NOW() - INTERVAL '1 year'
                ORDER BY date DESC
                LIMIT $1
            """, days)

            if not dates_data:
                return {}

            history = {}
            for row in dates_data:
                date = row['date']
                # Create temporary classifier for each date
                temp_classifier = USMarketEnvironmentClassifier(
                    db_manager=self.db_manager,
                    analysis_date=date,
                    use_mock=False
                )
                env = await temp_classifier.classify()
                history[date.strftime('%Y-%m-%d')] = env

            return history

        except Exception as e:
            self.logger.error(f"Error getting market environment history: {e}")
            return {}

    def _get_mock_history(self, days: int = 30) -> Dict[str, str]:
        """Get mock historical data for testing"""
        from datetime import date, timedelta

        history = {}
        base_date = date.today()

        for i in range(days):
            current_date = base_date - timedelta(days=i)
            # Simulate varying market conditions
            if i < 10:
                env = 'BULL'
            elif i < 15:
                env = 'NORMAL'
            elif i < 20:
                env = 'SIDEWAYS'
            elif i < 25:
                env = 'HIGH_VOLATILITY'
            else:
                env = 'BEAR'

            history[current_date.strftime('%Y-%m-%d')] = env

        return history

    def _load_from_cache(self) -> Optional[str]:
        """
        Load classification from cache if valid

        Returns:
            Cached environment or None
        """
        if not os.path.exists(self.cache_file):
            return None

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if (datetime.now() - cache_time).total_seconds() > self.cache_duration:
                self.logger.info("Cache expired")
                return None

            # Check if analysis_date matches (or both are None)
            cached_date = cache_data.get('analysis_date')
            current_date = self.analysis_date.strftime('%Y-%m-%d') if self.analysis_date else None

            if cached_date != current_date:
                return None

            return cache_data['environment']

        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, environment: str):
        """
        Save classification to cache

        Args:
            environment: Market environment to cache
        """
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'analysis_date': self.analysis_date.strftime('%Y-%m-%d') if self.analysis_date else None,
                'environment': environment
            }

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            self.logger.debug(f"Cached environment: {environment}")

        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")

    def get_environment_info(self, environment: str) -> Dict:
        """
        Get detailed information about a market environment

        Args:
            environment: Market environment name

        Returns:
            Dict with environment details
        """
        return MARKET_ENVIRONMENTS.get(environment, {})


# ========================================================================
# Test Functions
# ========================================================================

async def test_with_real_data():
    """Test with real database connection"""
    import os
    from dotenv import load_dotenv
    from us_db_async import AsyncDatabaseManager

    load_dotenv()

    # Initialize database
    db_manager = AsyncDatabaseManager()
    await db_manager.initialize()

    print("\n" + "="*80)
    print("US Market Environment Classification - REAL DATA TEST")
    print("="*80)

    try:
        # Test current environment
        classifier = USMarketEnvironmentClassifier(db_manager)
        current_env = await classifier.classify()

        print(f"\n>>> Current Market Environment: {current_env}")

        # Get environment details
        env_info = classifier.get_environment_info(current_env)
        print(f"\nDescription: {env_info.get('description', 'N/A')}")
        print(f"Characteristics: {env_info.get('characteristics', 'N/A')}")
        print(f"Typical VIX: {env_info.get('typical_vix', 'N/A')}")
        print(f"Trend: {env_info.get('trend', 'N/A')}")

        # Test historical classifications
        print("\n" + "-"*80)
        print("Last 10 Trading Days:")
        print("-"*80)

        history = await classifier.get_history(days=10)
        for date, env in sorted(history.items(), reverse=True):
            print(f"{date}: {env}")

    finally:
        await db_manager.close()


def test_with_mock_data():
    """Test with mock data (no database required)"""
    import asyncio

    print("\n" + "="*80)
    print("US Market Environment Classification - MOCK DATA TEST")
    print("="*80)

    # Test current environment with mock data
    classifier = USMarketEnvironmentClassifier(db_manager=None, use_mock=True)

    async def run_test():
        current_env = await classifier.classify()

        print(f"\n>>> Current Market Environment (Mock): {current_env}")

        # Get environment details
        env_info = classifier.get_environment_info(current_env)
        print(f"\nDescription: {env_info.get('description', 'N/A')}")
        print(f"Characteristics: {env_info.get('characteristics', 'N/A')}")

        # Test historical classifications with mock data
        print("\n" + "-"*80)
        print("Last 10 Days (Mock History):")
        print("-"*80)

        history = await classifier.get_history(days=10)
        for date, env in sorted(history.items(), reverse=True):
            print(f"{date}: {env}")

    asyncio.run(run_test())


async def test_all_environments():
    """Test all 5 environment classifications with different mock data"""
    print("\n" + "="*80)
    print("Testing All 5 Market Environments")
    print("="*80)

    # Clear cache before testing
    cache_file = os.path.join(os.getcwd(), 'cache', 'market_environment_cache.json')
    if os.path.exists(cache_file):
        os.remove(cache_file)

    test_scenarios = {
        'BULL': {
            'close': 580.0, 'ma_200': 520.0, 'ma_50': 565.0,
            'volatility_30d': 0.12, 'distance_from_ma200': 0.115,
            'ma_200_trend_pct': 0.08, 'vix': 14.5
        },
        'BEAR': {
            'close': 450.0, 'ma_200': 520.0, 'ma_50': 465.0,
            'volatility_30d': 0.35, 'distance_from_ma200': -0.135,
            'ma_200_trend_pct': -0.07, 'vix': 35.0
        },
        'SIDEWAYS': {
            'close': 520.0, 'ma_200': 518.0, 'ma_50': 522.0,
            'volatility_30d': 0.18, 'distance_from_ma200': 0.004,
            'ma_200_trend_pct': 0.01, 'vix': 18.0
        },
        'HIGH_VOLATILITY': {
            'close': 500.0, 'ma_200': 520.0, 'ma_50': 510.0,
            'volatility_30d': 0.40, 'distance_from_ma200': -0.038,
            'ma_200_trend_pct': -0.02, 'vix': 32.0
        },
        'NORMAL': {
            'close': 530.0, 'ma_200': 520.0, 'ma_50': 525.0,
            'volatility_30d': 0.20, 'distance_from_ma200': 0.019,
            'ma_200_trend_pct': 0.025, 'vix': 22.0
        }
    }

    for expected_env, mock_data in test_scenarios.items():
        classifier = USMarketEnvironmentClassifier(db_manager=None, use_mock=True)
        # Override mock data
        classifier._get_mock_market_data = lambda data=mock_data: data
        # Disable cache for testing
        classifier.cache_duration = 0

        result_env = await classifier.classify()
        status = "[OK]" if result_env == expected_env else "[FAIL]"

        print(f"\n{status} Expected: {expected_env}, Got: {result_env}")
        print(f"   VIX: {mock_data['vix']:.1f}, MA200 Trend: {mock_data['ma_200_trend_pct']:.1%}, "
              f"Distance from MA200: {mock_data['distance_from_ma200']:.1%}")


# ========================================================================
# Main
# ========================================================================

if __name__ == "__main__":
    import asyncio
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--real':
        # Test with real database
        asyncio.run(test_with_real_data())
    elif len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Test all environments
        asyncio.run(test_all_environments())
    else:
        # Test with mock data (default)
        test_with_mock_data()
