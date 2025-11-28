"""
US Agent Metrics - Multi-Agent AI Support Module

This module provides additional metrics for multi-agent AI systems:
1. ATR-based stop loss / take profit
2. Entry timing score
3. Position size calculation
4. IV percentile (US-specific)
5. Insider signal (US-specific)
6. Action triggers generation
7. Scenario probability analysis

Usage:
    from us_agent_metrics import USAgentMetrics

    agent_metrics = USAgentMetrics(db_manager)
    result = await agent_metrics.calculate_all(symbol, analysis_date)

File: us/us_agent_metrics.py
"""

import asyncio
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

from us_db_async import AsyncDatabaseManager

logger = logging.getLogger(__name__)


class USAgentMetrics:
    """
    US Agent Metrics Calculator

    Provides additional metrics for multi-agent AI investment decisions:
    - Risk management (ATR-based stop/take profit, position sizing)
    - Entry timing analysis
    - US-specific indicators (IV percentile, insider signals)
    - Action triggers and scenario analysis
    """

    def __init__(self, db: AsyncDatabaseManager):
        """
        Initialize USAgentMetrics

        Args:
            db: AsyncDatabaseManager instance
        """
        self.db = db

    # ========================================================================
    # 1. ATR-based Stop Loss / Take Profit
    # ========================================================================

    async def calculate_atr_stop_take_profit(
        self, symbol: str, analysis_date: date
    ) -> Dict[str, Optional[float]]:
        """
        Calculate ATR-based stop loss and take profit levels

        Uses us_atr table (14-day ATR) and us_daily table for current price.

        Logic:
            - atr_pct = (ATR / close) * 100
            - stop_loss_pct = -2 * atr_pct (2 ATR below)
            - take_profit_pct = 3 * atr_pct (3 ATR above)
            - risk_reward_ratio = 1.5 (3/2)

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date

        Returns:
            {
                'atr_pct': float or None,
                'stop_loss_pct': float or None,
                'take_profit_pct': float or None,
                'risk_reward_ratio': float or None
            }
        """
        query = """
        SELECT a.atr, d.close
        FROM us_atr a
        JOIN us_daily d ON a.symbol = d.symbol AND a.date = d.date
        WHERE a.symbol = $1 AND a.date <= $2
        ORDER BY a.date DESC
        LIMIT 1
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)

            if not result or not result[0].get('atr') or not result[0].get('close'):
                return {
                    'atr_pct': None,
                    'stop_loss_pct': None,
                    'take_profit_pct': None,
                    'risk_reward_ratio': None
                }

            atr = float(result[0]['atr'])
            close = float(result[0]['close'])

            if close <= 0:
                return {
                    'atr_pct': None,
                    'stop_loss_pct': None,
                    'take_profit_pct': None,
                    'risk_reward_ratio': None
                }

            atr_pct = (atr / close) * 100
            stop_loss_pct = -2 * atr_pct
            take_profit_pct = 3 * atr_pct
            risk_reward_ratio = 1.5  # 3 / 2

            return {
                'atr_pct': round(atr_pct, 2),
                'stop_loss_pct': round(stop_loss_pct, 2),
                'take_profit_pct': round(take_profit_pct, 2),
                'risk_reward_ratio': round(risk_reward_ratio, 2)
            }

        except Exception as e:
            logger.error(f"{symbol}: ATR calculation failed - {e}")
            return {
                'atr_pct': None,
                'stop_loss_pct': None,
                'take_profit_pct': None,
                'risk_reward_ratio': None
            }

    # ========================================================================
    # 2. Entry Timing Score
    # ========================================================================

    async def calculate_entry_timing_score(
        self, symbol: str, analysis_date: date, current_score: float
    ) -> Dict[str, Optional[float]]:
        """
        Calculate entry timing score

        Components:
            1. score_trend_2w: Score change over 2 weeks (from us_stock_grade)
            2. price_position_52w: Current price position in 52-week range (0-100)
            3. entry_timing_score: Composite score (0-100)

        Logic:
            - Score trend positive + price not at 52w high = Good entry
            - Score trend negative + price at 52w high = Bad entry

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            current_score: Current final_score from us_stock_grade

        Returns:
            {
                'entry_timing_score': float or None,
                'score_trend_2w': float or None,
                'price_position_52w': float or None
            }
        """
        # Get score from 2 weeks ago
        score_query = """
        SELECT final_score
        FROM us_stock_grade
        WHERE symbol = $1 AND date <= $2 - INTERVAL '14 days'
        ORDER BY date DESC
        LIMIT 1
        """

        # Get 52-week high/low and current price
        price_query = """
        WITH price_range AS (
            SELECT
                MAX(high) as high_52w,
                MIN(low) as low_52w
            FROM us_daily
            WHERE symbol = $1
              AND date > $2 - INTERVAL '252 days'
              AND date <= $2
        ),
        current_price AS (
            SELECT close
            FROM us_daily
            WHERE symbol = $1 AND date <= $2
            ORDER BY date DESC
            LIMIT 1
        )
        SELECT
            pr.high_52w,
            pr.low_52w,
            cp.close as current_price
        FROM price_range pr, current_price cp
        """

        try:
            score_result, price_result = await asyncio.gather(
                self.db.execute_query(score_query, symbol, analysis_date),
                self.db.execute_query(price_query, symbol, analysis_date)
            )

            # Score trend calculation
            score_trend_2w = None
            if score_result and score_result[0].get('final_score') is not None:
                old_score = float(score_result[0]['final_score'])
                score_trend_2w = current_score - old_score

            # Price position calculation (0 = 52w low, 100 = 52w high)
            price_position_52w = None
            if price_result and price_result[0].get('high_52w'):
                r = price_result[0]
                high_52w = float(r['high_52w'])
                low_52w = float(r['low_52w'])
                current_price = float(r['current_price'])

                if high_52w > low_52w:
                    price_position_52w = ((current_price - low_52w) / (high_52w - low_52w)) * 100

            # Entry timing score calculation
            entry_timing_score = None
            if score_trend_2w is not None and price_position_52w is not None:
                # Positive score trend is good (0-50 points)
                trend_component = min(50, max(0, (score_trend_2w + 10) * 2.5))

                # Price not at extreme high is good (0-50 points)
                # Best entry: price at 30-70% of 52w range
                if price_position_52w <= 30:
                    position_component = 50  # Near 52w low - good entry
                elif price_position_52w <= 70:
                    position_component = 40  # Middle range - decent entry
                elif price_position_52w <= 90:
                    position_component = 25  # Near high - cautious
                else:
                    position_component = 10  # At 52w high - risky entry

                entry_timing_score = trend_component + position_component

            return {
                'entry_timing_score': round(entry_timing_score, 1) if entry_timing_score else None,
                'score_trend_2w': round(score_trend_2w, 2) if score_trend_2w else None,
                'price_position_52w': round(price_position_52w, 2) if price_position_52w else None
            }

        except Exception as e:
            logger.error(f"{symbol}: Entry timing calculation failed - {e}")
            return {
                'entry_timing_score': None,
                'score_trend_2w': None,
                'price_position_52w': None
            }

    # ========================================================================
    # 3. Position Size Calculation
    # ========================================================================

    async def calculate_position_size(
        self,
        symbol: str,
        analysis_date: date,
        stop_loss_pct: Optional[float],
        var_95: Optional[float],
        max_risk_per_trade: float = 2.0
    ) -> Optional[float]:
        """
        Calculate recommended position size as percentage of portfolio

        Logic:
            - Based on risk per trade (default 2% of portfolio)
            - Position size = max_risk_per_trade / abs(stop_loss_pct)
            - Adjusted by VaR if available
            - Capped at 10% maximum

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            stop_loss_pct: Stop loss percentage (negative)
            var_95: Value at Risk 95% (positive)
            max_risk_per_trade: Maximum risk per trade as % of portfolio (default 2%)

        Returns:
            position_size_pct: Recommended position size (0-10%)
        """
        try:
            if stop_loss_pct is None or stop_loss_pct >= 0:
                return None

            # Base position size from stop loss
            base_position = max_risk_per_trade / abs(stop_loss_pct) * 100

            # Adjust by VaR if available (more volatile = smaller position)
            if var_95 and var_95 > 0:
                var_adjustment = min(1.0, 2.0 / var_95)  # Reduce if VaR > 2%
                base_position *= var_adjustment

            # Cap at 10% maximum position size
            position_size_pct = min(10.0, max(1.0, base_position))

            return round(position_size_pct, 1)

        except Exception as e:
            logger.error(f"{symbol}: Position size calculation failed - {e}")
            return None

    # ========================================================================
    # 4. IV Percentile (US-specific)
    # ========================================================================

    async def calculate_iv_percentile(
        self, symbol: str, analysis_date: date
    ) -> Optional[int]:
        """
        Calculate IV Percentile - where current IV ranks in 1-year range

        Uses us_option_daily_summary table.
        Stored in us_stock_grade (requires 1-year data aggregation).

        Logic:
            - Get 252 trading days of IV data
            - Calculate percentile rank of current IV
            - 0 = lowest IV in year, 100 = highest IV in year

        Interpretation:
            - < 20: Low IV (cheap options, good for buying)
            - 20-80: Normal IV
            - > 80: High IV (expensive options, good for selling)

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date

        Returns:
            iv_percentile: 0-100 or None if insufficient data
        """
        query = """
        SELECT avg_implied_volatility
        FROM us_option_daily_summary
        WHERE symbol = $1
          AND date >= $2 - INTERVAL '252 days'
          AND date <= $2
          AND avg_implied_volatility IS NOT NULL
        ORDER BY date DESC
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)

            if not result or len(result) < 20:
                return None

            current_iv = float(result[0]['avg_implied_volatility'])
            iv_list = [float(r['avg_implied_volatility']) for r in result]

            # Calculate percentile
            count_below = sum(1 for iv in iv_list if iv <= current_iv)
            iv_percentile = (count_below / len(iv_list)) * 100

            return round(iv_percentile)

        except Exception as e:
            logger.error(f"{symbol}: IV percentile calculation failed - {e}")
            return None

    # ========================================================================
    # 5. Insider Signal (US-specific)
    # ========================================================================

    async def calculate_insider_signal(
        self, symbol: str, analysis_date: date
    ) -> Optional[str]:
        """
        Calculate insider trading signal from last 30 days

        Uses us_insider_transactions table.
        Stored in us_stock_grade (requires 30-day aggregation + judgment).

        Logic:
            - Aggregate buy/sell shares over 30 days
            - CEO/CFO trades weighted 2x
            - Net shares > 10,000: STRONG_BUY
            - Net shares > 0: BUY
            - Net shares < -10,000: STRONG_SELL
            - Net shares < 0: SELL
            - Otherwise: NEUTRAL

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date

        Returns:
            insider_signal: 'STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL'
        """
        query = """
        SELECT
            acquisition_or_disposal,
            shares,
            executive_title
        FROM us_insider_transactions
        WHERE symbol = $1
          AND date >= $2 - INTERVAL '30 days'
          AND date <= $2
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)

            if not result:
                return 'NEUTRAL'

            buy_shares = 0
            sell_shares = 0

            for r in result:
                shares = float(r['shares']) if r['shares'] else 0
                titles = r['executive_title'] or []

                # CEO/CFO trades weighted 2x
                weight = 2 if any('CEO' in str(t) or 'CFO' in str(t) for t in titles) else 1

                if r['acquisition_or_disposal'] == 'A':
                    buy_shares += shares * weight
                else:
                    sell_shares += shares * weight

            net_shares = buy_shares - sell_shares

            if net_shares > 10000:
                return 'STRONG_BUY'
            elif net_shares > 0:
                return 'BUY'
            elif net_shares < -10000:
                return 'STRONG_SELL'
            elif net_shares < 0:
                return 'SELL'
            else:
                return 'NEUTRAL'

        except Exception as e:
            logger.error(f"{symbol}: Insider signal calculation failed - {e}")
            return 'NEUTRAL'

    # ========================================================================
    # 6. Action Triggers Generation
    # ========================================================================

    def generate_triggers(
        self,
        final_score: float,
        sector_percentile: Optional[float],
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
        insider_signal: Optional[str]
    ) -> Dict[str, List[str]]:
        """
        Generate action triggers for buy/sell/hold decisions

        Args:
            final_score: Current final score
            sector_percentile: Sector ranking percentile
            stop_loss_pct: Stop loss percentage (negative)
            take_profit_pct: Take profit percentage (positive)
            insider_signal: Current insider signal

        Returns:
            {
                'buy_triggers': List[str],
                'sell_triggers': List[str],
                'hold_triggers': List[str]
            }
        """
        buy_triggers = [
            f"final_score {final_score + 5:.0f}+ (current: {final_score:.0f})"
        ]

        if sector_percentile:
            buy_triggers.append(
                f"Sector rank top {max(5, sector_percentile - 10):.0f}%"
            )

        # US-specific buy triggers
        buy_triggers.append("Put/Call < 0.7 (check us_option_daily_summary)")

        if insider_signal == 'NEUTRAL':
            buy_triggers.append("Insider net buy (CEO/CFO acquisition)")

        buy_triggers.append("EPS revision +3% (check us_earnings_estimates)")

        # Sell triggers
        sell_triggers = [
            f"final_score < {final_score - 15:.0f} (current: {final_score:.0f})"
        ]

        if stop_loss_pct:
            sell_triggers.append(f"Price hits stop loss {stop_loss_pct:.1f}%")

        if take_profit_pct:
            sell_triggers.append(f"Price hits take profit +{take_profit_pct:.1f}%")

        # US-specific sell triggers
        sell_triggers.append("Put/Call > 1.2 (check us_option_daily_summary)")

        if insider_signal in ['NEUTRAL', 'BUY', 'STRONG_BUY']:
            sell_triggers.append("Insider net sell (CEO/CFO disposal)")

        # Hold triggers
        hold_triggers = [
            f"final_score {final_score - 5:.0f}~{final_score + 5:.0f}",
            "Earnings within 1 week (volatility expected)"
        ]

        return {
            'buy_triggers': buy_triggers,
            'sell_triggers': sell_triggers,
            'hold_triggers': hold_triggers
        }

    # ========================================================================
    # 7. Scenario Probability (Placeholder - High Complexity)
    # ========================================================================

    async def calculate_scenario_probability(
        self,
        symbol: str,
        analysis_date: date,
        final_score: float,
        sector: str
    ) -> Dict[str, Any]:
        """
        Calculate scenario probabilities based on historical patterns

        This is a simplified version. Full implementation requires:
        - Historical pattern matching
        - Machine learning models
        - Backtesting validation

        Current Logic (Simplified):
            - Based on final_score ranges
            - High score (75+): Higher bullish probability
            - Medium score (50-75): Higher sideways probability
            - Low score (<50): Higher bearish probability

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            final_score: Current final score
            sector: Stock sector

        Returns:
            {
                'scenario_bullish_prob': int (0-100),
                'scenario_sideways_prob': int (0-100),
                'scenario_bearish_prob': int (0-100),
                'scenario_bullish_return': str,
                'scenario_sideways_return': str,
                'scenario_bearish_return': str,
                'scenario_sample_count': int
            }
        """
        try:
            # Simplified probability based on score
            if final_score >= 75:
                bullish_prob = 55
                sideways_prob = 30
                bearish_prob = 15
            elif final_score >= 60:
                bullish_prob = 40
                sideways_prob = 40
                bearish_prob = 20
            elif final_score >= 45:
                bullish_prob = 25
                sideways_prob = 45
                bearish_prob = 30
            else:
                bullish_prob = 15
                sideways_prob = 35
                bearish_prob = 50

            # Expected returns (simplified estimates)
            return {
                'scenario_bullish_prob': bullish_prob,
                'scenario_sideways_prob': sideways_prob,
                'scenario_bearish_prob': bearish_prob,
                'scenario_bullish_return': '+10~20%',
                'scenario_sideways_return': '-5~+5%',
                'scenario_bearish_return': '-10~-20%',
                'scenario_sample_count': None  # No actual sample in simplified version
            }

        except Exception as e:
            logger.error(f"{symbol}: Scenario calculation failed - {e}")
            return {
                'scenario_bullish_prob': None,
                'scenario_sideways_prob': None,
                'scenario_bearish_prob': None,
                'scenario_bullish_return': None,
                'scenario_sideways_return': None,
                'scenario_bearish_return': None,
                'scenario_sample_count': None
            }

    # ========================================================================
    # Main Entry Point: Calculate All Metrics
    # ========================================================================

    async def calculate_all(
        self,
        symbol: str,
        analysis_date: date,
        current_score: float,
        sector: str,
        var_95: Optional[float] = None,
        sector_percentile: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate all agent metrics for a stock

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            current_score: Current final_score
            sector: Stock sector
            var_95: VaR 95% value (optional, for position sizing)
            sector_percentile: Sector percentile rank (optional, for triggers)

        Returns:
            Dictionary containing all calculated metrics
        """
        # Parallel execution of independent calculations
        atr_task = self.calculate_atr_stop_take_profit(symbol, analysis_date)
        timing_task = self.calculate_entry_timing_score(symbol, analysis_date, current_score)
        iv_task = self.calculate_iv_percentile(symbol, analysis_date)
        insider_task = self.calculate_insider_signal(symbol, analysis_date)
        scenario_task = self.calculate_scenario_probability(symbol, analysis_date, current_score, sector)

        atr_result, timing_result, iv_percentile, insider_signal, scenario_result = await asyncio.gather(
            atr_task, timing_task, iv_task, insider_task, scenario_task
        )

        # Position size (depends on ATR result)
        position_size_pct = await self.calculate_position_size(
            symbol,
            analysis_date,
            atr_result.get('stop_loss_pct'),
            var_95
        )

        # Generate triggers (synchronous)
        triggers = self.generate_triggers(
            final_score=current_score,
            sector_percentile=sector_percentile,
            stop_loss_pct=atr_result.get('stop_loss_pct'),
            take_profit_pct=atr_result.get('take_profit_pct'),
            insider_signal=insider_signal
        )

        return {
            # ATR-based risk management
            'atr_pct': atr_result.get('atr_pct'),
            'stop_loss_pct': atr_result.get('stop_loss_pct'),
            'take_profit_pct': atr_result.get('take_profit_pct'),
            'risk_reward_ratio': atr_result.get('risk_reward_ratio'),

            # Entry timing
            'entry_timing_score': timing_result.get('entry_timing_score'),
            'score_trend_2w': timing_result.get('score_trend_2w'),
            'price_position_52w': timing_result.get('price_position_52w'),

            # Position sizing
            'position_size_pct': position_size_pct,

            # US-specific (stored in us_stock_grade)
            'iv_percentile': iv_percentile,
            'insider_signal': insider_signal,

            # Triggers
            'buy_triggers': triggers['buy_triggers'],
            'sell_triggers': triggers['sell_triggers'],
            'hold_triggers': triggers['hold_triggers'],

            # Scenario analysis
            'scenario_bullish_prob': scenario_result.get('scenario_bullish_prob'),
            'scenario_sideways_prob': scenario_result.get('scenario_sideways_prob'),
            'scenario_bearish_prob': scenario_result.get('scenario_bearish_prob'),
            'scenario_bullish_return': scenario_result.get('scenario_bullish_return'),
            'scenario_sideways_return': scenario_result.get('scenario_sideways_return'),
            'scenario_bearish_return': scenario_result.get('scenario_bearish_return'),
            'scenario_sample_count': scenario_result.get('scenario_sample_count')
        }
