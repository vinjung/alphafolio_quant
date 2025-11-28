"""
Additional Quantitative Metrics Calculator
추가 정량 지표 계산기

Calculates 13 metrics for comprehensive stock analysis:

Fundamental & Risk Metrics (9):
1. Analysis Confidence Score
2. Volatility (Annualized)
3. Expected Price Range (68% CI)
4. Downside Risk (VaR 95%)
5. Institutional/Foreign Flow
6. Factor Momentum
7. Industry Ranking
8. Maximum Drawdown (1Y)
9. Beta (Estimated)

Technical Indicators (4):
10. Support/Resistance Levels (2 levels each)
11. SuperTrend Indicator
12. Relative Strength (RS)
13. RS Rank

All results designed for kr_stock_grade table storage.
"""

import os
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdditionalMetricsCalculator:
    """
    Calculate additional quantitative metrics for stock analysis
    Designed for kr_stock_grade table storage
    """

    def __init__(self, symbol: str, db_manager, factor_scores: Dict = None, analysis_date=None):
        """
        Initialize calculator

        Args:
            symbol: Stock symbol
            db_manager: AsyncDatabaseManager instance
            factor_scores: Dict containing factor analysis results from kr_main
            analysis_date: Optional analysis date (defaults to current date)
        """
        self.symbol = symbol
        self.db_manager = db_manager
        self.factor_scores = factor_scores or {}
        self.analysis_date = analysis_date
        logger.info(f"Additional metrics calculator initialized for {symbol}")

    async def execute_query(self, query: str, *params):
        """Execute SQL query and return results"""
        try:
            return await self.db_manager.execute_query(query, *params)
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return None

    async def calculate_confidence_score(self) -> float:
        """
        1. Calculate analysis confidence score (0-100)
        Based on data completeness and valid strategy count

        Storage: confidence_score FLOAT
        """
        try:
            total_strategies = 0
            valid_strategies = 0

            # Count valid strategies from each factor
            for factor_name in ['value', 'quality', 'momentum', 'growth']:
                factor_data = self.factor_scores.get(factor_name, {})
                strategies = factor_data.get('strategies', {})

                for score in strategies.values():
                    total_strategies += 1
                    if score is not None:
                        valid_strategies += 1

            if total_strategies == 0:
                return 0.0

            # Data completeness ratio
            completeness = (valid_strategies / total_strategies) * 100

            # Additional confidence factors
            # Check if all 4 factors have weighted results
            weighted_results_count = sum(
                1 for factor in ['value', 'quality', 'momentum', 'growth']
                if self.factor_scores.get(factor, {}).get('weighted_result') is not None
            )
            factor_confidence = (weighted_results_count / 4) * 100

            # Combined confidence score
            confidence = (completeness * 0.7) + (factor_confidence * 0.3)

            logger.info(f"Confidence score: {confidence:.1f}% ({valid_strategies}/{total_strategies} strategies)")
            return round(confidence, 1)

        except Exception as e:
            logger.error(f"Confidence score calculation failed: {e}")
            return 0.0

    async def calculate_volatility_annual(self) -> Optional[float]:
        """
        2. Calculate annualized volatility (%)
        Based on 90-day daily returns

        Storage: volatility_annual FLOAT
        """
        try:
            query = """
            WITH daily_returns AS (
                SELECT
                    date,
                    close,
                    LAG(close) OVER (ORDER BY date) as prev_close,
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            )
            SELECT
                STDDEV(daily_return) as std_dev,
                COUNT(*) as days_count
            FROM daily_returns
            WHERE daily_return IS NOT NULL
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or not result[0]['std_dev']:
                return None

            daily_std = float(result[0]['std_dev'])
            days_count = int(result[0]['days_count'])

            # Annualize: daily_std * sqrt(252)
            annual_volatility = daily_std * np.sqrt(252)

            logger.info(f"Annual volatility: {annual_volatility:.2f}% (based on {days_count} days)")
            return round(annual_volatility, 2)

        except Exception as e:
            logger.error(f"Volatility calculation failed: {e}")
            return None

    async def calculate_expected_range(self, volatility: float = None) -> dict:
        """
        3. Calculate expected price range (68% confidence interval)
        Current price ± 1 standard deviation (3 months and 1 year)

        Storage: expected_range_3m_min, expected_range_3m_max, expected_range_1y_min, expected_range_1y_max

        Returns:
            dict with keys: range_3m_min, range_3m_max, range_1y_min, range_1y_max
        """
        try:
            # Get current price
            query = """
            SELECT close as current_price
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date = $2)
            ORDER BY date DESC
            LIMIT 1
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or not result[0]['current_price']:
                return None

            current_price = float(result[0]['current_price'])

            # Get or calculate volatility
            if volatility is None:
                volatility = await self.calculate_volatility_annual()
                if volatility is None:
                    return None

            # Convert annual volatility to 3-month volatility
            # 3 months ≈ 63 trading days
            volatility_3m = volatility / np.sqrt(252) * np.sqrt(63)

            # 1 year = 252 trading days (use annual volatility as-is)
            volatility_1y = volatility

            # Calculate 3-month range (current price ± volatility%)
            range_3m_min = current_price * (1 - volatility_3m / 100)
            range_3m_max = current_price * (1 + volatility_3m / 100)

            # Calculate 1-year range (current price ± volatility%)
            range_1y_min = current_price * (1 - volatility_1y / 100)
            range_1y_max = current_price * (1 + volatility_1y / 100)

            logger.info(f"Expected range (3M, 68% CI): {range_3m_min:.0f} - {range_3m_max:.0f} (current: {current_price:.0f})")
            logger.info(f"Expected range (1Y, 68% CI): {range_1y_min:.0f} - {range_1y_max:.0f} (current: {current_price:.0f})")

            return {
                'range_3m_min': round(range_3m_min, 2),
                'range_3m_max': round(range_3m_max, 2),
                'range_1y_min': round(range_1y_min, 2),
                'range_1y_max': round(range_1y_max, 2)
            }

        except Exception as e:
            logger.error(f"Expected range calculation failed: {e}")
            return None

    async def calculate_var_95(self) -> Optional[float]:
        """
        4. Calculate Value at Risk (VaR 95%)
        5th percentile of 90-day return distribution

        Storage: var_95 FLOAT (negative value, %)
        """
        try:
            query = """
            WITH daily_returns AS (
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            )
            SELECT
                PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY daily_return) as var_95
            FROM daily_returns
            WHERE daily_return IS NOT NULL
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or result[0]['var_95'] is None:
                return None

            var_95 = float(result[0]['var_95'])

            logger.info(f"VaR 95%: {var_95:.2f}% (95% chance loss won't exceed this)")
            return round(var_95, 2)

        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return None

    async def calculate_cvar_95(self) -> Optional[float]:
        """
        4-1. Calculate Conditional Value at Risk (CVaR 95%) = Expected Shortfall
        Average of returns below VaR 95% threshold

        CVaR is more conservative than VaR as it considers the tail risk
        - VaR 95%: "95% of the time, loss won't exceed X%"
        - CVaR 95%: "When loss exceeds VaR, average loss is Y%"

        Storage: cvar_95 FLOAT (negative value, %)
        """
        try:
            query = """
            WITH daily_returns AS (
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            ),
            var_threshold AS (
                SELECT PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY daily_return) as var_95
                FROM daily_returns
                WHERE daily_return IS NOT NULL
            )
            SELECT
                AVG(dr.daily_return) as cvar_95,
                COUNT(*) as tail_count
            FROM daily_returns dr, var_threshold vt
            WHERE dr.daily_return IS NOT NULL
                AND dr.daily_return <= vt.var_95
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or result[0]['cvar_95'] is None:
                return None

            cvar_95 = float(result[0]['cvar_95'])
            tail_count = int(result[0]['tail_count']) if result[0]['tail_count'] else 0

            logger.info(f"CVaR 95%: {cvar_95:.2f}% (avg loss in worst {tail_count} days)")
            return round(cvar_95, 2)

        except Exception as e:
            logger.error(f"CVaR calculation failed: {e}")
            return None

    # ========================================================================
    # 멀티 에이전트 AI용 신규 지표 (Phase Agent)
    # ========================================================================

    async def calculate_atr_stop_take_profit(self) -> dict:
        """
        ATR 기반 손절/익절 계산 (Phase Agent)

        공식:
            ATR% = 14일 ATR / 현재가 × 100
            stop_loss = -2 × ATR%
            take_profit = 3 × ATR% (기본 RR 1.5)

        Returns:
            dict: {
                'atr_pct': ATR % (14일),
                'stop_loss_pct': 손절 기준 % (음수),
                'take_profit_pct': 익절 기준 % (양수),
                'risk_reward_ratio': 리스크리워드 비율
            }
        """
        try:
            query = """
            WITH daily_data AS (
                SELECT
                    date,
                    high,
                    low,
                    close,
                    LAG(close) OVER (ORDER BY date) as prev_close
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY date DESC
                LIMIT 15
            ),
            true_range AS (
                SELECT
                    date,
                    GREATEST(
                        high - low,
                        ABS(high - COALESCE(prev_close, high)),
                        ABS(low - COALESCE(prev_close, low))
                    ) as tr,
                    close
                FROM daily_data
                WHERE prev_close IS NOT NULL
            )
            SELECT
                AVG(tr) as atr,
                (SELECT close FROM true_range ORDER BY date DESC LIMIT 1) as current_close
            FROM true_range
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or not result[0]['atr'] or not result[0]['current_close']:
                return None

            atr = float(result[0]['atr'])
            close = float(result[0]['current_close'])
            atr_pct = (atr / close) * 100

            # 손절: 2 ATR, 익절: 3 ATR (RR 1.5)
            stop_loss_pct = -2 * atr_pct
            take_profit_pct = 3 * atr_pct
            risk_reward = abs(take_profit_pct / stop_loss_pct) if stop_loss_pct != 0 else 0

            logger.info(f"ATR Stop/Take: ATR%={atr_pct:.2f}%, SL={stop_loss_pct:.2f}%, TP={take_profit_pct:.2f}%, RR={risk_reward:.2f}")

            return {
                'atr_pct': round(atr_pct, 2),
                'stop_loss_pct': round(stop_loss_pct, 2),
                'take_profit_pct': round(take_profit_pct, 2),
                'risk_reward_ratio': round(risk_reward, 2)
            }

        except Exception as e:
            logger.error(f"ATR stop/take profit calculation failed: {e}")
            return None

    async def calculate_entry_timing_score(self, final_score: float, momentum_score: float) -> dict:
        """
        진입 타이밍 점수 계산 (Phase Agent)

        공식:
            entry_timing = final_score_level × 0.4
                         + final_score_trend × 0.3
                         + price_position × 0.2
                         + momentum_confirm × 0.1

        Args:
            final_score: 현재 final_score
            momentum_score: 현재 momentum_score

        Returns:
            dict: {
                'entry_timing_score': 진입 타이밍 점수 (0-100),
                'score_trend_2w': 2주간 final_score 변화,
                'price_position_52w': 52주 고저 대비 위치 (0-100%)
            }
        """
        try:
            # 1. 2주 전 final_score 조회
            score_query = """
            SELECT final_score
            FROM kr_stock_grade
            WHERE symbol = $1
                AND date <= COALESCE($2::date, CURRENT_DATE) - INTERVAL '14 days'
            ORDER BY date DESC
            LIMIT 1
            """
            score_result = await self.execute_query(score_query, self.symbol, self.analysis_date)
            score_2w_ago = float(score_result[0]['final_score']) if score_result and score_result[0]['final_score'] else final_score

            # 2. 52주 고저 대비 위치
            position_query = """
            SELECT
                close as current_close,
                MIN(close) OVER () as low_52w,
                MAX(close) OVER () as high_52w
            FROM kr_intraday_total
            WHERE symbol = $1
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '252 days'
                AND date <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY date DESC
            LIMIT 1
            """
            position_result = await self.execute_query(position_query, self.symbol, self.analysis_date)

            if not position_result:
                return None

            current = float(position_result[0]['current_close'])
            low_52w = float(position_result[0]['low_52w'])
            high_52w = float(position_result[0]['high_52w'])

            # 52주 고저 대비 위치 (0-100%)
            price_range = high_52w - low_52w
            price_position = ((current - low_52w) / price_range * 100) if price_range > 0 else 50

            # 점수 계산
            score_level = final_score  # 0-100
            score_trend = final_score - score_2w_ago  # -20 ~ +20 예상
            trend_normalized = min(100, max(0, 50 + score_trend * 2.5))  # 0-100 정규화
            position_score = 100 - price_position  # 저점일수록 높은 점수
            momentum_confirm = 70 if momentum_score > 50 else 30

            # 가중 평균
            entry_timing = (
                score_level * 0.4 +
                trend_normalized * 0.3 +
                position_score * 0.2 +
                momentum_confirm * 0.1
            )

            logger.info(f"Entry Timing: {entry_timing:.1f} (level={score_level:.1f}, trend={score_trend:+.1f}, pos={price_position:.1f}%)")

            return {
                'entry_timing_score': round(entry_timing, 1),
                'score_trend_2w': round(score_trend, 2),
                'price_position_52w': round(price_position, 2)
            }

        except Exception as e:
            logger.error(f"Entry timing score calculation failed: {e}")
            return None

    async def calculate_position_size(self, volatility_annual: float = None) -> float:
        """
        변동성 기반 포지션 사이징 (Phase Agent)

        공식:
            base_allocation = 5%
            volatility_factor = 시장평균변동성 / 종목변동성
            position_size = base_allocation × volatility_factor

        제한: 최소 1%, 최대 10%

        Args:
            volatility_annual: 연환산 변동성 (없으면 계산)

        Returns:
            float: 권장 포트폴리오 비중 %
        """
        try:
            # 변동성 가져오기
            if volatility_annual is None:
                volatility_annual = await self.calculate_volatility_annual()

            if volatility_annual is None or volatility_annual <= 0:
                return 3.0  # 기본값

            # 시장 평균 변동성 (KOSPI 기준 약 20%)
            market_vol = 20.0

            # 변동성 역가중
            volatility_factor = market_vol / volatility_annual

            # 기본 배분 5%
            base_allocation = 5.0
            position_size = base_allocation * volatility_factor

            # 범위 제한 (1% ~ 10%)
            position_size = min(10.0, max(1.0, position_size))

            logger.info(f"Position Size: {position_size:.2f}% (vol={volatility_annual:.1f}%, factor={volatility_factor:.2f})")

            return round(position_size, 2)

        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 3.0  # 기본값

    def generate_action_triggers(self, final_score: float, sector_percentile: float,
                                  stop_loss_pct: float, take_profit_pct: float) -> dict:
        """
        행동 트리거 자동 생성 (Phase Agent)

        Args:
            final_score: 현재 final_score
            sector_percentile: 섹터 내 백분위
            stop_loss_pct: 손절 기준 %
            take_profit_pct: 익절 기준 %

        Returns:
            dict: {
                'buy_triggers': 매수 조건 리스트,
                'sell_triggers': 매도 조건 리스트,
                'hold_triggers': 보유 유지 조건 리스트
            }
        """
        try:
            buy_triggers = [
                f"final_score {final_score + 5:.0f}점 이상 상승 시",
                "외국인 5일 연속 순매수 전환 시",
                f"섹터 순위 상위 {max(5, sector_percentile - 10):.0f}% 진입 시"
            ]

            sell_triggers = [
                f"final_score {final_score - 15:.0f}점 이하 하락 시",
                f"손절선 {stop_loss_pct:.1f}% 도달 시",
                f"익절선 +{take_profit_pct:.1f}% 도달 시",
                "기관/외국인 10일 연속 순매도 시"
            ]

            hold_triggers = [
                f"final_score {final_score - 5:.0f}~{final_score + 5:.0f}점 유지 시",
                "섹터 순위 현재 수준 유지 시",
                "주요 지표 급변 없을 시"
            ]

            logger.info(f"Generated triggers: buy={len(buy_triggers)}, sell={len(sell_triggers)}, hold={len(hold_triggers)}")

            return {
                'buy_triggers': buy_triggers,
                'sell_triggers': sell_triggers,
                'hold_triggers': hold_triggers
            }

        except Exception as e:
            logger.error(f"Trigger generation failed: {e}")
            return {
                'buy_triggers': [],
                'sell_triggers': [],
                'hold_triggers': []
            }

    async def calculate_scenario_probability(self, final_score: float, industry: str) -> dict:
        """
        백테스트 기반 시나리오 확률 계산 (Phase Agent)

        과정:
            1. 과거 유사 조건 케이스 검색 (final_score ±5, 동일 섹터)
            2. 3개월 후 결과 집계:
               - 강세: >+10%
               - 횡보: -10% ~ +10%
               - 약세: <-10%

        Args:
            final_score: 현재 final_score
            industry: 업종

        Returns:
            dict: 시나리오 확률 및 예상 수익률
        """
        try:
            query = """
            WITH similar_cases AS (
                SELECT
                    g.symbol,
                    g.date as analysis_date,
                    g.final_score,
                    current_price.close as current_close,
                    future_price.close as future_close,
                    CASE
                        WHEN future_price.close IS NOT NULL AND current_price.close > 0
                        THEN (future_price.close - current_price.close) / current_price.close * 100
                        ELSE NULL
                    END as return_3m
                FROM kr_stock_grade g
                JOIN kr_stock_detail d ON g.symbol = d.symbol
                JOIN kr_intraday_total current_price
                    ON g.symbol = current_price.symbol AND g.date = current_price.date
                LEFT JOIN LATERAL (
                    SELECT close
                    FROM kr_intraday_total
                    WHERE symbol = g.symbol
                        AND date >= g.date + INTERVAL '60 days'
                        AND date <= g.date + INTERVAL '66 days'
                    ORDER BY date
                    LIMIT 1
                ) future_price ON true
                WHERE g.final_score BETWEEN $1 - 5 AND $1 + 5
                    AND d.industry = $2
                    AND g.date < CURRENT_DATE - INTERVAL '63 days'
                    AND future_price.close IS NOT NULL
            )
            SELECT
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE return_3m > 10) as bullish_count,
                COUNT(*) FILTER (WHERE return_3m BETWEEN -10 AND 10) as sideways_count,
                COUNT(*) FILTER (WHERE return_3m < -10) as bearish_count,
                ROUND(AVG(return_3m) FILTER (WHERE return_3m > 10)::numeric, 1) as bullish_avg,
                ROUND(AVG(return_3m) FILTER (WHERE return_3m BETWEEN -10 AND 10)::numeric, 1) as sideways_avg,
                ROUND(AVG(return_3m) FILTER (WHERE return_3m < -10)::numeric, 1) as bearish_avg,
                ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY return_3m)
                    FILTER (WHERE return_3m > 10)::numeric, 1) as bullish_lower,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY return_3m)
                    FILTER (WHERE return_3m > 10)::numeric, 1) as bullish_upper,
                ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY return_3m)
                    FILTER (WHERE return_3m < -10)::numeric, 1) as bearish_lower,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY return_3m)
                    FILTER (WHERE return_3m < -10)::numeric, 1) as bearish_upper
            FROM similar_cases
            """

            result = await self.execute_query(query, final_score, industry)

            if not result or not result[0]['total_count'] or result[0]['total_count'] < 10:
                # 샘플 부족 시 기본값
                logger.warning(f"Insufficient samples for scenario analysis (industry={industry})")
                return {
                    'scenario_bullish_prob': 33,
                    'scenario_sideways_prob': 34,
                    'scenario_bearish_prob': 33,
                    'scenario_bullish_return': "+10~20%",
                    'scenario_sideways_return': "-5~+5%",
                    'scenario_bearish_return': "-15~-10%",
                    'scenario_sample_count': 0
                }

            total = result[0]['total_count']
            bullish = result[0]['bullish_count'] or 0
            sideways = result[0]['sideways_count'] or 0
            bearish = result[0]['bearish_count'] or 0

            # 확률 계산
            bullish_prob = round(bullish / total * 100) if total > 0 else 33
            sideways_prob = round(sideways / total * 100) if total > 0 else 34
            bearish_prob = round(bearish / total * 100) if total > 0 else 33

            # 예상 수익률 범위
            bullish_lower = result[0]['bullish_lower'] or 10
            bullish_upper = result[0]['bullish_upper'] or 25
            bearish_lower = result[0]['bearish_lower'] or -20
            bearish_upper = result[0]['bearish_upper'] or -10

            logger.info(f"Scenario: bullish={bullish_prob}%, sideways={sideways_prob}%, bearish={bearish_prob}% (n={total})")

            return {
                'scenario_bullish_prob': bullish_prob,
                'scenario_sideways_prob': sideways_prob,
                'scenario_bearish_prob': bearish_prob,
                'scenario_bullish_return': f"+{bullish_lower:.0f}~+{bullish_upper:.0f}%",
                'scenario_sideways_return': "-10~+10%",
                'scenario_bearish_return': f"{bearish_upper:.0f}~{bearish_lower:.0f}%",
                'scenario_sample_count': total
            }

        except Exception as e:
            logger.error(f"Scenario probability calculation failed: {e}")
            return {
                'scenario_bullish_prob': 33,
                'scenario_sideways_prob': 34,
                'scenario_bearish_prob': 33,
                'scenario_bullish_return': "+10~20%",
                'scenario_sideways_return': "-5~+5%",
                'scenario_bearish_return': "-15~-10%",
                'scenario_sample_count': 0
            }

    async def calculate_investor_flow(self) -> Tuple[Optional[int], Optional[int]]:
        """
        5. Calculate institutional and foreign investor flow (30 days)
        Net buying volume in shares

        Storage: inst_net_30d BIGINT, foreign_net_30d BIGINT
        """
        try:
            query = """
            SELECT
                SUM(inst_net_volume) as inst_net_30d,
                SUM(foreign_net_volume) as foreign_net_30d
            FROM kr_individual_investor_daily_trading
            WHERE symbol = $1
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '30 days'
                AND date <= COALESCE($2::date, CURRENT_DATE)
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result:
                return None, None

            inst_net = result[0]['inst_net_30d']
            foreign_net = result[0]['foreign_net_30d']

            inst_net = int(inst_net) if inst_net else 0
            foreign_net = int(foreign_net) if foreign_net else 0

            logger.info(f"30-day flow - Institution: {inst_net:,} shares, Foreign: {foreign_net:,} shares")
            return inst_net, foreign_net

        except Exception as e:
            logger.error(f"Investor flow calculation failed: {e}")
            return None, None

    async def calculate_factor_momentum(self) -> Dict[str, Optional[float]]:
        """
        6. Calculate factor momentum (trend)
        Based on factor score time series if available, otherwise use current vs benchmark

        Storage: value_momentum, quality_momentum, momentum_momentum, growth_momentum FLOAT

        Returns simple indicator: positive (growing), negative (declining), 0 (neutral)
        """
        try:
            # For now, use simplified approach based on current factor scores vs 50 (neutral)
            # In full implementation, would compare current vs historical factor scores

            factor_momentum = {}

            for factor_name in ['value', 'quality', 'momentum', 'growth']:
                factor_data = self.factor_scores.get(factor_name, {})
                weighted_result = factor_data.get('weighted_result')

                if weighted_result:
                    score = weighted_result['weighted_score']
                    # Momentum indicator: score > 50 = positive, < 50 = negative
                    momentum = score - 50
                    factor_momentum[f'{factor_name}_momentum'] = round(momentum, 2)
                else:
                    factor_momentum[f'{factor_name}_momentum'] = None

            logger.info(f"Factor momentum: {factor_momentum}")
            return factor_momentum

        except Exception as e:
            logger.error(f"Factor momentum calculation failed: {e}")
            return {
                'value_momentum': None,
                'quality_momentum': None,
                'momentum_momentum': None,
                'growth_momentum': None
            }

    async def calculate_industry_ranking(self) -> Tuple[Optional[int], Optional[float]]:
        """
        7. Calculate industry ranking
        Rank within same industry based on final score (if available) or market cap

        Storage: industry_rank INTEGER, industry_percentile FLOAT
        """
        try:
            # Get stock's industry
            query_industry = """
            SELECT industry
            FROM kr_stock_detail
            WHERE symbol = $1
            """

            result = await self.execute_query(query_industry, self.symbol)

            if not result or not result[0]['industry']:
                return None, None

            industry = result[0]['industry']

            # Rank by market cap within industry (since we don't have historical scores yet)
            query_rank = """
            WITH industry_stocks AS (
                SELECT
                    sd.symbol,
                    it.market_cap,
                    ROW_NUMBER() OVER (ORDER BY it.market_cap DESC NULLS LAST) as rank,
                    COUNT(*) OVER () as total_count
                FROM kr_stock_detail sd
                JOIN kr_intraday_total it ON sd.symbol = it.symbol
                WHERE sd.industry = $1
                    AND it.date = COALESCE($3::date, (SELECT MAX(date) FROM kr_intraday_total))
                    AND it.market_cap IS NOT NULL
            )
            SELECT
                rank,
                total_count,
                ((total_count - rank + 1)::NUMERIC / total_count * 100) as percentile
            FROM industry_stocks
            WHERE symbol = $2
            """

            result = await self.db_manager.execute_query(query_rank, industry, self.symbol, self.analysis_date)

            if not result:
                return None, None

            rank = int(result[0]['rank'])
            total = int(result[0]['total_count'])
            percentile = float(result[0]['percentile'])

            logger.info(f"Industry ranking: {rank}/{total} (top {100-percentile:.1f}%)")
            return rank, round(percentile, 1)

        except Exception as e:
            logger.error(f"Industry ranking calculation failed: {e}")
            return None, None

    async def calculate_max_drawdown_1y(self) -> Optional[float]:
        """
        8. Calculate maximum drawdown over past 1 year

        Storage: max_drawdown_1y FLOAT (negative value, %)
        """
        try:
            query = """
            WITH daily_prices AS (
                SELECT
                    date,
                    close,
                    MAX(close) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_max
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '252 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            )
            SELECT
                MIN(((close - running_max)::NUMERIC / NULLIF(running_max, 0) * 100)) as max_drawdown
            FROM daily_prices
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or result[0]['max_drawdown'] is None:
                return None

            mdd = float(result[0]['max_drawdown'])

            logger.info(f"Max drawdown (1Y): {mdd:.2f}%")
            return round(mdd, 2)

        except Exception as e:
            logger.error(f"Max drawdown calculation failed: {e}")
            return None

    async def calculate_beta(self) -> Optional[float]:
        """
        9. Calculate beta vs market index (KOSPI or KOSDAQ)

        Storage: beta FLOAT
        """
        try:
            # Get stock's exchange
            query_exchange = """
            SELECT exchange
            FROM kr_stock_detail
            WHERE symbol = $1
            """

            result = await self.execute_query(query_exchange, self.symbol)

            if not result or not result[0]['exchange']:
                return None

            exchange = result[0]['exchange']

            # Use appropriate index
            # For now, estimate beta using volatility ratio as proxy
            # In full implementation, would use regression against index returns

            # Get stock volatility
            query_stock_vol = """
            WITH daily_returns AS (
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0)) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            )
            SELECT STDDEV(daily_return) as stock_std
            FROM daily_returns
            WHERE daily_return IS NOT NULL
            """

            result = await self.execute_query(query_stock_vol, self.symbol, self.analysis_date)

            if not result or not result[0]['stock_std']:
                return None

            stock_std = float(result[0]['stock_std'])

            # Estimate market std (simplified: KOSPI ~1.5%, KOSDAQ ~2.0% daily)
            market_std = 0.015 if exchange == 'KOSPI' else 0.020

            # Beta estimate = stock_std / market_std
            beta = stock_std / market_std

            logger.info(f"Beta (estimated): {beta:.2f} vs {exchange}")
            return round(beta, 2)

        except Exception as e:
            logger.error(f"Beta calculation failed: {e}")
            return None

    async def calculate_support_resistance(self) -> Dict[str, Optional[float]]:
        """
        10. Calculate support and resistance levels (2 levels each)
        Using Pivot Points + Recent highs/lows method

        Storage: support_1, support_2, resistance_1, resistance_2 DECIMAL(21,2)

        Returns:
            dict with keys: support_1, support_2, resistance_1, resistance_2
        """
        try:
            # Get recent price data (20 days for short-term, 60 days for mid-term levels)
            query = """
            WITH recent_prices AS (
                SELECT
                    high,
                    low,
                    close
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND high IS NOT NULL
                    AND low IS NOT NULL
                    AND close IS NOT NULL
                ORDER BY date DESC
                LIMIT 60
            ),
            latest_price AS (
                SELECT high, low, close
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND ($2::date IS NULL OR date = $2)
                ORDER BY date DESC
                LIMIT 1
            ),
            price_stats AS (
                SELECT
                    MAX(high) as high_60d,
                    MIN(low) as low_60d,
                    (SELECT MAX(high) FROM (SELECT high FROM recent_prices LIMIT 20) t) as high_20d,
                    (SELECT MIN(low) FROM (SELECT low FROM recent_prices LIMIT 20) t) as low_20d,
                    (SELECT high FROM latest_price) as latest_high,
                    (SELECT low FROM latest_price) as latest_low,
                    (SELECT close FROM latest_price) as latest_close
                FROM recent_prices
            )
            SELECT
                latest_high,
                latest_low,
                latest_close,
                high_20d,
                low_20d,
                high_60d,
                low_60d
            FROM price_stats
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or not result[0]['latest_close']:
                return {
                    'support_1': None,
                    'support_2': None,
                    'resistance_1': None,
                    'resistance_2': None
                }

            data = result[0]
            high = float(data['latest_high'])
            low = float(data['latest_low'])
            close = float(data['latest_close'])
            high_20d = float(data['high_20d'])
            low_20d = float(data['low_20d'])
            high_60d = float(data['high_60d'])
            low_60d = float(data['low_60d'])

            # Calculate Pivot Point
            pivot = (high + low + close) / 3

            # Calculate support and resistance using Pivot Points
            resistance_pivot_1 = (2 * pivot) - low
            resistance_pivot_2 = pivot + (high - low)
            support_pivot_1 = (2 * pivot) - high
            support_pivot_2 = pivot - (high - low)

            # Combine with recent highs/lows for more realistic levels
            # Resistance: use max of pivot calculation and recent highs
            resistance_1 = round(max(resistance_pivot_1, high_20d), 2)
            resistance_2 = round(max(resistance_pivot_2, high_60d), 2)

            # Support: use min of pivot calculation and recent lows
            support_1 = round(min(support_pivot_1, low_20d), 2)
            support_2 = round(min(support_pivot_2, low_60d), 2)

            logger.info(f"Support levels: S1={support_1}, S2={support_2}")
            logger.info(f"Resistance levels: R1={resistance_1}, R2={resistance_2}")

            return {
                'support_1': support_1,
                'support_2': support_2,
                'resistance_1': resistance_1,
                'resistance_2': resistance_2
            }

        except Exception as e:
            logger.error(f"Support/Resistance calculation failed: {e}")
            return {
                'support_1': None,
                'support_2': None,
                'resistance_1': None,
                'resistance_2': None
            }

    async def calculate_supertrend(self, period: int = 10, multiplier: float = 3.0) -> Dict[str, Optional[any]]:
        """
        11. Calculate SuperTrend indicator
        Uses ATR (Average True Range) for trend detection

        Storage: supertrend_value DECIMAL(21,2), trend TEXT, signal TEXT

        Args:
            period: ATR calculation period (default: 10)
            multiplier: ATR multiplier (default: 3.0)

        Returns:
            dict with keys: supertrend_value, trend, signal
        """
        try:
            # Get recent price data for ATR calculation
            query = """
            WITH price_data AS (
                SELECT
                    date,
                    high,
                    low,
                    close,
                    LAG(close) OVER (ORDER BY date) as prev_close
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '30 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND high IS NOT NULL
                    AND low IS NOT NULL
                    AND close IS NOT NULL
                ORDER BY date
            ),
            true_range AS (
                SELECT
                    date,
                    high,
                    low,
                    close,
                    GREATEST(
                        high - low,
                        ABS(high - prev_close),
                        ABS(low - prev_close)
                    ) as tr
                FROM price_data
                WHERE prev_close IS NOT NULL
            ),
            atr_calc AS (
                SELECT
                    date,
                    high,
                    low,
                    close,
                    tr,
                    AVG(tr) OVER (
                        ORDER BY date
                        ROWS BETWEEN $3 - 1 PRECEDING AND CURRENT ROW
                    ) as atr
                FROM true_range
            )
            SELECT
                high,
                low,
                close,
                atr
            FROM atr_calc
            ORDER BY date DESC
            LIMIT 1
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date, period)

            if not result or not result[0]['atr']:
                return {
                    'supertrend_value': None,
                    'trend': None,
                    'signal': None
                }

            data = result[0]
            high = float(data['high'])
            low = float(data['low'])
            close = float(data['close'])
            atr = float(data['atr'])

            # Calculate basic bands
            hl_avg = (high + low) / 2
            basic_upper_band = hl_avg + (multiplier * atr)
            basic_lower_band = hl_avg - (multiplier * atr)

            # Determine trend
            # If close > upper band: uptrend, use lower band as SuperTrend
            # If close < lower band: downtrend, use upper band as SuperTrend
            if close > basic_upper_band:
                supertrend_value = basic_lower_band
                trend = '상승'
                signal = '매수' if close > supertrend_value else '보유'
            elif close < basic_lower_band:
                supertrend_value = basic_upper_band
                trend = '하락'
                signal = '매도' if close < supertrend_value else '보유'
            else:
                # Neutral zone
                supertrend_value = hl_avg
                trend = '중립'
                signal = '보유'

            logger.info(f"SuperTrend: {supertrend_value:.2f}, Trend: {trend}, Signal: {signal}")

            return {
                'supertrend_value': round(supertrend_value, 2),
                'trend': trend,
                'signal': signal
            }

        except Exception as e:
            logger.error(f"SuperTrend calculation failed: {e}")
            return {
                'supertrend_value': None,
                'trend': None,
                'signal': None
            }

    async def calculate_relative_strength(self, period: int = 20) -> Dict[str, Optional[any]]:
        """
        12-13. Calculate Relative Strength (RS) vs market index
        Measures stock performance relative to its market (KOSPI/KOSDAQ)

        Storage: rs_value DECIMAL(21,2), rs_rank TEXT

        Args:
            period: Comparison period in days (default: 20)

        Returns:
            dict with keys: rs_value, rs_rank
        """
        try:
            # Get stock's exchange first
            query_exchange = """
            SELECT exchange
            FROM kr_stock_detail
            WHERE symbol = $1
            """

            result = await self.execute_query(query_exchange, self.symbol)

            if not result or not result[0]['exchange']:
                return {'rs_value': None, 'rs_rank': None}

            exchange = result[0]['exchange']

            # Determine market index symbol based on exchange
            # KOSPI: ^KS11, KOSDAQ: ^KQ11 (Korean market index symbols)
            # For simplicity, we'll calculate RS using market cap weighted average
            # In production, you would use actual index data

            query = """
            WITH target_date AS (
                SELECT COALESCE($2::date, CURRENT_DATE) as analysis_date
            ),
            stock_current AS (
                SELECT close as current_price
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date = (SELECT analysis_date FROM target_date)
                LIMIT 1
            ),
            stock_past AS (
                SELECT close as past_price
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date <= (SELECT analysis_date FROM target_date) - MAKE_INTERVAL(days => $3)
                ORDER BY date DESC
                LIMIT 1
            ),
            market_stocks AS (
                SELECT
                    sd.symbol,
                    current_prices.close as current_close,
                    (
                        SELECT close
                        FROM kr_intraday_total
                        WHERE symbol = sd.symbol
                            AND date <= (SELECT analysis_date FROM target_date) - MAKE_INTERVAL(days => $3)
                        ORDER BY date DESC
                        LIMIT 1
                    ) as past_close
                FROM kr_stock_detail sd
                JOIN kr_intraday_total current_prices ON sd.symbol = current_prices.symbol
                WHERE sd.exchange = $4
                    AND current_prices.date = (SELECT analysis_date FROM target_date)
                    AND current_prices.close IS NOT NULL
                    AND current_prices.market_cap IS NOT NULL
            ),
            market_avg AS (
                SELECT
                    AVG(
                        ((current_close - past_close)::NUMERIC / NULLIF(past_close, 0) * 100)
                    ) as market_return
                FROM market_stocks
                WHERE past_close IS NOT NULL
            )
            SELECT
                ((sc.current_price - sp.past_price)::NUMERIC / NULLIF(sp.past_price, 0) * 100) as stock_return,
                ma.market_return
            FROM stock_current sc, stock_past sp, market_avg ma
            WHERE sp.past_price IS NOT NULL
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date, period, exchange)

            if not result or result[0]['stock_return'] is None:
                return {'rs_value': None, 'rs_rank': None}

            stock_return = float(result[0]['stock_return'])
            market_return = float(result[0]['market_return']) if result[0]['market_return'] else 0.0

            # Calculate RS: difference between stock and market returns
            rs_value = stock_return - market_return

            # Determine RS rank
            if rs_value > 5.0:
                rs_rank = '매우강함'
            elif rs_value > 2.0:
                rs_rank = '강함'
            elif rs_value > -2.0:
                rs_rank = '중립'
            elif rs_value > -5.0:
                rs_rank = '약함'
            else:
                rs_rank = '매우약함'

            logger.info(f"RS ({period}d): {rs_value:.2f}% (Rank: {rs_rank})")
            logger.info(f"  Stock Return: {stock_return:.2f}%, Market Return: {market_return:.2f}%")

            return {
                'rs_value': round(rs_value, 2),
                'rs_rank': rs_rank
            }

        except Exception as e:
            logger.error(f"Relative Strength calculation failed: {e}")
            return {'rs_value': None, 'rs_rank': None}

    async def calculate_smart_money_signal_score(self) -> Tuple[Optional[int], Optional[str], Optional[int]]:
        """
        Calculate Smart Money Signal Score (0-100)
        Based on institutional and foreign investor flow patterns

        Returns validated pattern from 2,757 stock analysis:
        - Strong Inflow (>100%): +3.39% avg return -> 70-100 score
        - Accelerating Inflow: +2.51% avg return -> boosted score

        Storage: smart_money_signal_score INTEGER, smart_money_signal_text TEXT, smart_money_confidence INTEGER

        Returns:
            Tuple of (score, signal_text, confidence)
        """
        try:
            # Get current 30-day flow
            inst_net, foreign_net = await self.calculate_investor_flow()

            if inst_net is None and foreign_net is None:
                return 50, "No institutional data available", 0

            # Get 30-day flow from 1 month ago for change calculation
            query = """
            SELECT
                SUM(inst_net_volume) as inst_net_prev,
                SUM(foreign_net_volume) as foreign_net_prev
            FROM kr_individual_investor_daily_trading
            WHERE symbol = $1
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
                AND date <= COALESCE($2::date, CURRENT_DATE) - INTERVAL '30 days'
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result:
                inst_net_prev = 0
                foreign_net_prev = 0
            else:
                inst_net_prev = int(result[0]['inst_net_prev']) if result[0]['inst_net_prev'] else 0
                foreign_net_prev = int(result[0]['foreign_net_prev']) if result[0]['foreign_net_prev'] else 0

            # Calculate change percentage
            inst_change_pct = 0
            foreign_change_pct = 0

            if inst_net_prev != 0:
                inst_change_pct = ((inst_net - inst_net_prev) / abs(inst_net_prev)) * 100

            if foreign_net_prev != 0:
                foreign_change_pct = ((foreign_net - foreign_net_prev) / abs(foreign_net_prev)) * 100

            # Score calculation based on validated patterns
            score = 50  # Neutral baseline
            signal_text = ""
            confidence = 50

            # Explosive inflow (>500%)
            if inst_change_pct > 500 or foreign_change_pct > 500:
                score = 100
                signal_text = "EXPLOSIVE INSTITUTIONAL BUYING - Very strong bullish signal"
                confidence = 95

            # Very strong inflow (>200%)
            elif inst_change_pct > 200 or foreign_change_pct > 200:
                score = 85
                signal_text = "VERY STRONG INSTITUTIONAL BUYING - Strong bullish signal"
                confidence = 90

            # Strong inflow (>100%) - Validated: +3.39% avg return
            elif inst_change_pct > 100 or foreign_change_pct > 100:
                score = 70
                signal_text = "STRONG INSTITUTIONAL BUYING - Bullish signal (+3.39% historical avg)"
                confidence = 85

            # Moderate inflow (both >50%)
            elif inst_change_pct > 50 and foreign_change_pct > 50:
                score = 60
                signal_text = "MODERATE INSTITUTIONAL BUYING - Positive signal"
                confidence = 75

            # Mild inflow (both >0%)
            elif inst_change_pct > 0 and foreign_change_pct > 0:
                score = 55
                signal_text = "MILD INSTITUTIONAL BUYING - Slightly positive"
                confidence = 60

            # Mixed signals
            elif (inst_change_pct > 0 and foreign_change_pct < 0) or (inst_change_pct < 0 and foreign_change_pct > 0):
                score = 45
                signal_text = "MIXED INSTITUTIONAL SIGNALS - Uncertain"
                confidence = 40

            # Moderate outflow
            elif inst_change_pct < 0 and foreign_change_pct < 0:
                score = 30
                signal_text = "INSTITUTIONAL SELLING - Negative signal"
                confidence = 70

            # Strong outflow (<-50%) - Validated: -3.08% avg return
            elif inst_change_pct < -50 and foreign_change_pct < -50:
                score = 10
                signal_text = "STRONG INSTITUTIONAL SELLING - Bearish signal (-3.08% historical avg)"
                confidence = 85

            logger.info(f"Smart Money Score: {score}/100")
            logger.info(f"  Institutional change: {inst_change_pct:+.1f}%")
            logger.info(f"  Foreign change: {foreign_change_pct:+.1f}%")
            logger.info(f"  Signal: {signal_text}")

            return score, signal_text, confidence

        except Exception as e:
            logger.error(f"Smart money signal calculation failed: {e}")
            return 50, "Calculation error", 0

    async def calculate_volatility_context_score(self, volatility: Optional[float], smart_money_score: int) -> Tuple[Optional[int], Optional[str]]:
        """
        Calculate Volatility Context Score (0-100)
        Validates if volatility is opportunity or risk based on smart money flow

        Validated pattern from 888 stocks:
        - High Vol (>40%) + Smart Money Inflow: +5.89% avg return -> 85-90 score
        - High Vol (>40%) + Smart Money Outflow: -3.71% avg return -> 20-25 score
        - Difference: +9.60%p

        Storage: volatility_context_score INTEGER, volatility_context_text TEXT

        Returns:
            Tuple of (score, context_text)
        """
        try:
            if volatility is None:
                return 50, "Volatility data unavailable"

            # Classify volatility level
            if volatility > 70:
                vol_category = 'Very High'
            elif volatility > 50:
                vol_category = 'High'
            elif volatility > 30:
                vol_category = 'Medium'
            else:
                vol_category = 'Low'

            # Volatility context depends on smart money
            score = 50
            context_text = ""

            if vol_category == 'Very High':
                if smart_money_score >= 70:  # Strong inflow
                    score = 85
                    context_text = "Very high volatility with strong institutional buying - HIGH OPPORTUNITY"
                elif smart_money_score >= 50:  # Mild inflow
                    score = 60
                    context_text = "Very high volatility with mild institutional support - MODERATE RISK"
                else:  # Outflow or mixed
                    score = 20
                    context_text = "Very high volatility without institutional support - HIGH RISK"

            elif vol_category == 'High':
                if smart_money_score >= 70:  # Strong inflow
                    score = 90  # Best combination: +5.89% avg from 328 stocks
                    context_text = "High volatility driven by institutional accumulation - EXCELLENT OPPORTUNITY (+5.89% historical avg)"
                elif smart_money_score >= 50:
                    score = 65
                    context_text = "High volatility with institutional support - GOOD OPPORTUNITY"
                else:
                    score = 25  # Dangerous: -3.71% avg
                    context_text = "High volatility with fund exodus - STRONG RISK (-3.71% historical avg)"

            elif vol_category == 'Medium':
                if smart_money_score >= 70:
                    score = 75
                    context_text = "Medium volatility with strong buying - GOOD"
                elif smart_money_score >= 50:
                    score = 60
                    context_text = "Medium volatility - BALANCED"
                else:
                    score = 40
                    context_text = "Medium volatility with selling - CAUTION"

            else:  # Low volatility
                if smart_money_score >= 70:
                    score = 65
                    context_text = "Low volatility with institutional buying - STABLE ACCUMULATION"
                else:
                    score = 50
                    context_text = "Low volatility - NEUTRAL"

            logger.info(f"Volatility Context Score: {score}/100")
            logger.info(f"  Volatility: {volatility:.1f}% ({vol_category})")
            logger.info(f"  Context: {context_text}")

            return score, context_text

        except Exception as e:
            logger.error(f"Volatility context calculation failed: {e}")
            return 50, "Calculation error"

    async def calculate_factor_combination_bonus(self, value_score: float, quality_score: float,
                                                  momentum_score: float, growth_score: float,
                                                  smart_money_score: int) -> int:
        """
        Calculate Factor Combination Bonus (Phase 2)
        Rewards validated winning combinations from 2,757 stock analysis

        Validated patterns:
        - Low Value + High Growth + SM Inflow: +11.31% (358 stocks) -> +30 bonus
        - Med Value + High Growth + SM Inflow: +9.78% (233 stocks) -> +25 bonus
        - High Quality + High/Med Momentum + SM Inflow: +4.4~4.8% -> +15 bonus
        - Low Growth: Negative returns -> -20 penalty

        Storage: factor_combination_bonus INTEGER

        Returns:
            Bonus score (can be negative for penalties)
        """
        try:
            bonus = 0

            # Define thresholds based on 33rd, 67th percentile from analysis
            # Value: Low < 41.3 < Medium < 59.3 < High
            # Quality: Low < 64.8 < Medium < 77.6 < High
            # Momentum: Low < 32.7 < Medium < 42.0 < High
            # Growth: Low < 29.7 < Medium < 41.3 < High

            value_high = value_score > 59.3
            value_med = 41.3 < value_score <= 59.3
            value_low = value_score <= 41.3

            quality_high = quality_score > 77.6
            quality_med = 64.8 < quality_score <= 77.6

            momentum_high = momentum_score > 42.0
            momentum_med = 32.7 < momentum_score <= 42.0

            growth_high = growth_score > 41.3
            growth_med = 29.7 < growth_score <= 41.3
            growth_low = growth_score <= 29.7

            strong_sm = smart_money_score >= 70

            # Pattern 1: Growth Stock at Good Price + Smart Money
            # Low Value + High Growth + SM Inflow = +11.31% (358 stocks, 65.6% win rate)
            if value_low and growth_high and strong_sm:
                bonus += 30
                logger.info(f"  [Bonus +30] Growth stock at good price + Smart Money (+11.31% pattern)")

            # Pattern 2: Medium Value + High Growth + SM Inflow = +9.78% (233 stocks, 63.9% win rate)
            elif value_med and growth_high and strong_sm:
                bonus += 25
                logger.info(f"  [Bonus +25] Growth stock at fair price + Smart Money (+9.78% pattern)")

            # Pattern 3: Blue Chip Uptrend
            # High Quality + High/Med Momentum + SM Inflow = +4.4~4.8%
            elif quality_high and (momentum_high or momentum_med) and strong_sm:
                bonus += 15
                logger.info(f"  [Bonus +15] Blue chip uptrend + Smart Money (+4.4~4.8% pattern)")

            # Pattern 4: High Growth + Inflow (without value consideration)
            elif growth_high and strong_sm:
                bonus += 10
                logger.info(f"  [Bonus +10] High growth + Smart Money inflow")

            # Negative Pattern: Low Growth
            # Growth Low: Most had negative returns
            if growth_low:
                bonus -= 20
                logger.info(f"  [Penalty -20] Low growth stock - historically negative returns")

            logger.info(f"Factor Combination Bonus: {bonus:+d}")
            return bonus

        except Exception as e:
            logger.error(f"Factor combination bonus calculation failed: {e}")
            return 0

    async def calculate_smart_money_momentum_score(self) -> int:
        """
        Calculate Smart Money Momentum Score (Phase 2)
        Based on institutional/foreign flow persistence and acceleration

        Validated patterns from 2,757 stocks:
        - Accelerating Inflow (consistent acceleration): +2.51% (2,485 stocks) -> 80 score
        - Recent Jump Only (sudden surge): -8.08% (18 stocks) -> 20 score (RISK!)
        - Persistent Inflow: Stable support -> 60 score

        Storage: smart_money_momentum_score INTEGER

        Returns:
            Momentum score (0-100)
        """
        try:
            # Get 3 data points to check momentum/acceleration
            # We need data from ~2 months ago, ~1 month ago, and current

            # Current 30-day (most recent)
            query_current = """
            SELECT
                SUM(inst_net_volume) as inst_net,
                SUM(foreign_net_volume) as foreign_net
            FROM kr_individual_investor_daily_trading
            WHERE symbol = $1
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '30 days'
                AND date <= COALESCE($2::date, CURRENT_DATE)
            """

            # 30-day from 1 month ago
            query_1m_ago = """
            SELECT
                SUM(inst_net_volume) as inst_net,
                SUM(foreign_net_volume) as foreign_net
            FROM kr_individual_investor_daily_trading
            WHERE symbol = $1
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
                AND date <= COALESCE($2::date, CURRENT_DATE) - INTERVAL '30 days'
            """

            # 30-day from 2 months ago
            query_2m_ago = """
            SELECT
                SUM(inst_net_volume) as inst_net,
                SUM(foreign_net_volume) as foreign_net
            FROM kr_individual_investor_daily_trading
            WHERE symbol = $1
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                AND date <= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
            """

            result_current = await self.execute_query(query_current, self.symbol, self.analysis_date)
            result_1m = await self.execute_query(query_1m_ago, self.symbol, self.analysis_date)
            result_2m = await self.execute_query(query_2m_ago, self.symbol, self.analysis_date)

            if not result_current or not result_1m or not result_2m:
                return 50  # Neutral if data incomplete

            inst_current = int(result_current[0]['inst_net']) if result_current[0]['inst_net'] else 0
            foreign_current = int(result_current[0]['foreign_net']) if result_current[0]['foreign_net'] else 0

            inst_1m = int(result_1m[0]['inst_net']) if result_1m[0]['inst_net'] else 0
            foreign_1m = int(result_1m[0]['foreign_net']) if result_1m[0]['foreign_net'] else 0

            inst_2m = int(result_2m[0]['inst_net']) if result_2m[0]['inst_net'] else 0
            foreign_2m = int(result_2m[0]['foreign_net']) if result_2m[0]['foreign_net'] else 0

            # Check patterns
            inst_accelerating = inst_current > inst_1m and inst_1m > inst_2m
            foreign_accelerating = foreign_current > foreign_1m and foreign_1m > foreign_2m

            inst_persistent = inst_current > 0 and inst_1m > 0 and inst_2m > 0
            foreign_persistent = foreign_current > 0 and foreign_1m > 0 and foreign_2m > 0

            # Recent jump: current >> 1m ago, but 1m ago <= 2m ago
            inst_jump = (inst_current > inst_1m * 2) and (inst_1m <= inst_2m)
            foreign_jump = (foreign_current > foreign_1m * 2) and (foreign_1m <= foreign_2m)

            score = 50  # Neutral baseline

            # Accelerating Inflow - Validated: +2.51% (2,485 stocks)
            if inst_accelerating or foreign_accelerating:
                score = 80
                pattern = "ACCELERATING INFLOW - Consistent buying pressure (+2.51% historical avg)"

            # Recent Jump Only - Validated: -8.08% (18 stocks) - HIGH RISK!
            elif inst_jump or foreign_jump:
                score = 20
                pattern = "SUDDEN JUMP - Likely too late, high risk (-8.08% historical avg)"

            # Persistent Inflow
            elif inst_persistent or foreign_persistent:
                score = 60
                pattern = "PERSISTENT INFLOW - Ongoing support, stable"

            # Declining or No Pattern
            else:
                score = 40
                pattern = "NO CLEAR MOMENTUM - Uncertain"

            logger.info(f"Smart Money Momentum Score: {score}/100")
            logger.info(f"  Pattern: {pattern}")
            logger.info(f"  Inst Flow: 2m={inst_2m:,}, 1m={inst_1m:,}, Current={inst_current:,}")
            logger.info(f"  Foreign Flow: 2m={foreign_2m:,}, 1m={foreign_1m:,}, Current={foreign_current:,}")

            return score

        except Exception as e:
            logger.error(f"Smart money momentum calculation failed: {e}")
            return 50

    async def calculate_score_momentum_signal(self) -> Tuple[int, str, Optional[float], Optional[float]]:
        """
        Calculate score momentum signal based on historical final_score changes

        Pattern validation (Sep-Oct 2025 data):
        - Rising Fast (>+10): +48.97% avg return (92 stocks) -> 100 points
        - Rising (+5 to +10): +9.17% avg return (226 stocks) -> 70 points
        - Stable (-5 to +5): +1.26% avg return (1,942 stocks) -> 50 points
        - Falling (-10 to -5): -2.12% avg return (233 stocks) -> 30 points
        - Falling Fast (<-10): -2.35% avg return (264 stocks) -> 10 points

        Returns:
            (score, signal_text, change_1m, change_2m)
        """
        try:
            # Get current score (from factor_scores context, not yet saved to DB)
            # Note: This will be the calculated score, not from DB
            # For first-time analysis, we won't have historical data
            current_score = None
            if hasattr(self, 'current_final_score'):
                current_score = self.current_final_score
            else:
                # Calculate on the fly from factor scores (handle None weighted_result)
                value_result = self.factor_scores.get('value', {}).get('weighted_result') or {}
                quality_result = self.factor_scores.get('quality', {}).get('weighted_result') or {}
                momentum_result = self.factor_scores.get('momentum', {}).get('weighted_result') or {}
                growth_result = self.factor_scores.get('growth', {}).get('weighted_result') or {}

                value_score = value_result.get('weighted_score', 0)
                quality_score = quality_result.get('weighted_score', 0)
                momentum_score = momentum_result.get('weighted_score', 0)
                growth_score = growth_result.get('weighted_score', 0)

                # Use Phase 2 weighting for current score estimate
                base_factor_score = (
                    value_score * 0.15 +
                    quality_score * 0.20 +
                    momentum_score * 0.15 +
                    growth_score * 0.50
                )
                current_score = base_factor_score * 0.30 + 50 * 0.70  # Conservative estimate

            # Get historical scores from database
            query_1m = """
                SELECT final_score
                FROM kr_stock_grade
                WHERE symbol = $1
                  AND date >= CURRENT_DATE - INTERVAL '40 days'
                  AND date <= CURRENT_DATE - INTERVAL '20 days'
                  AND final_score IS NOT NULL
                ORDER BY date DESC
                LIMIT 1
            """

            query_2m = """
                SELECT final_score
                FROM kr_stock_grade
                WHERE symbol = $1
                  AND date >= CURRENT_DATE - INTERVAL '70 days'
                  AND date <= CURRENT_DATE - INTERVAL '50 days'
                  AND final_score IS NOT NULL
                ORDER BY date DESC
                LIMIT 1
            """

            score_1m_ago = None
            score_2m_ago = None

            # Fetch 1 month ago score
            rows_1m = await self.db_manager.execute_query(query_1m, self.symbol)
            if rows_1m and len(rows_1m) > 0:
                score_1m_ago = float(rows_1m[0]['final_score'])

            # Fetch 2 months ago score
            rows_2m = await self.db_manager.execute_query(query_2m, self.symbol)
            if rows_2m and len(rows_2m) > 0:
                score_2m_ago = float(rows_2m[0]['final_score'])

            # Calculate changes
            change_1m = None
            change_2m = None

            if score_1m_ago is not None and current_score is not None:
                change_1m = current_score - score_1m_ago

            if score_2m_ago is not None and score_1m_ago is not None:
                change_2m = score_1m_ago - score_2m_ago

            # Determine score and signal based on patterns
            if change_1m is None:
                # No historical data - return neutral
                score = 50
                signal_text = "No historical data - Score momentum neutral"
                logger.info(f"Score Momentum: {score}/100 (No historical data)")

            elif change_1m > 10:
                # Check if accelerating
                if change_2m is not None and change_2m > 5:
                    score = 100
                    signal_text = f"ACCELERATING UPWARD: Score rising fast for 2 months (+{change_1m:.1f} recent, +{change_2m:.1f} prior) - STRONG BUY (+48.97% historical avg)"
                else:
                    score = 85
                    signal_text = f"RISING FAST: Score jumped +{change_1m:.1f} points - Potential breakout (+48.97% historical avg)"
                logger.info(f"Score Momentum: {score}/100 - RISING FAST")
                logger.info(f"  Change: 1m=+{change_1m:.1f}, 2m={'+' + str(change_2m) if change_2m else 'N/A'}")

            elif change_1m > 5:
                score = 70
                signal_text = f"RISING STEADILY: Score up +{change_1m:.1f} points - Positive momentum (+9.17% historical avg)"
                logger.info(f"Score Momentum: {score}/100 - RISING")
                logger.info(f"  Change: 1m=+{change_1m:.1f}")

            elif change_1m >= -5:
                score = 50
                signal_text = f"STABLE: Score change {change_1m:+.1f} points - Neutral (+1.26% historical avg)"
                logger.info(f"Score Momentum: {score}/100 - STABLE")

            elif change_1m >= -10:
                score = 30
                signal_text = f"DECLINING: Score down {change_1m:.1f} points - Caution (-2.12% historical avg)"
                logger.info(f"Score Momentum: {score}/100 - DECLINING")
                logger.info(f"  Change: 1m={change_1m:.1f}")

            else:  # change_1m < -10
                score = 10
                signal_text = f"FALLING FAST: Score dropped {change_1m:.1f} points - WARNING (-2.35% historical avg)"
                logger.info(f"Score Momentum: {score}/100 - FALLING FAST")
                logger.info(f"  Change: 1m={change_1m:.1f}, 2m={change_2m if change_2m else 'N/A'}")

            return score, signal_text, change_1m, change_2m

        except Exception as e:
            logger.error(f"Score momentum calculation failed: {e}")
            return 50, "Score momentum calculation error", None, None

    async def calculate_sector_rotation_signal(self) -> Tuple[int, Optional[float], Optional[int], Optional[float]]:
        """
        Calculate sector rotation signal based on recent sector performance

        OPTIMIZED VERSION (Phase 3.4 - Materialized View Integration)
        - Uses mv_sector_daily_performance for pre-calculated sector aggregates
        - Memory reduction: 5GB -> 10MB (500x improvement)
        - Speed improvement: 5-10s -> 0.01s (1000x faster)
        - Data refreshed automatically by kr_main.py before each date analysis

        Analysis from Sep-Oct 2025 data:
        - Top sector (금융서비스): +24.10% return, 82.5% win rate
        - Bottom sector (화학섬유소재): -3.32% return, 28.9% win rate
        - Gap: 27.42% alpha potential from sector selection

        Scoring:
        - Top 3 sectors: 90-100 points (+20 bonus)
        - Top 20% sectors: 70-89 points (+10 bonus)
        - Middle sectors: 40-69 points (neutral)
        - Bottom 20% sectors: 10-39 points (-15 penalty)
        - Bottom 3 sectors: 0-9 points (-20 penalty)

        Returns:
            (sector_score, sector_momentum, sector_rank, sector_percentile)
        """
        try:
            # Step 1: Get current stock's theme from kr_stock_detail
            query_theme = """
                SELECT theme
                FROM kr_stock_detail
                WHERE symbol = $1
            """

            theme_result = await self.db_manager.execute_query(query_theme, self.symbol)
            if not theme_result or len(theme_result) == 0:
                logger.info("Sector Rotation: Symbol not found in kr_stock_detail, returning neutral")
                return 50, None, None, None

            current_sector = theme_result[0]['theme']
            if not current_sector:
                logger.info("Sector Rotation: theme is NULL, returning neutral")
                return 50, None, None, None

            # Step 2: Get sector performance from Materialized View
            # Uses pre-calculated sector aggregates (refreshed by kr_main.py)
            # BEFORE: Complex JOIN query with 2,748 stocks × 30 days (5GB memory)
            # AFTER: Simple MV lookup for single row (10MB memory)
            query_mv = """
                SELECT
                    avg_return_30d as sector_momentum,
                    sector_rank,
                    stock_count,
                    (SELECT COUNT(DISTINCT sector_code)
                     FROM mv_sector_daily_performance
                     WHERE date = $2) as total_sectors
                FROM mv_sector_daily_performance
                WHERE date = $2 AND sector_code = $1
            """

            mv_result = await self.db_manager.execute_query(query_mv, current_sector, self.analysis_date)

            if not mv_result or len(mv_result) == 0:
                logger.info(f"Sector Rotation: No MV data for sector '{current_sector}' on {self.analysis_date}, returning neutral")
                return 50, None, None, None

            row = mv_result[0]
            current_sector_rank = row['sector_rank']
            current_sector_momentum = float(row['sector_momentum']) if row['sector_momentum'] else 0.0
            total_sectors = row['total_sectors']

            if not current_sector_rank or not total_sectors:
                logger.info("Sector Rotation: Invalid MV data, returning neutral")
                return 50, None, None, None

            # Step 3: Calculate percentile and score (기존 로직 유지)
            percentile = (1 - (current_sector_rank - 1) / total_sectors) * 100

            # Score based on rank
            score = 50  # Default neutral

            if current_sector_rank <= 3:
                # Top 3 sectors
                score = 90 + (3 - current_sector_rank) * 3  # 90, 93, 96
                logger.info(f"Sector Rotation: {score}/100 - TOP 3 SECTOR")
                logger.info(f"  Sector: {current_sector} (Rank {current_sector_rank}/{total_sectors})")
                logger.info(f"  Momentum: {current_sector_momentum:+.2f}%")
                logger.info(f"  Percentile: {percentile:.1f}% (Top sector bonus: +20)")

            elif current_sector_rank <= max(3, int(total_sectors * 0.2)):
                # Top 20%
                top_20_pct_rank = int(total_sectors * 0.2)
                score = 70 + int((top_20_pct_rank - current_sector_rank) / top_20_pct_rank * 19)
                logger.info(f"Sector Rotation: {score}/100 - TOP 20%")
                logger.info(f"  Sector: {current_sector} (Rank {current_sector_rank}/{total_sectors})")
                logger.info(f"  Momentum: {current_sector_momentum:+.2f}%")
                logger.info(f"  Percentile: {percentile:.1f}% (Strong sector bonus: +10)")

            elif current_sector_rank >= total_sectors - 2:
                # Bottom 3 sectors
                score = 10 - (total_sectors - current_sector_rank) * 3  # 10, 7, 4
                score = max(0, score)
                logger.info(f"Sector Rotation: {score}/100 - BOTTOM 3 SECTOR (WARNING)")
                logger.info(f"  Sector: {current_sector} (Rank {current_sector_rank}/{total_sectors})")
                logger.info(f"  Momentum: {current_sector_momentum:+.2f}%")
                logger.info(f"  Percentile: {percentile:.1f}% (Weak sector penalty: -20)")

            elif current_sector_rank >= max(total_sectors - 2, int(total_sectors * 0.8)):
                # Bottom 20%
                bottom_20_pct_rank = int(total_sectors * 0.8)
                score = 10 + int((current_sector_rank - bottom_20_pct_rank) / (total_sectors - bottom_20_pct_rank) * 29)
                logger.info(f"Sector Rotation: {score}/100 - BOTTOM 20%")
                logger.info(f"  Sector: {current_sector} (Rank {current_sector_rank}/{total_sectors})")
                logger.info(f"  Momentum: {current_sector_momentum:+.2f}%")
                logger.info(f"  Percentile: {percentile:.1f}% (Weak sector penalty: -15)")

            else:
                # Middle sectors (20%-80%)
                # Linear interpolation from 40 to 69
                mid_start = max(3, int(total_sectors * 0.2))
                mid_end = max(total_sectors - 2, int(total_sectors * 0.8))
                if mid_end > mid_start:
                    score = 40 + int((mid_end - current_sector_rank) / (mid_end - mid_start) * 29)
                else:
                    score = 50
                logger.info(f"Sector Rotation: {score}/100 - MIDDLE")
                logger.info(f"  Sector: {current_sector} (Rank {current_sector_rank}/{total_sectors})")
                logger.info(f"  Momentum: {current_sector_momentum:+.2f}%")
                logger.info(f"  Percentile: {percentile:.1f}%")

            return score, current_sector_momentum, current_sector_rank, percentile

        except Exception as e:
            logger.error(f"Sector rotation calculation failed: {e}")
            return 50, None, None, None

    async def calculate_all_metrics(self) -> Dict:
        """
        Calculate all 15 metrics (9 fundamental + 4 technical + 2 new signals)

        Returns:
            Dict with all metrics ready for kr_stock_grade table insertion
        """
        logger.info(f"Calculating all metrics for {self.symbol}...")

        # Calculate fundamental & risk metrics (9 + cvar_95)
        confidence = await self.calculate_confidence_score()
        volatility = await self.calculate_volatility_annual()
        expected_ranges = await self.calculate_expected_range(volatility)
        var_95 = await self.calculate_var_95()
        cvar_95 = await self.calculate_cvar_95()
        inst_net, foreign_net = await self.calculate_investor_flow()
        factor_momentum = await self.calculate_factor_momentum()
        industry_rank, industry_percentile = await self.calculate_industry_ranking()
        max_drawdown = await self.calculate_max_drawdown_1y()
        beta = await self.calculate_beta()

        # Calculate technical indicators (4)
        support_resistance = await self.calculate_support_resistance()
        supertrend = await self.calculate_supertrend()
        relative_strength = await self.calculate_relative_strength()

        # Calculate Phase 1 signal scores (2)
        smart_money_score, smart_money_text, smart_money_conf = await self.calculate_smart_money_signal_score()
        volatility_context_score, volatility_context_text = await self.calculate_volatility_context_score(volatility, smart_money_score)

        # Calculate Phase 2 signal scores (2)
        # Safely extract scores (handle None weighted_result from failed factor calculations)
        value_result = self.factor_scores.get('value', {}).get('weighted_result') or {}
        quality_result = self.factor_scores.get('quality', {}).get('weighted_result') or {}
        momentum_result = self.factor_scores.get('momentum', {}).get('weighted_result') or {}
        growth_result = self.factor_scores.get('growth', {}).get('weighted_result') or {}

        value_score = value_result.get('weighted_score', 0)
        quality_score = quality_result.get('weighted_score', 0)
        momentum_score = momentum_result.get('weighted_score', 0)
        growth_score = growth_result.get('weighted_score', 0)

        factor_combo_bonus = await self.calculate_factor_combination_bonus(
            value_score, quality_score, momentum_score, growth_score, smart_money_score
        )
        smart_money_momentum = await self.calculate_smart_money_momentum_score()

        # Calculate Phase 3.1 signal (1)
        score_momentum_signal, score_momentum_text, change_1m, change_2m = await self.calculate_score_momentum_signal()

        # Calculate Phase 3.2 signal (1)
        sector_rotation_score, sector_momentum, sector_rank, sector_percentile = await self.calculate_sector_rotation_signal()

        # Compile results
        metrics = {
            # Fundamental & Risk Metrics (9 + cvar_95)
            'confidence_score': confidence,
            'volatility_annual': volatility,
            'expected_range_3m_min': expected_ranges.get('range_3m_min') if expected_ranges else None,
            'expected_range_3m_max': expected_ranges.get('range_3m_max') if expected_ranges else None,
            'expected_range_1y_min': expected_ranges.get('range_1y_min') if expected_ranges else None,
            'expected_range_1y_max': expected_ranges.get('range_1y_max') if expected_ranges else None,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'inst_net_30d': inst_net,
            'foreign_net_30d': foreign_net,
            'value_momentum': factor_momentum.get('value_momentum'),
            'quality_momentum': factor_momentum.get('quality_momentum'),
            'momentum_momentum': factor_momentum.get('momentum_momentum'),
            'growth_momentum': factor_momentum.get('growth_momentum'),
            'industry_rank': industry_rank,
            'industry_percentile': industry_percentile,
            'max_drawdown_1y': max_drawdown,
            'beta': beta,

            # Technical Indicators (4)
            'support_1': support_resistance.get('support_1'),
            'support_2': support_resistance.get('support_2'),
            'resistance_1': support_resistance.get('resistance_1'),
            'resistance_2': support_resistance.get('resistance_2'),
            'supertrend_value': supertrend.get('supertrend_value'),
            'trend': supertrend.get('trend'),
            'signal': supertrend.get('signal'),
            'rs_value': relative_strength.get('rs_value'),
            'rs_rank': relative_strength.get('rs_rank'),

            # Phase 1 Signal Scores (2)
            'smart_money_signal_score': smart_money_score,
            'smart_money_signal_text': smart_money_text,
            'smart_money_confidence': smart_money_conf,
            'volatility_context_score': volatility_context_score,
            'volatility_context_text': volatility_context_text,

            # Phase 2 Signal Scores (2)
            'factor_combination_bonus': factor_combo_bonus,
            'smart_money_momentum_score': smart_money_momentum,

            # Phase 3.1 Signal (1)
            'score_momentum_signal': score_momentum_signal,
            'score_momentum_text': score_momentum_text,
            'score_change_1m': change_1m,
            'score_change_2m': change_2m,

            # Phase 3.2 Signal (1)
            'sector_rotation_score': sector_rotation_score,
            'sector_momentum': sector_momentum,
            'sector_rank': sector_rank,
            'sector_percentile': sector_percentile
        }

        logger.info("All 19 metrics calculated successfully")
        logger.info(f"  - Fundamental/Risk: 9 metrics")
        logger.info(f"  - Technical: 4 indicators (S/R, SuperTrend, RS)")
        logger.info(f"  - Phase 1 Signals: 2 scores (Smart Money, Volatility Context)")
        logger.info(f"  - Phase 2 Signals: 2 scores (Factor Combo, SM Momentum)")
        logger.info(f"  - Phase 3.1 Signal: 1 score (Score Momentum)")
        logger.info(f"  - Phase 3.2 Signal: 1 score (Sector Rotation)")
        return metrics

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


if __name__ == '__main__':
    """Test additional metrics calculation"""
    from dotenv import load_dotenv
    load_dotenv()

    symbol = '005930'

    # Mock factor scores for testing
    mock_factor_scores = {
        'value': {'weighted_result': {'weighted_score': 45.0}},
        'quality': {'weighted_result': {'weighted_score': 75.0}},
        'momentum': {'weighted_result': {'weighted_score': 60.0}},
        'growth': {'weighted_result': {'weighted_score': 55.0}}
    }

    calc = AdditionalMetricsCalculator(symbol, mock_factor_scores)
    metrics = calc.calculate_all_metrics()

    print("\n" + "="*80)
    print(f"Additional Metrics for {symbol}")
    print("="*80)

    for key, value in metrics.items():
        if value is not None:
            if isinstance(value, float):
                print(f"{key:<25}: {value:.2f}")
            else:
                print(f"{key:<25}: {value:,}" if isinstance(value, int) else f"{key:<25}: {value}")
        else:
            print(f"{key:<25}: N/A")

    calc.close()
