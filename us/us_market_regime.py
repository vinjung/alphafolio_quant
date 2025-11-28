"""
US Market Regime Detector v2.0

시장 레짐 감지 및 동적 팩터 가중치 결정

Regimes:
- AI_BULL: AI 주도 성장주 랠리 (Growth/Momentum 강조)
- TIGHTENING: 금리 인상/긴축 환경 (Quality/Value 강조)
- RECOVERY: 경기 회복기 (Value/Momentum 균형)
- CRISIS: 위기/급락장 (Quality 방어)
- NEUTRAL: 혼조장

Data Sources:
- VIX Proxy: us_option_daily_summary (SPY avg_implied_volatility)
- SPY/QQQ: us_daily_etf
- Macro: us_fed_funds_rate, us_cpi, us_unemployment_rate

File: us/us_market_regime.py
"""

import logging
from typing import Dict, Optional
from datetime import date, timedelta
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# Regime Definitions
# ========================================================================

REGIME_WEIGHTS = {
    'AI_BULL': {
        'description': 'AI 주도 성장주 랠리',
        'value': 0.10,
        'quality': 0.25,
        'momentum': 0.35,
        'growth': 0.30
    },
    'TIGHTENING': {
        'description': '금리 인상/긴축 환경',
        'value': 0.30,
        'quality': 0.40,
        'momentum': 0.20,
        'growth': 0.10
    },
    'RECOVERY': {
        'description': '경기 회복기',
        'value': 0.30,
        'quality': 0.15,
        'momentum': 0.30,
        'growth': 0.25
    },
    'CRISIS': {
        'description': '위기/급락장',
        'value': 0.25,
        'quality': 0.50,
        'momentum': 0.10,
        'growth': 0.15
    },
    'NEUTRAL': {
        'description': '혼조장/중립',
        'value': 0.25,
        'quality': 0.25,
        'momentum': 0.25,
        'growth': 0.25
    }
}


class USMarketRegimeDetector:
    """시장 레짐 감지 및 동적 가중치 결정"""

    def __init__(self, db_manager, analysis_date: Optional[date] = None):
        """
        Args:
            db_manager: AsyncDatabaseManager instance
            analysis_date: 분석 날짜 (None이면 최신)
        """
        self.db = db_manager
        self.analysis_date = analysis_date
        self.indicators = {}

    async def detect_regime(self) -> Dict:
        """
        시장 레짐 감지 및 결과 반환

        Returns:
            {
                'regime': str,
                'weights': {'value': float, 'quality': float, ...},
                'indicators': {...},
                'description': str
            }
        """
        try:
            # 1. VIX Proxy 조회 (SPY 옵션 IV)
            vix_data = await self._get_vix_proxy()

            # 2. 시장 지표 조회 (SPY, QQQ)
            market_data = await self._get_market_indicators()

            # 3. 매크로 지표 조회
            macro_data = await self._get_macro_indicators()

            # 4. 지표 통합
            self.indicators = {
                **vix_data,
                **market_data,
                **macro_data
            }

            # 5. 레짐 분류
            regime = self._classify_regime()

            # 6. 가중치 조회
            weights = self._get_weights(regime)

            logger.info(f"Market Regime: {regime} ({REGIME_WEIGHTS[regime]['description']})")
            logger.info(f"Weights: V={weights['value']:.0%}, Q={weights['quality']:.0%}, "
                       f"M={weights['momentum']:.0%}, G={weights['growth']:.0%}")

            return {
                'regime': regime,
                'weights': weights,
                'indicators': self.indicators,
                'description': REGIME_WEIGHTS[regime]['description']
            }

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            # Fallback to NEUTRAL
            return {
                'regime': 'NEUTRAL',
                'weights': REGIME_WEIGHTS['NEUTRAL'].copy(),
                'indicators': {},
                'description': 'Fallback to NEUTRAL due to error'
            }

    async def _get_vix_proxy(self) -> Dict:
        """SPY 옵션 IV로 VIX 대용 지표 조회"""

        query = """
        SELECT
            date,
            avg_implied_volatility * 100 as vix_proxy,
            avg_call_iv * 100 as call_iv,
            avg_put_iv * 100 as put_iv,
            CASE WHEN total_call_volume > 0
                 THEN total_put_volume::float / total_call_volume
                 ELSE 1.0 END as put_call_ratio
        FROM us_option_daily_summary
        WHERE symbol = 'SPY'
            AND ($1::DATE IS NULL OR date <= $1::DATE)
        ORDER BY date DESC
        LIMIT 1
        """

        try:
            result = await self.db.execute_query(query, self.analysis_date)

            if result and result[0]['vix_proxy']:
                row = result[0]
                vix = float(row['vix_proxy']) if row['vix_proxy'] else 20.0
                call_iv = float(row['call_iv']) if row['call_iv'] else vix
                put_iv = float(row['put_iv']) if row['put_iv'] else vix
                pcr = float(row['put_call_ratio']) if row['put_call_ratio'] else 1.0

                return {
                    'vix_proxy': vix,
                    'put_call_ratio': pcr,
                    'iv_skew': put_iv - call_iv,
                    'vix_date': row['date']
                }
        except Exception as e:
            logger.warning(f"VIX proxy query failed: {e}")

        # Fallback
        return {
            'vix_proxy': 20.0,
            'put_call_ratio': 1.0,
            'iv_skew': 0.0,
            'vix_date': None
        }

    async def _get_market_indicators(self) -> Dict:
        """SPY, QQQ 시장 지표 조회"""

        # SPY 수익률 및 MA200
        spy_query = """
        WITH spy_prices AS (
            SELECT
                date,
                close,
                LAG(close, 21) OVER (ORDER BY date) as close_1m,
                LAG(close, 63) OVER (ORDER BY date) as close_3m,
                AVG(close) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) as ma200
            FROM us_daily_etf
            WHERE symbol = 'SPY'
            ORDER BY date DESC
        )
        SELECT
            date,
            close,
            close_1m,
            close_3m,
            ma200,
            CASE WHEN close_1m > 0 THEN (close - close_1m) / close_1m * 100 ELSE 0 END as return_1m,
            CASE WHEN close_3m > 0 THEN (close - close_3m) / close_3m * 100 ELSE 0 END as return_3m,
            CASE WHEN ma200 > 0 THEN (close - ma200) / ma200 * 100 ELSE 0 END as ma200_distance
        FROM spy_prices
        WHERE ($1::DATE IS NULL OR date <= $1::DATE)
        ORDER BY date DESC
        LIMIT 1
        """

        # QQQ 수익률 (NASDAQ 대용)
        qqq_query = """
        WITH qqq_prices AS (
            SELECT
                date,
                close,
                LAG(close, 63) OVER (ORDER BY date) as close_3m
            FROM us_daily_etf
            WHERE symbol = 'QQQ'
            ORDER BY date DESC
        )
        SELECT
            CASE WHEN close_3m > 0 THEN (close - close_3m) / close_3m * 100 ELSE 0 END as qqq_return_3m
        FROM qqq_prices
        WHERE ($1::DATE IS NULL OR date <= $1::DATE)
        ORDER BY date DESC
        LIMIT 1
        """

        result = {
            'spy_return_1m': 0.0,
            'spy_return_3m': 0.0,
            'spy_ma200_distance': 0.0,
            'nasdaq_vs_spy_3m': 0.0
        }

        try:
            spy_result = await self.db.execute_query(spy_query, self.analysis_date)
            if spy_result and spy_result[0]:
                row = spy_result[0]
                result['spy_return_1m'] = float(row['return_1m'] or 0)
                result['spy_return_3m'] = float(row['return_3m'] or 0)
                result['spy_ma200_distance'] = float(row['ma200_distance'] or 0)

            qqq_result = await self.db.execute_query(qqq_query, self.analysis_date)
            if qqq_result and qqq_result[0]:
                qqq_return = float(qqq_result[0]['qqq_return_3m'] or 0)
                result['nasdaq_vs_spy_3m'] = qqq_return - result['spy_return_3m']

        except Exception as e:
            logger.warning(f"Market indicators query failed: {e}")

        return result

    async def _get_macro_indicators(self) -> Dict:
        """매크로 지표 조회"""

        result = {
            'fed_rate': 5.0,
            'fed_rate_change_6m': 0.0,
            'cpi_yoy': 3.0,
            'unemployment_rate': 4.0
        }

        try:
            # Fed Rate
            fed_query = """
            SELECT
                (SELECT value FROM us_fed_funds_rate
                 WHERE ($1::DATE IS NULL OR date <= $1::DATE)
                 ORDER BY date DESC LIMIT 1) as current_rate,
                (SELECT value FROM us_fed_funds_rate
                 WHERE date <= COALESCE($1::DATE, CURRENT_DATE) - INTERVAL '6 months'
                 ORDER BY date DESC LIMIT 1) as rate_6m_ago
            """
            fed_result = await self.db.execute_query(fed_query, self.analysis_date)
            if fed_result and fed_result[0]:
                current = float(fed_result[0]['current_rate'] or 5.0)
                ago = float(fed_result[0]['rate_6m_ago'] or current)
                result['fed_rate'] = current
                result['fed_rate_change_6m'] = current - ago

            # CPI YoY
            cpi_query = """
            SELECT
                (SELECT value FROM us_cpi
                 WHERE ($1::DATE IS NULL OR date <= $1::DATE)
                 ORDER BY date DESC LIMIT 1) as current_cpi,
                (SELECT value FROM us_cpi
                 WHERE date <= COALESCE($1::DATE, CURRENT_DATE) - INTERVAL '1 year'
                 ORDER BY date DESC LIMIT 1) as cpi_1y_ago
            """
            cpi_result = await self.db.execute_query(cpi_query, self.analysis_date)
            if cpi_result and cpi_result[0]:
                current = float(cpi_result[0]['current_cpi'] or 300)
                ago = float(cpi_result[0]['cpi_1y_ago'] or 290)
                if ago > 0:
                    result['cpi_yoy'] = (current - ago) / ago * 100

            # Unemployment
            unemp_query = """
            SELECT value FROM us_unemployment_rate
            WHERE ($1::DATE IS NULL OR date <= $1::DATE)
            ORDER BY date DESC LIMIT 1
            """
            unemp_result = await self.db.execute_query(unemp_query, self.analysis_date)
            if unemp_result and unemp_result[0]:
                result['unemployment_rate'] = float(unemp_result[0]['value'] or 4.0)

        except Exception as e:
            logger.warning(f"Macro indicators query failed: {e}")

        return result

    def _classify_regime(self) -> str:
        """레짐 분류"""

        vix = self.indicators.get('vix_proxy', 20)
        spy_return_3m = self.indicators.get('spy_return_3m', 0)
        nasdaq_vs_spy = self.indicators.get('nasdaq_vs_spy_3m', 0)
        fed_change = self.indicators.get('fed_rate_change_6m', 0)
        cpi_yoy = self.indicators.get('cpi_yoy', 3)
        ma200_dist = self.indicators.get('spy_ma200_distance', 0)

        logger.info(f"Regime Indicators: VIX={vix:.1f}, SPY_3M={spy_return_3m:.1f}%, "
                   f"NASDAQ_vs_SPY={nasdaq_vs_spy:.1f}%, Fed_chg={fed_change:.2f}, CPI={cpi_yoy:.1f}%")

        # 1. CRISIS (최우선 - VIX 급등)
        if vix > 30:
            logger.info("Regime: CRISIS (VIX > 30)")
            return 'CRISIS'

        # 2. AI_BULL (성장주 랠리)
        if (vix < 22 and
            nasdaq_vs_spy > 3 and  # NASDAQ이 SPY 대비 3% 이상 아웃퍼폼
            spy_return_3m > 3 and
            ma200_dist > 0):  # 200일선 위
            logger.info("Regime: AI_BULL (Low VIX + NASDAQ outperform)")
            return 'AI_BULL'

        # 3. TIGHTENING (긴축 환경)
        if (fed_change > 0.25 and  # 금리 25bp 이상 인상
            cpi_yoy > 3.5):
            logger.info("Regime: TIGHTENING (Fed hiking + High CPI)")
            return 'TIGHTENING'

        # 4. RECOVERY (회복기)
        if (spy_return_3m > 8 and
            vix < 25 and
            ma200_dist > 3):  # 200일선 3% 이상 위
            logger.info("Regime: RECOVERY (Strong rally + Low VIX)")
            return 'RECOVERY'

        # 5. NEUTRAL (기본)
        logger.info("Regime: NEUTRAL (No strong signals)")
        return 'NEUTRAL'

    def _get_weights(self, regime: str) -> Dict[str, float]:
        """레짐별 팩터 가중치 반환"""

        weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS['NEUTRAL'])
        return {
            'value': weights['value'],
            'quality': weights['quality'],
            'momentum': weights['momentum'],
            'growth': weights['growth']
        }

    async def save_to_db(self, regime_data: Dict) -> bool:
        """레짐 결과를 DB에 저장"""

        try:
            query = """
            INSERT INTO us_market_regime (
                date, regime,
                vix_proxy, put_call_ratio, iv_skew,
                spy_return_1m, spy_return_3m, spy_ma200_distance, nasdaq_vs_spy_3m,
                fed_rate, fed_rate_change_6m, cpi_yoy, unemployment_rate,
                weight_value, weight_quality, weight_momentum, weight_growth
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
            )
            ON CONFLICT (date) DO UPDATE SET
                regime = EXCLUDED.regime,
                vix_proxy = EXCLUDED.vix_proxy,
                put_call_ratio = EXCLUDED.put_call_ratio,
                iv_skew = EXCLUDED.iv_skew,
                spy_return_1m = EXCLUDED.spy_return_1m,
                spy_return_3m = EXCLUDED.spy_return_3m,
                spy_ma200_distance = EXCLUDED.spy_ma200_distance,
                nasdaq_vs_spy_3m = EXCLUDED.nasdaq_vs_spy_3m,
                fed_rate = EXCLUDED.fed_rate,
                fed_rate_change_6m = EXCLUDED.fed_rate_change_6m,
                cpi_yoy = EXCLUDED.cpi_yoy,
                unemployment_rate = EXCLUDED.unemployment_rate,
                weight_value = EXCLUDED.weight_value,
                weight_quality = EXCLUDED.weight_quality,
                weight_momentum = EXCLUDED.weight_momentum,
                weight_growth = EXCLUDED.weight_growth
            """

            ind = regime_data['indicators']
            weights = regime_data['weights']
            target_date = self.analysis_date or date.today()

            await self.db.execute(
                query,
                target_date,
                regime_data['regime'],
                ind.get('vix_proxy'),
                ind.get('put_call_ratio'),
                ind.get('iv_skew'),
                ind.get('spy_return_1m'),
                ind.get('spy_return_3m'),
                ind.get('spy_ma200_distance'),
                ind.get('nasdaq_vs_spy_3m'),
                ind.get('fed_rate'),
                ind.get('fed_rate_change_6m'),
                ind.get('cpi_yoy'),
                ind.get('unemployment_rate'),
                weights['value'],
                weights['quality'],
                weights['momentum'],
                weights['growth']
            )

            logger.info(f"Regime saved to DB: {target_date} -> {regime_data['regime']}")
            return True

        except Exception as e:
            logger.error(f"Failed to save regime to DB: {e}")
            return False


async def get_current_regime(db_manager, analysis_date: Optional[date] = None) -> Dict:
    """현재 시장 레짐 조회 (편의 함수)"""

    detector = USMarketRegimeDetector(db_manager, analysis_date)
    return await detector.detect_regime()


# ========================================================================
# Test
# ========================================================================

if __name__ == '__main__':
    import asyncio
    from us_db_async import AsyncDatabaseManager

    async def test():
        db = AsyncDatabaseManager()
        await db.initialize()

        detector = USMarketRegimeDetector(db)
        result = await detector.detect_regime()

        print("\n" + "="*60)
        print("Market Regime Detection Result")
        print("="*60)
        print(f"Regime: {result['regime']}")
        print(f"Description: {result['description']}")
        print(f"\nWeights:")
        for k, v in result['weights'].items():
            print(f"  {k}: {v:.0%}")
        print(f"\nIndicators:")
        for k, v in result['indicators'].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

        await db.close()

    asyncio.run(test())
