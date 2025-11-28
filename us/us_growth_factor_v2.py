"""
US Growth Factor v2.0 - Future Growth

핵심: "과거 성장"보다 "미래 성장 가능성" 중시

Strategies:
- FG1: Revenue Growth (20%) - 매출 성장률 (YoY)
- FG2: EPS Growth (20%) - EPS 성장률 (YoY)
- FG3: Forward Growth (20%) - 애널리스트 예상 성장률
- FG4: Growth Consistency (15%) - 성장 일관성 (분기별 변동)
- FG5: Revenue Acceleration (15%) - 성장 가속화 여부
- FG6: Profit Growth (10%) - 이익 성장 (수익성 개선)

File: us/us_growth_factor_v2.py
"""

import logging
from typing import Dict, Optional, List
from datetime import date
from decimal import Decimal
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# Strategy Weights
# ========================================================================

STRATEGY_WEIGHTS = {
    'FG1': 0.20,  # Revenue Growth
    'FG2': 0.20,  # EPS Growth
    'FG3': 0.20,  # Forward Growth
    'FG4': 0.15,  # Growth Consistency
    'FG5': 0.15,  # Revenue Acceleration
    'FG6': 0.10   # Profit Growth
}

# 섹터별 성장률 기준 (고성장 섹터는 기준이 높음)
SECTOR_GROWTH_THRESHOLDS = {
    'TECHNOLOGY': {'high': 0.30, 'good': 0.15, 'moderate': 0.08},
    'HEALTHCARE': {'high': 0.25, 'good': 0.12, 'moderate': 0.06},
    'COMMUNICATION SERVICES': {'high': 0.25, 'good': 0.12, 'moderate': 0.06},
    'CONSUMER CYCLICAL': {'high': 0.20, 'good': 0.10, 'moderate': 0.05},
    'FINANCIAL SERVICES': {'high': 0.15, 'good': 0.08, 'moderate': 0.04},
    'DEFAULT': {'high': 0.20, 'good': 0.10, 'moderate': 0.05}
}


class USGrowthFactorV2:
    """Growth Factor v2.0 - Future Growth"""

    def __init__(self, symbol: str, db_manager, sector_benchmarks: Dict,
                 analysis_date: Optional[date] = None):
        """
        Args:
            symbol: 종목 코드
            db_manager: AsyncDatabaseManager
            sector_benchmarks: 섹터 기준값 (from USSectorBenchmarks)
            analysis_date: 분석 날짜
        """
        self.symbol = symbol
        self.db = db_manager
        self.benchmarks = sector_benchmarks
        self.analysis_date = analysis_date
        self.stock_data = {}
        self.sector = None
        self.quarterly_data = []

    async def calculate(self) -> Dict:
        """
        Growth Factor 점수 계산

        Returns:
            {
                'growth_score': float (0-100),
                'strategies': {
                    'FG1': {'score': float, 'raw': float, ...},
                    ...
                }
            }
        """
        try:
            # 1. 데이터 로드
            await self._load_stock_data()
            await self._load_quarterly_data()

            if not self.stock_data:
                logger.warning(f"{self.symbol}: No data available")
                return {'growth_score': 50.0, 'strategies': {}}

            # 2. 각 전략별 점수 계산
            strategies = {}
            strategies['FG1'] = self._calc_fg1_revenue_growth()
            strategies['FG2'] = self._calc_fg2_eps_growth()
            strategies['FG3'] = self._calc_fg3_forward_growth()
            strategies['FG4'] = self._calc_fg4_growth_consistency()
            strategies['FG5'] = self._calc_fg5_revenue_acceleration()
            strategies['FG6'] = self._calc_fg6_profit_growth()

            # 3. 가중 평균 계산
            total_weight = 0
            weighted_sum = 0

            for strategy_id, result in strategies.items():
                if result['score'] is not None:
                    weight = STRATEGY_WEIGHTS[strategy_id]
                    weighted_sum += result['score'] * weight
                    total_weight += weight

            growth_score = weighted_sum / total_weight if total_weight > 0 else 50.0

            logger.info(f"{self.symbol}: Growth Score = {growth_score:.1f}")

            return {
                'growth_score': round(growth_score, 1),
                'strategies': strategies
            }

        except Exception as e:
            logger.error(f"{self.symbol}: Growth calculation failed - {e}")
            return {'growth_score': 50.0, 'strategies': {}}

    async def _load_stock_data(self):
        """종목 기본 데이터 로드"""

        query = """
        SELECT
            symbol,
            sector,
            quarterlyrevenuegrowthyoy as revenue_growth,
            quarterlyearningsgrowthyoy as earnings_growth,
            peg,
            forwardpe,
            per,
            revenuettm as revenue,
            grossprofitttm as gross_profit,
            operatingmarginttm as operating_margin,
            profitmargin as net_margin,
            dilutedepsttm as eps,
            ebitda
        FROM us_stock_basic
        WHERE symbol = $1
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result and result[0]:
                row = result[0]
                self.sector = row['sector']
                self.stock_data = {
                    'revenue_growth': self._to_float(row['revenue_growth']),
                    'earnings_growth': self._to_float(row['earnings_growth']),
                    'peg': self._to_float(row['peg']),
                    'forward_pe': self._to_float(row['forwardpe']),
                    'per': self._to_float(row['per']),
                    'revenue': self._to_float(row['revenue']),
                    'gross_profit': self._to_float(row['gross_profit']),
                    'operating_margin': self._to_float(row['operating_margin']),
                    'net_margin': self._to_float(row['net_margin']),
                    'eps': self._to_float(row['eps']),
                    'ebitda': self._to_float(row['ebitda'])
                }

        except Exception as e:
            logger.error(f"{self.symbol}: Failed to load stock data - {e}")

    async def _load_quarterly_data(self):
        """분기별 재무 데이터 로드"""

        query = """
        SELECT
            fiscal_date_ending,
            total_revenue,
            gross_profit,
            operating_income,
            net_income,
            basic_eps
        FROM us_income_statement
        WHERE symbol = $1
        ORDER BY fiscal_date_ending DESC
        LIMIT 12
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result:
                self.quarterly_data = [
                    {
                        'date': r['fiscal_date_ending'],
                        'revenue': self._to_float(r['total_revenue']),
                        'gross_profit': self._to_float(r['gross_profit']),
                        'operating_income': self._to_float(r['operating_income']),
                        'net_income': self._to_float(r['net_income']),
                        'eps': self._to_float(r['basic_eps'])
                    }
                    for r in result
                ]

        except Exception as e:
            logger.warning(f"{self.symbol}: Failed to load quarterly data - {e}")

    def _calc_fg1_revenue_growth(self) -> Dict:
        """FG1: Revenue Growth (YoY)"""

        revenue_growth = self.stock_data.get('revenue_growth')

        # 분기 데이터에서 직접 계산 시도
        if revenue_growth is None and len(self.quarterly_data) >= 5:
            recent_rev = self.quarterly_data[0]['revenue']
            yoy_rev = self.quarterly_data[4]['revenue'] if len(self.quarterly_data) > 4 else None

            if recent_rev and yoy_rev and yoy_rev > 0:
                revenue_growth = (recent_rev - yoy_rev) / yoy_rev

        if revenue_growth is None:
            return {'score': None, 'raw': None, 'reason': 'No revenue growth data'}

        # 섹터별 기준 적용
        thresholds = SECTOR_GROWTH_THRESHOLDS.get(self.sector, SECTOR_GROWTH_THRESHOLDS['DEFAULT'])

        # 점수 계산
        if revenue_growth >= thresholds['high']:
            score = 85 + min(15, (revenue_growth - thresholds['high']) / 0.20 * 15)
        elif revenue_growth >= thresholds['good']:
            diff = thresholds['high'] - thresholds['good']
            score = 65 + (revenue_growth - thresholds['good']) / diff * 20
        elif revenue_growth >= thresholds['moderate']:
            diff = thresholds['good'] - thresholds['moderate']
            score = 50 + (revenue_growth - thresholds['moderate']) / diff * 15
        elif revenue_growth >= 0:
            score = 35 + revenue_growth / thresholds['moderate'] * 15
        elif revenue_growth >= -0.10:
            score = 25 + (revenue_growth + 0.10) / 0.10 * 10
        else:
            score = max(10, 25 + (revenue_growth + 0.10) * 100)

        return {
            'score': min(100, max(0, score)),
            'raw': revenue_growth,
            'thresholds': thresholds
        }

    def _calc_fg2_eps_growth(self) -> Dict:
        """FG2: EPS Growth (YoY)"""

        earnings_growth = self.stock_data.get('earnings_growth')

        # 분기 데이터에서 직접 계산 시도
        if earnings_growth is None and len(self.quarterly_data) >= 5:
            recent_eps = self.quarterly_data[0]['eps']
            yoy_eps = self.quarterly_data[4]['eps'] if len(self.quarterly_data) > 4 else None

            if recent_eps and yoy_eps and yoy_eps > 0:
                earnings_growth = (recent_eps - yoy_eps) / abs(yoy_eps)

        if earnings_growth is None:
            return {'score': None, 'raw': None, 'reason': 'No earnings growth data'}

        # 점수 계산 (EPS 성장은 매출 성장보다 기준이 유연함)
        if earnings_growth >= 0.50:
            score = 90 + min(10, (earnings_growth - 0.50) / 0.50 * 10)
        elif earnings_growth >= 0.25:
            score = 75 + (earnings_growth - 0.25) / 0.25 * 15
        elif earnings_growth >= 0.10:
            score = 60 + (earnings_growth - 0.10) / 0.15 * 15
        elif earnings_growth >= 0:
            score = 45 + earnings_growth / 0.10 * 15
        elif earnings_growth >= -0.15:
            score = 30 + (earnings_growth + 0.15) / 0.15 * 15
        elif earnings_growth >= -0.30:
            score = 20 + (earnings_growth + 0.30) / 0.15 * 10
        else:
            score = max(10, 20 + (earnings_growth + 0.30) * 30)

        return {
            'score': min(100, max(0, score)),
            'raw': earnings_growth
        }

    def _calc_fg3_forward_growth(self) -> Dict:
        """FG3: Forward Growth (PEG에서 추정)"""

        peg = self.stock_data.get('peg')
        forward_pe = self.stock_data.get('forward_pe')
        trailing_pe = self.stock_data.get('per')

        forward_growth = None

        # PEG에서 역산
        if peg and forward_pe and peg > 0 and forward_pe > 0:
            # PEG = PE / (Growth * 100)
            # Growth = PE / (PEG * 100)
            forward_growth = forward_pe / (peg * 100)

        # Forward PE vs Trailing PE에서 추정
        if forward_growth is None and forward_pe and trailing_pe and forward_pe > 0:
            if trailing_pe > 0:
                # Forward PE가 낮으면 성장 예상
                pe_ratio = trailing_pe / forward_pe
                if pe_ratio > 1:
                    forward_growth = pe_ratio - 1  # 대략적 추정

        if forward_growth is None:
            # 현재 성장률을 미래 성장률로 대체
            revenue_growth = self.stock_data.get('revenue_growth')
            earnings_growth = self.stock_data.get('earnings_growth')

            if revenue_growth is not None and earnings_growth is not None:
                forward_growth = (revenue_growth + earnings_growth) / 2 * 0.8  # 보수적 할인
            elif revenue_growth is not None:
                forward_growth = revenue_growth * 0.8
            elif earnings_growth is not None:
                forward_growth = earnings_growth * 0.8

        if forward_growth is None:
            return {'score': None, 'raw': None, 'reason': 'Cannot estimate forward growth'}

        # 점수 계산
        if forward_growth >= 0.40:
            score = 90 + min(10, (forward_growth - 0.40) / 0.20 * 10)
        elif forward_growth >= 0.25:
            score = 75 + (forward_growth - 0.25) / 0.15 * 15
        elif forward_growth >= 0.15:
            score = 60 + (forward_growth - 0.15) / 0.10 * 15
        elif forward_growth >= 0.08:
            score = 50 + (forward_growth - 0.08) / 0.07 * 10
        elif forward_growth >= 0:
            score = 40 + forward_growth / 0.08 * 10
        elif forward_growth >= -0.10:
            score = 30 + (forward_growth + 0.10) / 0.10 * 10
        else:
            score = max(15, 30 + (forward_growth + 0.10) * 100)

        return {
            'score': min(100, max(0, score)),
            'raw': forward_growth,
            'source': 'peg' if peg else 'pe_ratio' if trailing_pe else 'current_growth'
        }

    def _calc_fg4_growth_consistency(self) -> Dict:
        """FG4: Growth Consistency (분기별 변동성)"""

        if len(self.quarterly_data) < 6:
            return {'score': None, 'raw': None, 'reason': 'Insufficient quarterly data'}

        # 분기별 매출 성장률 계산 (YoY)
        revenue_growths = []

        for i in range(min(4, len(self.quarterly_data) - 4)):
            recent = self.quarterly_data[i]['revenue']
            yoy = self.quarterly_data[i + 4]['revenue']

            if recent and yoy and yoy > 0:
                growth = (recent - yoy) / yoy
                revenue_growths.append(growth)

        if len(revenue_growths) < 3:
            return {'score': None, 'raw': None, 'reason': 'Cannot calculate growth consistency'}

        # 성장률 평균 및 표준편차
        avg_growth = sum(revenue_growths) / len(revenue_growths)
        variance = sum((g - avg_growth) ** 2 for g in revenue_growths) / len(revenue_growths)
        std_growth = math.sqrt(variance)

        # Coefficient of Variation (변동계수)
        if avg_growth != 0:
            cv = abs(std_growth / avg_growth)
        else:
            cv = float('inf') if std_growth > 0 else 0

        # 일관된 양의 성장 확인
        positive_quarters = sum(1 for g in revenue_growths if g > 0)
        positive_ratio = positive_quarters / len(revenue_growths)

        # 점수 계산
        # 낮은 변동성 + 높은 양의 성장 비율 = 높은 점수
        if cv <= 0.3 and positive_ratio >= 0.75:
            base_score = 85 + min(15, (0.3 - cv) / 0.3 * 10 + positive_ratio * 5)
        elif cv <= 0.5 and positive_ratio >= 0.5:
            base_score = 65 + (0.5 - cv) / 0.2 * 10 + positive_ratio * 10
        elif cv <= 0.8 and positive_ratio >= 0.25:
            base_score = 45 + (0.8 - cv) / 0.3 * 10 + positive_ratio * 10
        elif positive_ratio > 0:
            base_score = 30 + positive_ratio * 15
        else:
            base_score = 20

        # 성장 추세 보너스 (최근 성장률 > 이전 성장률)
        if len(revenue_growths) >= 2:
            if revenue_growths[0] > revenue_growths[-1]:
                trend_bonus = 5
            elif revenue_growths[0] < revenue_growths[-1]:
                trend_bonus = -5
            else:
                trend_bonus = 0
        else:
            trend_bonus = 0

        final_score = base_score + trend_bonus

        return {
            'score': min(100, max(0, final_score)),
            'raw': cv,
            'avg_growth': avg_growth,
            'std_growth': std_growth,
            'positive_ratio': positive_ratio,
            'revenue_growths': revenue_growths
        }

    def _calc_fg5_revenue_acceleration(self) -> Dict:
        """FG5: Revenue Acceleration (성장 가속화)"""

        if len(self.quarterly_data) < 8:
            return {'score': None, 'raw': None, 'reason': 'Insufficient data'}

        # 최근 4분기 매출 성장률 (YoY)
        recent_growths = []
        older_growths = []

        for i in range(min(4, len(self.quarterly_data) - 4)):
            recent_rev = self.quarterly_data[i]['revenue']
            yoy_rev = self.quarterly_data[i + 4]['revenue']

            if recent_rev and yoy_rev and yoy_rev > 0:
                recent_growths.append((recent_rev - yoy_rev) / yoy_rev)

        # 4분기 전의 성장률 (비교용)
        if len(self.quarterly_data) >= 8:
            for i in range(4, min(8, len(self.quarterly_data) - 4)):
                recent_rev = self.quarterly_data[i]['revenue']
                yoy_rev = self.quarterly_data[i + 4]['revenue'] if i + 4 < len(self.quarterly_data) else None

                if recent_rev and yoy_rev and yoy_rev > 0:
                    older_growths.append((recent_rev - yoy_rev) / yoy_rev)

        if not recent_growths:
            return {'score': None, 'raw': None, 'reason': 'Cannot calculate acceleration'}

        recent_avg = sum(recent_growths) / len(recent_growths)
        older_avg = sum(older_growths) / len(older_growths) if older_growths else 0

        # 가속도 = 최근 성장률 - 이전 성장률
        acceleration = recent_avg - older_avg

        # 점수 계산
        if acceleration >= 0.10:  # 10%p 이상 가속
            score = 90 + min(10, (acceleration - 0.10) / 0.10 * 10)
        elif acceleration >= 0.05:
            score = 75 + (acceleration - 0.05) / 0.05 * 15
        elif acceleration >= 0:
            score = 55 + acceleration / 0.05 * 20
        elif acceleration >= -0.05:
            score = 40 + (acceleration + 0.05) / 0.05 * 15
        elif acceleration >= -0.10:
            score = 25 + (acceleration + 0.10) / 0.05 * 15
        else:
            score = max(10, 25 + (acceleration + 0.10) * 100)

        # 절대 성장률 보너스
        if recent_avg > 0.20:
            abs_bonus = 5
        elif recent_avg > 0.10:
            abs_bonus = 3
        elif recent_avg < 0:
            abs_bonus = -5
        else:
            abs_bonus = 0

        final_score = score + abs_bonus

        return {
            'score': min(100, max(0, final_score)),
            'raw': acceleration,
            'recent_avg_growth': recent_avg,
            'older_avg_growth': older_avg
        }

    def _calc_fg6_profit_growth(self) -> Dict:
        """FG6: Profit Growth (이익 성장)"""

        if len(self.quarterly_data) < 5:
            return {'score': None, 'raw': None, 'reason': 'Insufficient data'}

        # Operating Income 성장률
        recent_oi = self.quarterly_data[0]['operating_income']
        yoy_oi = self.quarterly_data[4]['operating_income'] if len(self.quarterly_data) > 4 else None

        oi_growth = None
        if recent_oi and yoy_oi and yoy_oi != 0:
            oi_growth = (recent_oi - yoy_oi) / abs(yoy_oi)

        # Net Income 성장률
        recent_ni = self.quarterly_data[0]['net_income']
        yoy_ni = self.quarterly_data[4]['net_income'] if len(self.quarterly_data) > 4 else None

        ni_growth = None
        if recent_ni and yoy_ni and yoy_ni != 0:
            ni_growth = (recent_ni - yoy_ni) / abs(yoy_ni)

        # 두 지표 평균 (가능한 경우)
        growths = [g for g in [oi_growth, ni_growth] if g is not None]

        if not growths:
            # Operating Margin 변화로 대체
            op_margin = self.stock_data.get('operating_margin')
            net_margin = self.stock_data.get('net_margin')

            if op_margin and net_margin:
                # 마진율 기반 점수
                avg_margin = (op_margin + net_margin) / 2
                if avg_margin >= 0.20:
                    score = 70
                elif avg_margin >= 0.10:
                    score = 55
                elif avg_margin >= 0:
                    score = 45
                else:
                    score = 30

                return {
                    'score': score,
                    'raw': avg_margin,
                    'source': 'margin_level'
                }

            return {'score': None, 'raw': None, 'reason': 'No profit growth data'}

        profit_growth = sum(growths) / len(growths)

        # 점수 계산
        if profit_growth >= 0.50:
            score = 90 + min(10, (profit_growth - 0.50) / 0.50 * 10)
        elif profit_growth >= 0.25:
            score = 75 + (profit_growth - 0.25) / 0.25 * 15
        elif profit_growth >= 0.10:
            score = 60 + (profit_growth - 0.10) / 0.15 * 15
        elif profit_growth >= 0:
            score = 45 + profit_growth / 0.10 * 15
        elif profit_growth >= -0.20:
            score = 30 + (profit_growth + 0.20) / 0.20 * 15
        else:
            score = max(10, 30 + (profit_growth + 0.20) * 50)

        # 흑자 전환 보너스
        if recent_oi and yoy_oi:
            if yoy_oi < 0 and recent_oi > 0:  # 흑자 전환
                score = min(100, score + 10)
            elif yoy_oi > 0 and recent_oi < 0:  # 적자 전환
                score = max(0, score - 10)

        return {
            'score': min(100, max(0, score)),
            'raw': profit_growth,
            'oi_growth': oi_growth,
            'ni_growth': ni_growth
        }

    def _to_float(self, value) -> Optional[float]:
        """Decimal/None을 float로 변환"""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return float(value)
        try:
            return float(value)
        except:
            return None


# ========================================================================
# Convenience Function
# ========================================================================

async def calculate_growth_score(symbol: str, db_manager, sector_benchmarks: Dict,
                                  analysis_date: Optional[date] = None) -> Dict:
    """Growth Factor 점수 계산 (편의 함수)"""

    calculator = USGrowthFactorV2(symbol, db_manager, sector_benchmarks, analysis_date)
    return await calculator.calculate()


# ========================================================================
# Test
# ========================================================================

if __name__ == '__main__':
    import asyncio
    from us_db_async import AsyncDatabaseManager
    from us_sector_benchmarks import USSectorBenchmarks

    async def test():
        db = AsyncDatabaseManager()
        await db.initialize()

        # 섹터 기준값 로드
        sector_calc = USSectorBenchmarks(db)
        benchmarks = await sector_calc.get_sector_benchmarks('TECHNOLOGY')

        # 테스트 종목
        test_symbols = ['AAPL', 'NVDA', 'MSFT']

        print("\n" + "="*60)
        print("Growth Factor v2.0 Test")
        print("="*60)

        for symbol in test_symbols:
            result = await calculate_growth_score(symbol, db, benchmarks)
            print(f"\n{symbol}: Growth Score = {result['growth_score']}")

            for strat_id, strat_data in result['strategies'].items():
                if strat_data['score'] is not None:
                    raw = strat_data.get('raw', 'N/A')
                    if isinstance(raw, dict):
                        raw = 'complex'
                    print(f"  {strat_id}: {strat_data['score']:.1f} (raw: {raw})")

        await db.close()

    asyncio.run(test())
