"""
US Quality Factor v2.0 - Management Quality

핵심: 지속 가능한 고품질 기업 (일시적 수익이 아닌 구조적 우위)

Strategies:
- MQ1: Gross Margin (20%) - 총이익률 (섹터 대비 경쟁 우위)
- MQ2: ROIC (20%) - 투하자본수익률 (자본 효율성)
- MQ3: Operating Leverage (15%) - 영업 레버리지 (확장성)
- MQ4: Earnings Quality (15%) - 이익의 질 (현금 vs 발생주의)
- MQ5: Balance Sheet (15%) - 재무건전성 (부채/유동성)
- MQ6: Margin Trend (15%) - 마진 추세 (개선 vs 악화)

File: us/us_quality_factor_v2.py
"""

import logging
from typing import Dict, Optional, List
from datetime import date
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# Strategy Weights
# ========================================================================

STRATEGY_WEIGHTS = {
    'MQ1': 0.20,  # Gross Margin
    'MQ2': 0.20,  # ROIC
    'MQ3': 0.15,  # Operating Leverage
    'MQ4': 0.15,  # Earnings Quality
    'MQ5': 0.15,  # Balance Sheet
    'MQ6': 0.15   # Margin Trend
}

# 섹터별 Gross Margin 기준 (구조적으로 마진이 다름)
SECTOR_GROSS_MARGIN_THRESHOLDS = {
    'TECHNOLOGY': {'excellent': 0.65, 'good': 0.50, 'fair': 0.35},
    'HEALTHCARE': {'excellent': 0.60, 'good': 0.45, 'fair': 0.30},
    'CONSUMER DEFENSIVE': {'excellent': 0.40, 'good': 0.30, 'fair': 0.20},
    'FINANCIAL SERVICES': {'excellent': 0.50, 'good': 0.35, 'fair': 0.20},
    'INDUSTRIALS': {'excellent': 0.35, 'good': 0.25, 'fair': 0.15},
    'ENERGY': {'excellent': 0.40, 'good': 0.30, 'fair': 0.20},
    'UTILITIES': {'excellent': 0.45, 'good': 0.35, 'fair': 0.25},
    'DEFAULT': {'excellent': 0.45, 'good': 0.35, 'fair': 0.25}
}

# ROIC 기준 (WACC 대비 - 일반적으로 8-10% 가정)
ROIC_THRESHOLDS = {
    'excellent': 0.20,  # 20%+ ROIC = 명확한 경쟁 우위
    'good': 0.12,       # 12-20% = 양호
    'fair': 0.08,       # 8-12% = WACC 수준
    'poor': 0.05        # 5-8% = WACC 하회
}


class USQualityFactorV2:
    """Quality Factor v2.0 - Management Quality"""

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
        self.historical_data = []

    async def calculate(self) -> Dict:
        """
        Quality Factor 점수 계산

        Returns:
            {
                'quality_score': float (0-100),
                'strategies': {
                    'MQ1': {'score': float, 'raw': float, ...},
                    ...
                }
            }
        """
        try:
            # 1. 종목 데이터 조회
            await self._load_stock_data()
            await self._load_historical_data()

            if not self.stock_data:
                logger.warning(f"{self.symbol}: No data available")
                return {'quality_score': 50.0, 'strategies': {}}

            # 2. 각 전략별 점수 계산
            strategies = {}
            strategies['MQ1'] = self._calc_mq1_gross_margin()
            strategies['MQ2'] = self._calc_mq2_roic()
            strategies['MQ3'] = self._calc_mq3_operating_leverage()
            strategies['MQ4'] = self._calc_mq4_earnings_quality()
            strategies['MQ5'] = self._calc_mq5_balance_sheet()
            strategies['MQ6'] = self._calc_mq6_margin_trend()

            # 3. 가중 평균 계산
            total_weight = 0
            weighted_sum = 0

            for strategy_id, result in strategies.items():
                if result['score'] is not None:
                    weight = STRATEGY_WEIGHTS[strategy_id]
                    weighted_sum += result['score'] * weight
                    total_weight += weight

            quality_score = weighted_sum / total_weight if total_weight > 0 else 50.0

            logger.info(f"{self.symbol}: Quality Score = {quality_score:.1f}")

            return {
                'quality_score': round(quality_score, 1),
                'strategies': strategies
            }

        except Exception as e:
            logger.error(f"{self.symbol}: Quality calculation failed - {e}")
            return {'quality_score': 50.0, 'strategies': {}}

    async def _load_stock_data(self):
        """종목 기본 데이터 로드"""

        query = """
        SELECT
            b.symbol,
            b.sector,
            b.grossprofitttm as gross_profit,
            b.revenuettm as revenue,
            b.operatingmarginttm as operating_margin,
            b.profitmargin as net_margin,
            b.returnonequityttm as roe,
            b.returnonassetsttm as roa,
            b.ebitda,
            b.market_cap,
            b.beta,
            b.sharesoutstanding as shares_outstanding,
            b.bookvalue as book_value,
            b.dilutedepsttm as eps
        FROM us_stock_basic b
        WHERE b.symbol = $1
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result and result[0]:
                row = result[0]
                self.sector = row['sector']
                self.stock_data = {
                    'gross_profit': self._to_float(row['gross_profit']),
                    'revenue': self._to_float(row['revenue']),
                    'operating_margin': self._to_float(row['operating_margin']),
                    'net_margin': self._to_float(row['net_margin']),
                    'roe': self._to_float(row['roe']),
                    'roa': self._to_float(row['roa']),
                    'ebitda': self._to_float(row['ebitda']),
                    'market_cap': self._to_float(row['market_cap']),
                    'beta': self._to_float(row['beta']),
                    'shares_outstanding': self._to_float(row['shares_outstanding']),
                    'book_value': self._to_float(row['book_value']),
                    'eps': self._to_float(row['eps'])
                }

            # Gross Margin 계산
            if self.stock_data.get('gross_profit') and self.stock_data.get('revenue'):
                self.stock_data['gross_margin'] = self.stock_data['gross_profit'] / self.stock_data['revenue']

            # Balance Sheet 데이터 추가
            await self._load_balance_sheet_data()

            # Cash Flow 데이터 추가 (Earnings Quality용)
            await self._load_cash_flow_data()

        except Exception as e:
            logger.error(f"{self.symbol}: Failed to load stock data - {e}")

    async def _load_balance_sheet_data(self):
        """Balance Sheet 데이터 조회"""

        query = """
        SELECT
            total_assets,
            total_liabilities,
            total_shareholder_equity,
            total_current_assets,
            total_current_liabilities,
            long_term_debt,
            short_term_debt,
            cash_and_cash_equivalents,
            total_debt
        FROM us_balance_sheet
        WHERE symbol = $1
        ORDER BY fiscal_date_ending DESC
        LIMIT 1
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result and result[0]:
                row = result[0]
                self.stock_data.update({
                    'total_assets': self._to_float(row['total_assets']),
                    'total_liabilities': self._to_float(row['total_liabilities']),
                    'shareholder_equity': self._to_float(row['total_shareholder_equity']),
                    'current_assets': self._to_float(row['total_current_assets']),
                    'current_liabilities': self._to_float(row['total_current_liabilities']),
                    'long_term_debt': self._to_float(row['long_term_debt']),
                    'short_term_debt': self._to_float(row['short_term_debt']),
                    'cash': self._to_float(row['cash_and_cash_equivalents']),
                    'total_debt': self._to_float(row['total_debt'])
                })

        except Exception as e:
            logger.warning(f"{self.symbol}: Balance sheet data not available - {e}")

    async def _load_cash_flow_data(self):
        """Cash Flow 데이터 조회"""

        query = """
        SELECT
            operating_cashflow,
            net_income,
            depreciation_and_amortization,
            capital_expenditures
        FROM us_cash_flow
        WHERE symbol = $1
        ORDER BY fiscal_date_ending DESC
        LIMIT 4
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result:
                # TTM 합계
                total_ocf = sum(self._to_float(r['operating_cashflow']) or 0 for r in result)
                total_ni = sum(self._to_float(r['net_income']) or 0 for r in result)
                total_da = sum(self._to_float(r['depreciation_and_amortization']) or 0 for r in result)
                total_capex = sum(abs(self._to_float(r['capital_expenditures']) or 0) for r in result)

                self.stock_data.update({
                    'operating_cashflow_ttm': total_ocf,
                    'net_income_ttm': total_ni,
                    'depreciation_ttm': total_da,
                    'capex_ttm': total_capex,
                    'fcf_ttm': total_ocf - total_capex
                })

        except Exception as e:
            logger.warning(f"{self.symbol}: Cash flow data not available - {e}")

    async def _load_historical_data(self):
        """과거 분기 데이터 로드 (Margin Trend용)"""

        query = """
        SELECT
            fiscal_date_ending,
            gross_profit,
            total_revenue,
            operating_income
        FROM us_income_statement
        WHERE symbol = $1
        ORDER BY fiscal_date_ending DESC
        LIMIT 8
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result:
                self.historical_data = []
                for row in result:
                    gross_profit = self._to_float(row['gross_profit'])
                    revenue = self._to_float(row['total_revenue'])
                    operating_income = self._to_float(row['operating_income'])

                    gross_margin = gross_profit / revenue if gross_profit and revenue else None
                    op_margin = operating_income / revenue if operating_income and revenue else None

                    self.historical_data.append({
                        'date': row['fiscal_date_ending'],
                        'gross_margin': gross_margin,
                        'operating_margin': op_margin,
                        'revenue': revenue
                    })

        except Exception as e:
            logger.warning(f"{self.symbol}: Historical data not available - {e}")

    def _calc_mq1_gross_margin(self) -> Dict:
        """MQ1: Gross Margin (섹터 대비)"""

        gross_margin = self.stock_data.get('gross_margin')
        sector_median = self.benchmarks.get('gross_margin_median')

        if gross_margin is None:
            return {'score': None, 'raw': None, 'reason': 'No gross margin data'}

        # 섹터별 절대 기준
        thresholds = SECTOR_GROSS_MARGIN_THRESHOLDS.get(
            self.sector, SECTOR_GROSS_MARGIN_THRESHOLDS['DEFAULT']
        )

        # 절대 기준 점수 (60%)
        if gross_margin >= thresholds['excellent']:
            abs_score = 90 + min(10, (gross_margin - thresholds['excellent']) / 0.1 * 10)
        elif gross_margin >= thresholds['good']:
            abs_score = 70 + (gross_margin - thresholds['good']) / (thresholds['excellent'] - thresholds['good']) * 20
        elif gross_margin >= thresholds['fair']:
            abs_score = 50 + (gross_margin - thresholds['fair']) / (thresholds['good'] - thresholds['fair']) * 20
        else:
            abs_score = max(20, 50 - (thresholds['fair'] - gross_margin) / thresholds['fair'] * 30)

        # 섹터 대비 점수 (40%)
        relative_score = 50  # 기본값
        if sector_median and sector_median > 0:
            ratio = gross_margin / sector_median
            if ratio >= 1.3:
                relative_score = 90
            elif ratio >= 1.1:
                relative_score = 70 + (ratio - 1.1) / 0.2 * 20
            elif ratio >= 0.9:
                relative_score = 50 + (ratio - 0.9) / 0.2 * 20
            elif ratio >= 0.7:
                relative_score = 30 + (ratio - 0.7) / 0.2 * 20
            else:
                relative_score = max(10, 30 - (0.7 - ratio) / 0.2 * 20)

        # 가중 평균
        score = abs_score * 0.6 + relative_score * 0.4

        return {
            'score': min(100, max(0, score)),
            'raw': gross_margin,
            'sector_median': sector_median,
            'thresholds': thresholds
        }

    def _calc_mq2_roic(self) -> Dict:
        """MQ2: ROIC (투하자본수익률)"""

        # ROIC 계산 시도
        # ROIC = NOPAT / Invested Capital
        # NOPAT ≈ Operating Income * (1 - Tax Rate) 또는 EBIT * (1 - 21%)
        # Invested Capital = Total Equity + Total Debt - Cash

        ebitda = self.stock_data.get('ebitda')
        equity = self.stock_data.get('shareholder_equity')
        total_debt = self.stock_data.get('total_debt')
        cash = self.stock_data.get('cash')

        # 대안: ROE 사용
        roe = self.stock_data.get('roe')

        roic = None

        if ebitda and equity:
            # 간이 ROIC 계산
            invested_capital = (equity or 0) + (total_debt or 0) - (cash or 0)
            if invested_capital > 0:
                # NOPAT ≈ EBITDA * 0.75 (세금 및 D&A 조정 근사)
                nopat_approx = ebitda * 0.75
                roic = nopat_approx / invested_capital

        if roic is None and roe:
            # ROE를 ROIC 대용으로 사용 (약간 할인)
            roic = roe * 0.85 if isinstance(roe, (int, float)) else self._to_float(roe) * 0.85

        if roic is None:
            return {'score': None, 'raw': None, 'reason': 'Cannot calculate ROIC'}

        # 점수 계산
        if roic >= ROIC_THRESHOLDS['excellent']:
            score = 90 + min(10, (roic - ROIC_THRESHOLDS['excellent']) / 0.1 * 10)
        elif roic >= ROIC_THRESHOLDS['good']:
            score = 70 + (roic - ROIC_THRESHOLDS['good']) / (ROIC_THRESHOLDS['excellent'] - ROIC_THRESHOLDS['good']) * 20
        elif roic >= ROIC_THRESHOLDS['fair']:
            score = 50 + (roic - ROIC_THRESHOLDS['fair']) / (ROIC_THRESHOLDS['good'] - ROIC_THRESHOLDS['fair']) * 20
        elif roic >= ROIC_THRESHOLDS['poor']:
            score = 30 + (roic - ROIC_THRESHOLDS['poor']) / (ROIC_THRESHOLDS['fair'] - ROIC_THRESHOLDS['poor']) * 20
        elif roic >= 0:
            score = 20 + roic / ROIC_THRESHOLDS['poor'] * 10
        else:
            score = max(5, 20 + roic * 100)  # 음수 ROIC 패널티

        return {
            'score': min(100, max(0, score)),
            'raw': roic,
            'roe': roe
        }

    def _calc_mq3_operating_leverage(self) -> Dict:
        """MQ3: Operating Leverage (확장성)"""

        # Operating Leverage = 매출 성장 대비 영업이익 성장
        # 높으면 규모의 경제 효과

        if len(self.historical_data) < 5:
            return {'score': None, 'raw': None, 'reason': 'Insufficient historical data'}

        # 최근 4분기 vs 이전 4분기 비교
        recent = self.historical_data[:4]
        older = self.historical_data[4:8] if len(self.historical_data) >= 8 else self.historical_data[4:]

        if len(older) < 2:
            return {'score': None, 'raw': None, 'reason': 'Insufficient comparison data'}

        # 매출 성장
        recent_rev = sum(r['revenue'] or 0 for r in recent)
        older_rev = sum(r['revenue'] or 0 for r in older)

        if older_rev <= 0:
            return {'score': None, 'raw': None, 'reason': 'Invalid revenue data'}

        rev_growth = (recent_rev - older_rev) / older_rev

        # Operating Margin 변화
        recent_op_margins = [r['operating_margin'] for r in recent if r['operating_margin'] is not None]
        older_op_margins = [r['operating_margin'] for r in older if r['operating_margin'] is not None]

        if not recent_op_margins or not older_op_margins:
            return {'score': None, 'raw': None, 'reason': 'No margin data'}

        recent_avg_margin = sum(recent_op_margins) / len(recent_op_margins)
        older_avg_margin = sum(older_op_margins) / len(older_op_margins)
        margin_change = recent_avg_margin - older_avg_margin

        # Operating Leverage 계산
        # 매출이 늘면서 마진도 개선 = 높은 레버리지
        if rev_growth > 0:
            # 매출 성장 중 마진 개선
            leverage_score = margin_change / rev_growth if rev_growth != 0 else 0
        else:
            # 매출 감소 중 마진 방어
            leverage_score = margin_change  # 마진 방어력

        # 점수 계산
        if rev_growth > 0.05 and margin_change > 0.02:
            # 매출 성장 + 마진 개선 = 최고
            score = 80 + min(20, margin_change * 200)
        elif rev_growth > 0 and margin_change >= 0:
            # 매출 성장 + 마진 유지
            score = 60 + rev_growth * 100 + margin_change * 100
        elif rev_growth > 0 and margin_change > -0.02:
            # 매출 성장, 소폭 마진 하락 (투자기)
            score = 50 + rev_growth * 50
        elif margin_change >= 0:
            # 매출 정체/하락, 마진 개선
            score = 45 + margin_change * 150
        else:
            # 마진 악화
            score = max(15, 45 + margin_change * 200)

        return {
            'score': min(100, max(0, score)),
            'raw': leverage_score,
            'revenue_growth': rev_growth,
            'margin_change': margin_change,
            'recent_margin': recent_avg_margin,
            'older_margin': older_avg_margin
        }

    def _calc_mq4_earnings_quality(self) -> Dict:
        """MQ4: Earnings Quality (OCF vs Net Income)"""

        ocf = self.stock_data.get('operating_cashflow_ttm')
        net_income = self.stock_data.get('net_income_ttm')

        if ocf is None or net_income is None:
            return {'score': None, 'raw': None, 'reason': 'No cash flow data'}

        if net_income <= 0:
            # 적자 기업
            if ocf > 0:
                # 적자지만 영업현금흐름은 양수 (감가상각 등)
                score = 45
                quality_ratio = None
            else:
                score = 25
                quality_ratio = None
        else:
            # OCF / Net Income 비율
            quality_ratio = ocf / net_income

            # 이상적: OCF > Net Income (1.0 이상)
            if quality_ratio >= 1.5:
                score = 95  # 매우 높은 현금 창출
            elif quality_ratio >= 1.2:
                score = 85 + (quality_ratio - 1.2) / 0.3 * 10
            elif quality_ratio >= 1.0:
                score = 70 + (quality_ratio - 1.0) / 0.2 * 15
            elif quality_ratio >= 0.8:
                score = 55 + (quality_ratio - 0.8) / 0.2 * 15
            elif quality_ratio >= 0.5:
                score = 35 + (quality_ratio - 0.5) / 0.3 * 20
            else:
                # 발생주의 이익 과다 (위험 신호)
                score = max(10, 35 - (0.5 - quality_ratio) * 50)

        return {
            'score': min(100, max(0, score)),
            'raw': quality_ratio,
            'operating_cashflow': ocf,
            'net_income': net_income
        }

    def _calc_mq5_balance_sheet(self) -> Dict:
        """MQ5: Balance Sheet 건전성"""

        total_debt = self.stock_data.get('total_debt') or 0
        cash = self.stock_data.get('cash') or 0
        equity = self.stock_data.get('shareholder_equity')
        current_assets = self.stock_data.get('current_assets')
        current_liabilities = self.stock_data.get('current_liabilities')
        ebitda = self.stock_data.get('ebitda')

        scores = []

        # 1. Net Debt / EBITDA (레버리지)
        if ebitda and ebitda > 0:
            net_debt = total_debt - cash
            net_debt_ebitda = net_debt / ebitda

            if net_debt_ebitda <= 0:  # 순현금
                debt_score = 95
            elif net_debt_ebitda <= 1:
                debt_score = 80 + (1 - net_debt_ebitda) * 15
            elif net_debt_ebitda <= 2:
                debt_score = 60 + (2 - net_debt_ebitda) * 20
            elif net_debt_ebitda <= 3:
                debt_score = 40 + (3 - net_debt_ebitda) * 20
            elif net_debt_ebitda <= 5:
                debt_score = 20 + (5 - net_debt_ebitda) / 2 * 20
            else:
                debt_score = max(5, 20 - (net_debt_ebitda - 5) * 3)

            scores.append(('debt_score', debt_score, net_debt_ebitda))

        # 2. Current Ratio (유동비율)
        if current_assets and current_liabilities and current_liabilities > 0:
            current_ratio = current_assets / current_liabilities

            if current_ratio >= 2.5:
                liquidity_score = 90
            elif current_ratio >= 2.0:
                liquidity_score = 75 + (current_ratio - 2.0) / 0.5 * 15
            elif current_ratio >= 1.5:
                liquidity_score = 60 + (current_ratio - 1.5) / 0.5 * 15
            elif current_ratio >= 1.0:
                liquidity_score = 40 + (current_ratio - 1.0) / 0.5 * 20
            else:
                liquidity_score = max(10, current_ratio * 40)

            scores.append(('liquidity_score', liquidity_score, current_ratio))

        # 3. Debt / Equity (자본 구조)
        if equity and equity > 0:
            de_ratio = total_debt / equity

            if de_ratio <= 0.2:
                de_score = 90
            elif de_ratio <= 0.5:
                de_score = 70 + (0.5 - de_ratio) / 0.3 * 20
            elif de_ratio <= 1.0:
                de_score = 50 + (1.0 - de_ratio) / 0.5 * 20
            elif de_ratio <= 2.0:
                de_score = 30 + (2.0 - de_ratio) / 1.0 * 20
            else:
                de_score = max(10, 30 - (de_ratio - 2.0) * 10)

            scores.append(('de_score', de_score, de_ratio))

        if not scores:
            return {'score': None, 'raw': None, 'reason': 'No balance sheet data'}

        # 가중 평균
        final_score = sum(s[1] for s in scores) / len(scores)

        return {
            'score': min(100, max(0, final_score)),
            'raw': {name: value for name, score, value in scores},
            'components': {name: score for name, score, value in scores}
        }

    def _calc_mq6_margin_trend(self) -> Dict:
        """MQ6: Margin Trend (개선 vs 악화)"""

        if len(self.historical_data) < 4:
            return {'score': None, 'raw': None, 'reason': 'Insufficient historical data'}

        # Gross Margin 추세
        gross_margins = [r['gross_margin'] for r in self.historical_data if r['gross_margin'] is not None]
        op_margins = [r['operating_margin'] for r in self.historical_data if r['operating_margin'] is not None]

        if len(gross_margins) < 3 and len(op_margins) < 3:
            return {'score': None, 'raw': None, 'reason': 'No margin trend data'}

        scores = []

        # Gross Margin 추세 분석
        if len(gross_margins) >= 3:
            gm_trend = self._calculate_trend(gross_margins)
            # 최근값 기준 개선/악화 판단
            gm_recent = gross_margins[0]
            gm_older = gross_margins[-1]
            gm_change = gm_recent - gm_older

            if gm_trend > 0 and gm_change > 0.02:
                gm_score = 80 + min(20, gm_change * 200)
            elif gm_trend > 0 or gm_change > 0:
                gm_score = 60 + min(20, max(gm_change, 0) * 150)
            elif gm_change > -0.02:
                gm_score = 50 + gm_change * 100
            else:
                gm_score = max(20, 50 + gm_change * 150)

            scores.append(gm_score)

        # Operating Margin 추세 분석
        if len(op_margins) >= 3:
            op_trend = self._calculate_trend(op_margins)
            op_recent = op_margins[0]
            op_older = op_margins[-1]
            op_change = op_recent - op_older

            if op_trend > 0 and op_change > 0.02:
                op_score = 80 + min(20, op_change * 200)
            elif op_trend > 0 or op_change > 0:
                op_score = 60 + min(20, max(op_change, 0) * 150)
            elif op_change > -0.02:
                op_score = 50 + op_change * 100
            else:
                op_score = max(20, 50 + op_change * 150)

            scores.append(op_score)

        if not scores:
            return {'score': None, 'raw': None, 'reason': 'Cannot calculate trend'}

        final_score = sum(scores) / len(scores)

        return {
            'score': min(100, max(0, final_score)),
            'raw': {
                'gross_margin_change': gm_change if len(gross_margins) >= 3 else None,
                'operating_margin_change': op_change if len(op_margins) >= 3 else None
            },
            'gross_margins': gross_margins[:4] if gross_margins else None,
            'op_margins': op_margins[:4] if op_margins else None
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """간단한 선형 추세 계산 (양수 = 상승 추세)"""
        if len(values) < 2:
            return 0

        # 시간 역순이므로 뒤집어서 계산
        values = values[::-1]
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator > 0 else 0

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

async def calculate_quality_score(symbol: str, db_manager, sector_benchmarks: Dict,
                                   analysis_date: Optional[date] = None) -> Dict:
    """Quality Factor 점수 계산 (편의 함수)"""

    calculator = USQualityFactorV2(symbol, db_manager, sector_benchmarks, analysis_date)
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
        print("Quality Factor v2.0 Test")
        print("="*60)

        for symbol in test_symbols:
            result = await calculate_quality_score(symbol, db, benchmarks)
            print(f"\n{symbol}: Quality Score = {result['quality_score']}")

            for strat_id, strat_data in result['strategies'].items():
                if strat_data['score'] is not None:
                    raw = strat_data.get('raw', 'N/A')
                    if isinstance(raw, dict):
                        raw = 'complex'
                    print(f"  {strat_id}: {strat_data['score']:.1f} (raw: {raw})")

        await db.close()

    asyncio.run(test())
