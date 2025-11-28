"""
US Quant System v2.0 - Main Integration Module

핵심 변경:
1. 시장 레짐 기반 동적 가중치
2. 섹터별 기준값 적용
3. 절대 점수 기반 등급 (섹터 상대 순위 아님)
4. 4개 Factor v2.0 통합

Usage:
    python us_main_v2.py

File: us/us_main_v2.py
"""

import asyncio
import logging
import json
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import numpy as np

# Local imports
from us_db_async import AsyncDatabaseManager
from us_market_regime import USMarketRegimeDetector
from us_sector_benchmarks import USSectorBenchmarks
from us_value_factor_v2 import calculate_value_score
from us_quality_factor_v2 import calculate_quality_score
from us_momentum_factor_v2 import calculate_momentum_score
from us_growth_factor_v2 import calculate_growth_score
from us_factor_interactions import USFactorInteractions
from us_agent_metrics import USAgentMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# Grade Thresholds (Absolute Score Based)
# ========================================================================

# 절대 점수 기준 등급 (섹터 상대 순위 아님!)
# "아프리카 상위 15% 부자 vs 미국 상위 15% 부자" 문제 해결
GRADE_THRESHOLDS = {
    'A+': 85,  # 85점 이상 = 최우수
    'A':  75,  # 75-85점 = 우수
    'B+': 65,  # 65-75점 = 양호
    'B':  55,  # 55-65점 = 보통 이상
    'C':  45,  # 45-55점 = 보통
    'D':  35,  # 35-45점 = 주의
    'F':  0    # 35점 미만 = 위험
}

# 등급별 투자 의견
GRADE_OPINION = {
    'A+': '적극 매수',
    'A':  '매수',
    'B+': '관심',
    'B':  '중립',
    'C':  '관망',
    'D':  '주의',
    'F':  '매도'
}


class USQuantSystemV2:
    """US Quant System v2.0"""

    def __init__(self, db_manager: AsyncDatabaseManager):
        self.db = db_manager
        self.sector_benchmarks = USSectorBenchmarks(db_manager)

        # Cache
        self.current_regime = None
        self.regime_weights = None
        self.sector_cache = {}

    async def run(self, analysis_date: Optional[date] = None,
                  symbols: Optional[List[str]] = None) -> Dict:
        """
        전체 분석 실행

        Args:
            analysis_date: 분석 날짜 (기본: 오늘)
            symbols: 분석할 종목 리스트 (기본: 전체)

        Returns:
            {
                'date': date,
                'regime': str,
                'total_processed': int,
                'grade_distribution': Dict,
                'top_stocks': List
            }
        """
        analysis_date = analysis_date or date.today()
        logger.info(f"Starting US Quant System v2.0 - Date: {analysis_date}")

        # 1. 시장 레짐 감지
        logger.info("Step 1: Detecting market regime...")
        await self._detect_regime(analysis_date)
        logger.info(f"  Regime: {self.current_regime}")
        logger.info(f"  Weights: {self.regime_weights}")

        # 2. 분석 대상 종목 로드
        if symbols is None:
            symbols = await self._load_analysis_symbols()
        logger.info(f"Step 2: Loaded {len(symbols)} symbols for analysis")

        # 3. 각 종목 분석
        logger.info("Step 3: Analyzing stocks...")
        results = []
        processed = 0
        errors = 0

        for symbol in symbols:
            try:
                result = await self._analyze_stock(symbol, analysis_date)
                if result:
                    results.append(result)
                    processed += 1

                    if processed % 100 == 0:
                        logger.info(f"  Processed {processed}/{len(symbols)} stocks")

            except Exception as e:
                logger.error(f"  {symbol}: Analysis failed - {e}")
                errors += 1

        logger.info(f"  Total: {processed} processed, {errors} errors")

        # 4. 결과 저장
        logger.info("Step 4: Saving results...")
        await self._save_results(results, analysis_date)

        # 5. 통계 생성
        grade_dist = self._calculate_grade_distribution(results)
        top_stocks = sorted(results, key=lambda x: x['total_score'], reverse=True)[:20]

        logger.info("="*60)
        logger.info("US Quant System v2.0 Complete")
        logger.info(f"  Date: {analysis_date}")
        logger.info(f"  Regime: {self.current_regime}")
        logger.info(f"  Total Processed: {processed}")
        logger.info(f"  Grade Distribution: {grade_dist}")
        logger.info("="*60)

        return {
            'date': analysis_date,
            'regime': self.current_regime,
            'total_processed': processed,
            'grade_distribution': grade_dist,
            'top_stocks': [{'symbol': s['symbol'], 'score': s['total_score'], 'grade': s['grade']}
                           for s in top_stocks]
        }

    async def _detect_regime(self, analysis_date: date):
        """시장 레짐 감지 및 가중치 설정"""

        detector = USMarketRegimeDetector(self.db, analysis_date)
        regime_result = await detector.detect_regime()

        self.current_regime = regime_result['regime']
        self.regime_weights = regime_result['weights']

        # 레짐 저장
        try:
            await detector.save_to_db(regime_result)
        except Exception as e:
            logger.warning(f"Regime save failed: {e}")

    async def _load_analysis_symbols(self) -> List[str]:
        """분석 대상 종목 로드"""

        query = """
        SELECT DISTINCT symbol
        FROM us_stock_basic
        WHERE sector IS NOT NULL
          AND sector != ''
        ORDER BY symbol
        """

        result = await self.db.execute_query(query)
        return [r['symbol'] for r in result] if result else []

    async def _get_sector_benchmarks(self, sector: str) -> Dict:
        """섹터 벤치마크 로드 (캐시 사용)"""

        if sector not in self.sector_cache:
            benchmarks = await self.sector_benchmarks.get_sector_benchmarks(sector)
            self.sector_cache[sector] = benchmarks

        return self.sector_cache[sector]

    async def _analyze_stock(self, symbol: str, analysis_date: date) -> Optional[Dict]:
        """개별 종목 분석"""

        # 종목 기본 정보
        stock_info = await self._get_stock_info(symbol)
        if not stock_info:
            return None

        sector = stock_info.get('sector')
        if not sector:
            return None

        # 섹터 벤치마크 로드
        benchmarks = await self._get_sector_benchmarks(sector)

        # 4개 Factor 점수 + VaR/CVaR + RS Value 계산 (병렬 처리)
        value_task = calculate_value_score(symbol, self.db, benchmarks, analysis_date)
        quality_task = calculate_quality_score(symbol, self.db, benchmarks, analysis_date)
        momentum_task = calculate_momentum_score(symbol, self.db, benchmarks, analysis_date)
        growth_task = calculate_growth_score(symbol, self.db, benchmarks, analysis_date)
        var_cvar_task = self._calculate_var_cvar(symbol, analysis_date)
        rs_value_task = self._calculate_rs_value(symbol, analysis_date)

        value_result, quality_result, momentum_result, growth_result, var_cvar_result, rs_value = await asyncio.gather(
            value_task, quality_task, momentum_task, growth_task, var_cvar_task, rs_value_task
        )

        # 개별 점수
        value_score = value_result['value_score']
        quality_score = quality_result['quality_score']
        momentum_score = momentum_result['momentum_score']
        growth_score = growth_result['growth_score']

        # Base score (기존 가중합)
        base_score = (
            value_score * self.regime_weights['value'] +
            quality_score * self.regime_weights['quality'] +
            momentum_score * self.regime_weights['momentum'] +
            growth_score * self.regime_weights['growth']
        )

        # Factor Interaction 계산 (Phase 3)
        interaction_calc = USFactorInteractions({
            'value_score': value_score,
            'quality_score': quality_score,
            'momentum_score': momentum_score,
            'growth_score': growth_score
        })
        interaction_result = interaction_calc.calculate()
        interaction_score = interaction_result['interaction_score']
        conviction_score = interaction_result['conviction_score']

        # Final score = base 70% + interaction 30%
        total_score = base_score * 0.70 + interaction_score * 0.30

        # 등급 결정 (절대 점수 기반)
        grade = self._determine_grade(total_score)
        opinion = GRADE_OPINION[grade]

        # Agent Metrics 계산 (Phase 4)
        agent_metrics = USAgentMetrics(self.db)
        agent_result = await agent_metrics.calculate_all(
            symbol=symbol,
            analysis_date=analysis_date,
            current_score=total_score,
            sector=sector,
            var_95=var_cvar_result.get('var_95'),
            sector_percentile=None
        )

        return {
            'symbol': symbol,
            'sector': sector,
            'total_score': round(total_score, 1),
            'base_score': round(base_score, 1),
            'interaction_score': round(interaction_score, 1),
            'conviction_score': round(conviction_score, 1),
            'grade': grade,
            'opinion': opinion,
            'value_score': value_score,
            'quality_score': quality_score,
            'momentum_score': momentum_score,
            'growth_score': growth_score,
            'value_detail': value_result.get('strategies', {}),
            'quality_detail': quality_result.get('strategies', {}),
            'momentum_detail': momentum_result.get('strategies', {}),
            'growth_detail': growth_result.get('strategies', {}),
            'regime': self.current_regime,
            'regime_weights': self.regime_weights,
            'var_95': var_cvar_result['var_95'],
            'cvar_95': var_cvar_result['cvar_95'],
            'rs_value': rs_value,
            # Agent Metrics (Phase 4)
            'atr_pct': agent_result.get('atr_pct'),
            'stop_loss_pct': agent_result.get('stop_loss_pct'),
            'take_profit_pct': agent_result.get('take_profit_pct'),
            'risk_reward_ratio': agent_result.get('risk_reward_ratio'),
            'entry_timing_score': agent_result.get('entry_timing_score'),
            'score_trend_2w': agent_result.get('score_trend_2w'),
            'price_position_52w': agent_result.get('price_position_52w'),
            'position_size_pct': agent_result.get('position_size_pct'),
            'iv_percentile': agent_result.get('iv_percentile'),
            'insider_signal': agent_result.get('insider_signal'),
            'buy_triggers': agent_result.get('buy_triggers'),
            'sell_triggers': agent_result.get('sell_triggers'),
            'hold_triggers': agent_result.get('hold_triggers'),
            'scenario_bullish_prob': agent_result.get('scenario_bullish_prob'),
            'scenario_sideways_prob': agent_result.get('scenario_sideways_prob'),
            'scenario_bearish_prob': agent_result.get('scenario_bearish_prob'),
            'scenario_bullish_return': agent_result.get('scenario_bullish_return'),
            'scenario_sideways_return': agent_result.get('scenario_sideways_return'),
            'scenario_bearish_return': agent_result.get('scenario_bearish_return'),
            'scenario_sample_count': agent_result.get('scenario_sample_count')
        }

    async def _get_stock_info(self, symbol: str) -> Optional[Dict]:
        """종목 기본 정보 조회"""

        query = """
        SELECT symbol, sector, market_cap
        FROM us_stock_basic
        WHERE symbol = $1
        """

        result = await self.db.execute_query(query, symbol)
        return dict(result[0]) if result else None

    def _determine_grade(self, score: float) -> str:
        """절대 점수 기반 등급 결정"""

        if score >= GRADE_THRESHOLDS['A+']:
            return 'A+'
        elif score >= GRADE_THRESHOLDS['A']:
            return 'A'
        elif score >= GRADE_THRESHOLDS['B+']:
            return 'B+'
        elif score >= GRADE_THRESHOLDS['B']:
            return 'B'
        elif score >= GRADE_THRESHOLDS['C']:
            return 'C'
        elif score >= GRADE_THRESHOLDS['D']:
            return 'D'
        else:
            return 'F'

    async def _calculate_var_cvar(self, symbol: str, analysis_date: date) -> Dict:
        """
        Historical VaR 95% and CVaR 95% calculation
        - us_daily: date(DATE), symbol(VARCHAR), close(DECIMAL(21,2))
        - VaR 95%: 5th percentile of daily returns (past 252 days)
        - CVaR 95%: Average of returns below VaR 95%
        """
        query = """
        SELECT close
        FROM us_daily
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT 253
        """
        rows = await self.db.execute_query(query, symbol, analysis_date)

        if not rows or len(rows) < 30:
            return {'var_95': None, 'cvar_95': None}

        # close: DECIMAL(21,2) -> float
        prices = [float(r['close']) for r in rows][::-1]  # Oldest first
        returns = [(prices[i] - prices[i-1]) / prices[i-1] * 100
                   for i in range(1, len(prices))]

        var_95 = float(np.percentile(returns, 5))  # 5th percentile (worst 5%)
        cvar_returns = [r for r in returns if r <= var_95]
        cvar_95 = float(np.mean(cvar_returns)) if cvar_returns else var_95

        return {
            'var_95': round(abs(var_95), 2),   # DECIMAL(6,2) - positive value for loss
            'cvar_95': round(abs(cvar_95), 2)  # DECIMAL(6,2)
        }

    async def _calculate_rs_value(self, symbol: str, analysis_date: date) -> Optional[float]:
        """
        IBD-style Relative Strength Value
        - us_daily: date(DATE), symbol(VARCHAR), close(DECIMAL(21,2))
        - RS = 3M return * 40% + 6M return * 20% + 9M return * 20% + 12M return * 20%
        - Return: rs_value for DECIMAL(21,2)
        """
        query = """
        WITH price_data AS (
            SELECT
                date,
                close,
                LAG(close, 63) OVER (ORDER BY date) as price_3m,
                LAG(close, 126) OVER (ORDER BY date) as price_6m,
                LAG(close, 189) OVER (ORDER BY date) as price_9m,
                LAG(close, 252) OVER (ORDER BY date) as price_12m
            FROM us_daily
            WHERE symbol = $1 AND date <= $2
            ORDER BY date DESC
            LIMIT 1
        )
        SELECT close, price_3m, price_6m, price_9m, price_12m
        FROM price_data
        WHERE price_12m IS NOT NULL
        """
        rows = await self.db.execute_query(query, symbol, analysis_date)

        if not rows:
            return None

        r = rows[0]
        close = float(r['close'])

        ret_3m = (close / float(r['price_3m']) - 1) * 100 if r['price_3m'] else 0
        ret_6m = (close / float(r['price_6m']) - 1) * 100 if r['price_6m'] else 0
        ret_9m = (close / float(r['price_9m']) - 1) * 100 if r['price_9m'] else 0
        ret_12m = (close / float(r['price_12m']) - 1) * 100 if r['price_12m'] else 0

        rs_value = ret_3m * 0.4 + ret_6m * 0.2 + ret_9m * 0.2 + ret_12m * 0.2
        return round(rs_value, 2)  # DECIMAL(21,2)

    def _calculate_grade_distribution(self, results: List[Dict]) -> Dict:
        """등급 분포 계산"""

        distribution = {'A+': 0, 'A': 0, 'B+': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}

        for r in results:
            grade = r['grade']
            distribution[grade] = distribution.get(grade, 0) + 1

        return distribution

    async def _save_results(self, results: List[Dict], analysis_date: date):
        """결과 저장"""

        if not results:
            return

        # us_stock_grade 테이블에 저장 (UPSERT)
        for result in results:
            await self._save_stock_grade(result, analysis_date)

    async def _save_stock_grade(self, result: Dict, analysis_date: date):
        """개별 종목 등급 저장"""

        # detail을 JSON으로 변환
        def serialize_detail(detail: Dict) -> str:
            """detail dict를 JSON 문자열로 변환"""
            clean_detail = {}
            for key, value in detail.items():
                if value and isinstance(value, dict):
                    clean_value = {}
                    for k, v in value.items():
                        if v is not None:
                            if isinstance(v, (int, float)):
                                clean_value[k] = round(v, 4) if isinstance(v, float) else v
                            elif isinstance(v, dict):
                                clean_value[k] = {kk: round(vv, 4) if isinstance(vv, float) else vv
                                                   for kk, vv in v.items() if vv is not None}
                            else:
                                clean_value[k] = v
                    clean_detail[key] = clean_value
            return json.dumps(clean_detail)

        # Serialize triggers to JSON
        def serialize_triggers(triggers: list) -> str:
            """Convert triggers list to JSON string"""
            if not triggers:
                return json.dumps([])
            return json.dumps(triggers)

        query = """
        INSERT INTO us_stock_grade (
            date, symbol,
            final_score, final_grade,
            value_score, quality_score, momentum_score, growth_score,
            value_v2_detail, quality_v2_detail,
            momentum_v2_detail, growth_v2_detail,
            var_95, cvar_95, rs_value,
            interaction_score, conviction_score,
            entry_timing_score, score_trend_2w, price_position_52w,
            atr_pct, stop_loss_pct, take_profit_pct, risk_reward_ratio, position_size_pct,
            scenario_bullish_prob, scenario_sideways_prob, scenario_bearish_prob,
            scenario_bullish_return, scenario_sideways_return, scenario_bearish_return,
            scenario_sample_count,
            buy_triggers, sell_triggers, hold_triggers,
            iv_percentile, insider_signal
        ) VALUES (
            $1, $2,
            $3, $4,
            $5, $6, $7, $8,
            $9::jsonb, $10::jsonb, $11::jsonb, $12::jsonb,
            $13, $14, $15,
            $16, $17,
            $18, $19, $20,
            $21, $22, $23, $24, $25,
            $26, $27, $28,
            $29, $30, $31,
            $32,
            $33::jsonb, $34::jsonb, $35::jsonb,
            $36, $37
        )
        ON CONFLICT (date, symbol) DO UPDATE SET
            final_score = EXCLUDED.final_score,
            final_grade = EXCLUDED.final_grade,
            value_score = EXCLUDED.value_score,
            quality_score = EXCLUDED.quality_score,
            momentum_score = EXCLUDED.momentum_score,
            growth_score = EXCLUDED.growth_score,
            value_v2_detail = EXCLUDED.value_v2_detail,
            quality_v2_detail = EXCLUDED.quality_v2_detail,
            momentum_v2_detail = EXCLUDED.momentum_v2_detail,
            growth_v2_detail = EXCLUDED.growth_v2_detail,
            var_95 = EXCLUDED.var_95,
            cvar_95 = EXCLUDED.cvar_95,
            rs_value = EXCLUDED.rs_value,
            interaction_score = EXCLUDED.interaction_score,
            conviction_score = EXCLUDED.conviction_score,
            entry_timing_score = EXCLUDED.entry_timing_score,
            score_trend_2w = EXCLUDED.score_trend_2w,
            price_position_52w = EXCLUDED.price_position_52w,
            atr_pct = EXCLUDED.atr_pct,
            stop_loss_pct = EXCLUDED.stop_loss_pct,
            take_profit_pct = EXCLUDED.take_profit_pct,
            risk_reward_ratio = EXCLUDED.risk_reward_ratio,
            position_size_pct = EXCLUDED.position_size_pct,
            scenario_bullish_prob = EXCLUDED.scenario_bullish_prob,
            scenario_sideways_prob = EXCLUDED.scenario_sideways_prob,
            scenario_bearish_prob = EXCLUDED.scenario_bearish_prob,
            scenario_bullish_return = EXCLUDED.scenario_bullish_return,
            scenario_sideways_return = EXCLUDED.scenario_sideways_return,
            scenario_bearish_return = EXCLUDED.scenario_bearish_return,
            scenario_sample_count = EXCLUDED.scenario_sample_count,
            buy_triggers = EXCLUDED.buy_triggers,
            sell_triggers = EXCLUDED.sell_triggers,
            hold_triggers = EXCLUDED.hold_triggers,
            iv_percentile = EXCLUDED.iv_percentile,
            insider_signal = EXCLUDED.insider_signal,
            updated_at = CURRENT_TIMESTAMP
        """

        try:
            await self.db.execute_query(
                query,
                analysis_date,                                      # $1
                result['symbol'],                                   # $2
                result['total_score'],                              # $3
                result['grade'],                                    # $4
                result['value_score'],                              # $5
                result['quality_score'],                            # $6
                result['momentum_score'],                           # $7
                result['growth_score'],                             # $8
                serialize_detail(result['value_detail']),           # $9
                serialize_detail(result['quality_detail']),         # $10
                serialize_detail(result['momentum_detail']),        # $11
                serialize_detail(result['growth_detail']),          # $12
                result.get('var_95'),                               # $13
                result.get('cvar_95'),                              # $14
                result.get('rs_value'),                             # $15
                result.get('interaction_score'),                    # $16
                result.get('conviction_score'),                     # $17
                result.get('entry_timing_score'),                   # $18
                result.get('score_trend_2w'),                       # $19
                result.get('price_position_52w'),                   # $20
                result.get('atr_pct'),                              # $21
                result.get('stop_loss_pct'),                        # $22
                result.get('take_profit_pct'),                      # $23
                result.get('risk_reward_ratio'),                    # $24
                result.get('position_size_pct'),                    # $25
                result.get('scenario_bullish_prob'),                # $26
                result.get('scenario_sideways_prob'),               # $27
                result.get('scenario_bearish_prob'),                # $28
                result.get('scenario_bullish_return'),              # $29
                result.get('scenario_sideways_return'),             # $30
                result.get('scenario_bearish_return'),              # $31
                result.get('scenario_sample_count'),                # $32
                serialize_triggers(result.get('buy_triggers')),     # $33
                serialize_triggers(result.get('sell_triggers')),    # $34
                serialize_triggers(result.get('hold_triggers')),    # $35
                result.get('iv_percentile'),                        # $36
                result.get('insider_signal')                        # $37
            )
        except Exception as e:
            logger.error(f"{result['symbol']}: Failed to save - {e}")


# ========================================================================
# Convenience Functions
# ========================================================================

async def analyze_single_stock_standalone(symbol: str, db_manager, analysis_date: Optional[date] = None,
                                          verbose: bool = True) -> Optional[Dict]:
    """단일 종목 분석 (standalone) - 디버깅 로그 포함"""

    analysis_date = analysis_date or date.today()

    if verbose:
        print(f"\n{'='*80}")
        print(f"미국 주식 v2.0 분석 시작: {symbol}")
        print(f"분석 날짜: {analysis_date}")
        print(f"{'='*80}\n")

    try:
        system = USQuantSystemV2(db_manager)

        # Step 1: 레짐 감지
        if verbose:
            print("Step 1: 시장 레짐 감지 중...")
        await system._detect_regime(analysis_date)
        if verbose:
            print(f"  레짐: {system.current_regime}")
            print(f"  가중치: Value={system.regime_weights['value']:.0%}, "
                  f"Quality={system.regime_weights['quality']:.0%}, "
                  f"Momentum={system.regime_weights['momentum']:.0%}, "
                  f"Growth={system.regime_weights['growth']:.0%}\n")

        # Step 2: 종목 기본 정보 확인
        if verbose:
            print("Step 2: 종목 기본 정보 확인 중...")
        stock_info = await system._get_stock_info(symbol)
        if not stock_info:
            print(f"  [오류] {symbol}: 종목 기본 정보 없음 (us_stock_basic)")
            return None
        sector = stock_info.get('sector')
        if not sector:
            print(f"  [오류] {symbol}: 섹터 정보 없음")
            return None
        if verbose:
            print(f"  섹터: {sector}")
            print(f"  시가총액: ${stock_info.get('market_cap', 0):,.0f}\n")

        # Step 3: 섹터 벤치마크 로드
        if verbose:
            print("Step 3: 섹터 벤치마크 로드 중...")
        benchmarks = await system._get_sector_benchmarks(sector)
        if verbose:
            print(f"  PE Median: {benchmarks.get('pe_median', 'N/A')}")
            print(f"  Forward PE Median: {benchmarks.get('forward_pe_median', 'N/A')}")
            print(f"  Gross Margin Median: {benchmarks.get('gross_margin_median', 'N/A')}\n")

        # Step 4: Factor 점수 계산
        if verbose:
            print("Step 4: Factor 점수 계산 중 (병렬 처리)...")

        value_task = calculate_value_score(symbol, db_manager, benchmarks, analysis_date)
        quality_task = calculate_quality_score(symbol, db_manager, benchmarks, analysis_date)
        momentum_task = calculate_momentum_score(symbol, db_manager, benchmarks, analysis_date)
        growth_task = calculate_growth_score(symbol, db_manager, benchmarks, analysis_date)

        results = await asyncio.gather(
            value_task, quality_task, momentum_task, growth_task,
            return_exceptions=True
        )

        # 결과 처리
        value_result = results[0] if not isinstance(results[0], Exception) else {'value_score': 50.0, 'strategies': {}}
        quality_result = results[1] if not isinstance(results[1], Exception) else {'quality_score': 50.0, 'strategies': {}}
        momentum_result = results[2] if not isinstance(results[2], Exception) else {'momentum_score': 50.0, 'strategies': {}}
        growth_result = results[3] if not isinstance(results[3], Exception) else {'growth_score': 50.0, 'strategies': {}}

        # 예외 로깅
        for i, (name, res) in enumerate([('Value', results[0]), ('Quality', results[1]),
                                          ('Momentum', results[2]), ('Growth', results[3])]):
            if isinstance(res, Exception):
                print(f"  [경고] {name} Factor 계산 실패: {res}")

        value_score = value_result['value_score']
        quality_score = quality_result['quality_score']
        momentum_score = momentum_result['momentum_score']
        growth_score = growth_result['growth_score']

        if verbose:
            print(f"  가치:    {value_score:.1f}")
            print(f"  품질:    {quality_score:.1f}")
            print(f"  모멘텀:  {momentum_score:.1f}")
            print(f"  성장:    {growth_score:.1f}\n")

        # Step 5: 최종 점수 계산 (Phase 3: Factor Interaction 적용)
        if verbose:
            print("Step 5: 최종 점수 계산 중...")

        # Base score (기존 가중합)
        base_score = (
            value_score * system.regime_weights['value'] +
            quality_score * system.regime_weights['quality'] +
            momentum_score * system.regime_weights['momentum'] +
            growth_score * system.regime_weights['growth']
        )

        # Factor Interaction 계산 (Phase 3)
        interaction_calc = USFactorInteractions({
            'value_score': value_score,
            'quality_score': quality_score,
            'momentum_score': momentum_score,
            'growth_score': growth_score
        })
        interaction_result = interaction_calc.calculate()
        interaction_score = interaction_result['interaction_score']
        conviction_score = interaction_result['conviction_score']

        # Final score = base 70% + interaction 30%
        total_score = base_score * 0.70 + interaction_score * 0.30

        if verbose:
            print(f"  Base Score 계산:")
            print(f"    {value_score:.1f} x {system.regime_weights['value']:.2f} = {value_score * system.regime_weights['value']:.2f}")
            print(f"    {quality_score:.1f} x {system.regime_weights['quality']:.2f} = {quality_score * system.regime_weights['quality']:.2f}")
            print(f"    {momentum_score:.1f} x {system.regime_weights['momentum']:.2f} = {momentum_score * system.regime_weights['momentum']:.2f}")
            print(f"    {growth_score:.1f} x {system.regime_weights['growth']:.2f} = {growth_score * system.regime_weights['growth']:.2f}")
            print(f"  Base Score: {base_score:.1f}")
            print(f"\n  Factor Interaction:")
            print(f"    I1 (Growth x Quality):   {interaction_result['i1_growth_quality']:.1f}")
            print(f"    I2 (Growth x Momentum):  {interaction_result['i2_growth_momentum']:.1f}")
            print(f"    I3 (Quality x Value):    {interaction_result['i3_quality_value']:.1f}")
            print(f"    I4 (Momentum x Quality): {interaction_result['i4_momentum_quality']:.1f}")
            print(f"    I5 (Conviction):         {conviction_score:.1f}")
            print(f"  Interaction Score: {interaction_score:.1f}")
            print(f"\n  Final Score = {base_score:.1f} x 0.70 + {interaction_score:.1f} x 0.30 = {total_score:.1f}\n")

        # Step 6: 등급 결정
        grade = system._determine_grade(total_score)
        opinion = GRADE_OPINION[grade]

        if verbose:
            print("Step 6: 등급 결정...")
            print(f"  등급: {grade} ({opinion})")
            print(f"  기준: A+(85+), A(75+), B+(65+), B(55+), C(45+), D(35+), F(<35)\n")

        # 결과 구성
        result = {
            'symbol': symbol,
            'sector': sector,
            'total_score': round(total_score, 1),
            'base_score': round(base_score, 1),
            'interaction_score': round(interaction_score, 1),
            'conviction_score': round(conviction_score, 1),
            'grade': grade,
            'opinion': opinion,
            'value_score': value_score,
            'quality_score': quality_score,
            'momentum_score': momentum_score,
            'growth_score': growth_score,
            'value_detail': value_result.get('strategies', {}),
            'quality_detail': quality_result.get('strategies', {}),
            'momentum_detail': momentum_result.get('strategies', {}),
            'growth_detail': growth_result.get('strategies', {}),
            'regime': system.current_regime,
            'regime_weights': system.regime_weights
        }

        # Step 7: DB 저장
        if verbose:
            print("Step 7: 데이터베이스 저장 중...")
        await system._save_stock_grade(result, analysis_date)
        if verbose:
            print(f"  저장 완료 (날짜: {analysis_date})\n")

        # 최종 요약
        if verbose:
            print("="*80)
            print("분석 요약")
            print("="*80)
            print(f"심볼: {symbol}")
            print(f"섹터: {sector}")
            print(f"시장 레짐: {system.current_regime}")
            print(f"\n팩터 점수:")
            print(f"  가치:       {value_score:>6.1f}")
            print(f"  품질:       {quality_score:>6.1f}")
            print(f"  모멘텀:     {momentum_score:>6.1f}")
            print(f"  성장:       {growth_score:>6.1f}")
            print(f"\nPhase 3 점수:")
            print(f"  Base Score:        {base_score:>6.1f}")
            print(f"  Interaction Score: {interaction_score:>6.1f}")
            print(f"  Conviction Score:  {conviction_score:>6.1f}")
            print(f"\n최종 결과:")
            print(f"  총점:    {total_score:>6.1f}")
            print(f"  등급:    {grade} ({opinion})")
            print("="*80 + "\n")

        return result

    except Exception as e:
        logger.error(f"{symbol} 분석 실패: {e}", exc_info=True)
        if verbose:
            print(f"\n[오류] {symbol} 분석 실패")
            print(f"이유: {str(e)}\n")
        return None


async def analyze_all_stocks_specific_dates(db_manager, date_list: List[date]) -> Dict:
    """전체 종목 특정 날짜 분석 - 디버깅 로그 포함"""

    print("\n" + "="*80)
    print("미국 주식 v2.0 - 전체 종목 특정 날짜 분석")
    print("="*80)
    print(f"분석 날짜: {len(date_list)}개")
    for idx, d in enumerate(date_list, 1):
        print(f"  {idx}. {d}")
    print(f"DB 커넥션 풀: min_size=10, max_size=30")
    print(f"배치 크기: 25개 종목 (동시 처리)")
    print("="*80 + "\n")

    # Suppress verbose logging
    original_levels = {}
    loggers_to_suppress = [
        'root', '__main__', 'us_main_v2',
        'us_market_regime', 'us_sector_benchmarks',
        'us_value_factor_v2', 'us_quality_factor_v2',
        'us_momentum_factor_v2', 'us_growth_factor_v2'
    ]

    for logger_name in loggers_to_suppress:
        target_logger = logging.getLogger(logger_name)
        original_levels[logger_name] = target_logger.level
        target_logger.setLevel(logging.ERROR)

    total_start_time = time.time()
    date_results = {}
    total_success = 0
    total_fail = 0
    total_processed = 0
    all_failed_symbols = []  # 전체 실패 종목 추적

    system = USQuantSystemV2(db_manager)

    for date_idx, analysis_date in enumerate(date_list, 1):
        date_start_time = time.time()

        print(f"[날짜 {date_idx}/{len(date_list)}] {analysis_date} - 분석 시작...")

        # 레짐 감지
        try:
            await system._detect_regime(analysis_date)
            print(f"  레짐: {system.current_regime}")
            print(f"  가중치: V={system.regime_weights['value']:.0%}, "
                  f"Q={system.regime_weights['quality']:.0%}, "
                  f"M={system.regime_weights['momentum']:.0%}, "
                  f"G={system.regime_weights['growth']:.0%}")
        except Exception as e:
            print(f"  [오류] 레짐 감지 실패: {e}")
            print(f"  기본 레짐(NEUTRAL) 사용")

        # 해당 날짜에 데이터가 있는 종목 조회
        symbol_query = """
        SELECT DISTINCT b.symbol
        FROM us_stock_basic b
        INNER JOIN us_daily d ON b.symbol = d.symbol
        WHERE d.date = $1
          AND b.sector IS NOT NULL
          AND b.sector != ''
        ORDER BY b.symbol
        """
        symbol_result = await db_manager.execute_query(symbol_query, analysis_date)
        symbols = [r['symbol'] for r in symbol_result] if symbol_result else []

        if not symbols:
            print(f"  [경고] {analysis_date}에 데이터가 있는 종목이 없습니다.\n")
            continue

        print(f"  -> {len(symbols):,}개 종목 발견")

        # 섹터 캐시 초기화
        system.sector_cache = {}

        # 배치 처리
        BATCH_SIZE = 25
        batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]

        date_success = 0
        date_fail = 0
        date_failed_symbols = []  # 날짜별 실패 종목

        for batch_idx, batch in enumerate(batches, 1):
            batch_start_time = time.time()
            tasks = []
            for symbol in batch:
                tasks.append(system._analyze_stock(symbol, analysis_date))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 저장
            batch_success = 0
            batch_fail = 0
            batch_errors = []

            for idx, result in enumerate(results):
                symbol = batch[idx]
                if isinstance(result, Exception):
                    date_fail += 1
                    batch_fail += 1
                    date_failed_symbols.append(symbol)
                    batch_errors.append(f"{symbol}: {type(result).__name__}")
                elif result:
                    try:
                        await system._save_stock_grade(result, analysis_date)
                        date_success += 1
                        batch_success += 1
                    except Exception as save_err:
                        date_fail += 1
                        batch_fail += 1
                        date_failed_symbols.append(symbol)
                        batch_errors.append(f"{symbol}: DB저장실패-{save_err}")
                else:
                    date_fail += 1
                    batch_fail += 1
                    date_failed_symbols.append(symbol)
                    batch_errors.append(f"{symbol}: 결과없음")

            # 진행 상황
            batch_time = time.time() - batch_start_time
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"  [{current_time}] 배치 {batch_idx}/{len(batches)} | "
                  f"성공: {batch_success} | 실패: {batch_fail} | "
                  f"시간: {batch_time:.1f}초")

            # 배치 내 에러 상세 (있는 경우만)
            if batch_errors and len(batch_errors) <= 5:
                for err in batch_errors:
                    print(f"    [실패] {err}")
            elif batch_errors:
                print(f"    [실패] {len(batch_errors)}개 종목 (상세 생략)")

        date_time = time.time() - date_start_time
        success_rate = (date_success / len(symbols) * 100) if symbols else 0

        print(f"\n[날짜 {date_idx}/{len(date_list)}] {analysis_date} 완료")
        print(f"  종목: {len(symbols):,}개 | 성공: {date_success:,} ({success_rate:.1f}%) | "
              f"실패: {date_fail:,} | 시간: {date_time:.1f}초")

        # 실패 종목 요약 (10개 이하면 전체, 초과면 앞 10개)
        if date_failed_symbols:
            if len(date_failed_symbols) <= 10:
                print(f"  실패 종목: {', '.join(date_failed_symbols)}")
            else:
                print(f"  실패 종목 (일부): {', '.join(date_failed_symbols[:10])}... 외 {len(date_failed_symbols)-10}개")
            all_failed_symbols.extend(date_failed_symbols)

        # RS Rank 배치 계산 (해당 날짜의 모든 종목 기준 상대 순위)
        print(f"  RS Rank 계산 중...")
        rs_rank_query = """
        WITH rs_ranked AS (
            SELECT
                symbol,
                rs_value,
                NTILE(100) OVER (ORDER BY rs_value DESC) as percentile
            FROM us_stock_grade
            WHERE date = $1 AND rs_value IS NOT NULL
        )
        UPDATE us_stock_grade g
        SET rs_rank = CASE
            WHEN r.percentile <= 10 THEN 'A+ (Top 10%)'
            WHEN r.percentile <= 20 THEN 'A (Top 20%)'
            WHEN r.percentile <= 40 THEN 'B (Top 40%)'
            WHEN r.percentile <= 60 THEN 'C (Top 60%)'
            WHEN r.percentile <= 80 THEN 'D (Top 80%)'
            ELSE 'F (Bottom 20%)'
        END
        FROM rs_ranked r
        WHERE g.symbol = r.symbol AND g.date = $1
        """
        try:
            await db_manager.execute_query(rs_rank_query, analysis_date)
            print(f"  RS Rank 계산 완료")
        except Exception as e:
            print(f"  [경고] RS Rank 계산 실패: {e}")
        print()

        date_results[str(analysis_date)] = {
            'success': date_success,
            'fail': date_fail,
            'total': len(symbols),
            'regime': system.current_regime,
            'failed_symbols': date_failed_symbols
        }

        total_success += date_success
        total_fail += date_fail
        total_processed += len(symbols)

    total_time = time.time() - total_start_time

    # 로그 레벨 복구
    for logger_name, level in original_levels.items():
        logging.getLogger(logger_name).setLevel(level)

    # 최종 요약
    print("\n" + "="*80)
    print("전체 종목 특정 날짜 분석 완료 (v2.0)")
    print("="*80)
    print(f"분석 날짜: {len(date_list)}개")
    print(f"총 분석 수: {total_processed:,}개")
    print(f"성공: {total_success:,}개 ({(total_success/total_processed*100) if total_processed > 0 else 0:.1f}%)")
    print(f"실패: {total_fail:,}개 ({(total_fail/total_processed*100) if total_processed > 0 else 0:.1f}%)")
    print(f"총 시간: {total_time/60:.1f}분 ({total_time:.0f}초)")
    if total_processed > 0:
        print(f"평균 속도: {total_processed/total_time:.1f} 종목/초")

    # 날짜별 요약
    print(f"\n날짜별 요약:")
    for d, r in date_results.items():
        rate = (r['success'] / r['total'] * 100) if r['total'] > 0 else 0
        print(f"  {d}: {r['success']:,}/{r['total']:,} ({rate:.1f}%) - {r['regime']}")

    # 전체 실패 종목 (중복 제거)
    if all_failed_symbols:
        unique_failed = list(set(all_failed_symbols))
        print(f"\n실패 종목 총 {len(unique_failed)}개 (고유)")
        if len(unique_failed) <= 20:
            print(f"  {', '.join(sorted(unique_failed))}")

    print("="*80 + "\n")

    return {
        'dates': len(date_list),
        'total_processed': total_processed,
        'total_success': total_success,
        'total_fail': total_fail,
        'total_time': total_time,
        'date_results': date_results,
        'all_failed_symbols': list(set(all_failed_symbols))
    }


def parse_dates(dates_input: str) -> List[date]:
    """날짜 문자열 파싱 (개별 날짜, 범위, 혼합 지원)"""

    segments = [seg.strip() for seg in dates_input.split(',')]
    date_list = []

    for segment in segments:
        if '~' in segment:
            # 날짜 범위 처리
            parts = segment.split('~')
            if len(parts) != 2:
                print(f"잘못된 범위 형식: {segment}")
                continue

            start_str = parts[0].strip()
            end_str = parts[1].strip()

            start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_str, '%Y-%m-%d').date()

            if start_date > end_date:
                print(f"시작 날짜가 종료 날짜보다 늦습니다: {segment}")
                continue

            current_date = start_date
            while current_date <= end_date:
                if current_date not in date_list:
                    date_list.append(current_date)
                current_date += timedelta(days=1)
        else:
            # 개별 날짜 처리
            parsed_date = datetime.strptime(segment, '%Y-%m-%d').date()
            if parsed_date not in date_list:
                date_list.append(parsed_date)

    date_list.sort()
    return date_list


# ========================================================================
# Main Entry Point
# ========================================================================

async def main():
    """메인 실행"""

    print("\n" + "="*80)
    print("미국 주식 퀀트 분석 시스템 v2.0")
    print("="*80 + "\n")

    # DB 초기화
    db_manager = AsyncDatabaseManager()
    await db_manager.initialize()

    try:
        while True:
            print("분석 옵션 선택:")
            print("  1. 개별 종목 분석")
            print("  2. 전체 종목 분석 (특정 날짜)")
            print()

            choice = input("선택 (1-2): ").strip()

            if choice == '1':
                # 옵션 1: 개별 종목 분석
                symbol = input("\n종목 심볼 입력 (예: AAPL): ").strip().upper()
                if not symbol:
                    print("심볼이 입력되지 않았습니다.\n")
                    continue

                print("\n분석 날짜 입력")
                print("사용법:")
                print("  - 개별 날짜: 2025-08-01, 2025-08-05")
                print("  - 날짜 범위: 2025-08-01~2025-08-05 (1일부터 5일까지)")
                print("  - 혼합: 2025-08-01, 2025-08-05~2025-08-08, 2025-09-04")
                print("  - Enter만 입력 시 최신 날짜로 분석")
                dates_input = input("\n날짜 입력: ").strip()

                # 빈 입력 시 최신 날짜
                if not dates_input:
                    await analyze_single_stock_standalone(symbol, db_manager)
                    print("\n분석 완료. 프로그램 종료합니다.")
                    break

                try:
                    date_list = parse_dates(dates_input)

                    if not date_list:
                        print("유효한 날짜가 없습니다.\n")
                        continue

                    print(f"\n종목: {symbol}")
                    print(f"분석 날짜: {len(date_list)}개")
                    for i, d in enumerate(date_list, 1):
                        print(f"  {i}. {d}")

                    confirm = input("\n분석을 시작하시겠습니까? (y/n): ").strip().lower()
                    if confirm != 'y':
                        print("취소되었습니다.\n")
                        continue

                    # 각 날짜별 분석 실행
                    print("\n" + "="*80)
                    print(f"{symbol} 종목 다중 날짜 분석 시작")
                    print("="*80)

                    success_count = 0
                    for idx, analysis_date in enumerate(date_list, 1):
                        print(f"\n[{idx}/{len(date_list)}] {analysis_date} 분석 중...")
                        print("-"*80)

                        try:
                            result = await analyze_single_stock_standalone(
                                symbol, db_manager, analysis_date, verbose=True
                            )
                            if result:
                                success_count += 1
                        except Exception as e:
                            print(f"  오류 발생: {str(e)}")

                    print("\n" + "="*80)
                    print(f"분석 완료: {success_count}/{len(date_list)}개 성공")
                    print("="*80)
                    print("\n프로그램 종료합니다.")
                    break

                except ValueError as e:
                    print(f"잘못된 날짜 형식입니다. YYYY-MM-DD 형식을 사용하세요. (오류: {str(e)})\n")
                    continue

            elif choice == '2':
                # 옵션 2: 전체 종목 분석 (특정 날짜)
                print("\n전체 종목 특정 날짜 분석 (v2.0)")
                print("분석할 날짜 입력")
                print("사용법:")
                print("  - 개별 날짜: 2025-08-01, 2025-08-05")
                print("  - 날짜 범위: 2025-08-01~2025-08-05 (1일부터 5일까지)")
                print("  - 혼합: 2025-08-01, 2025-08-05~2025-08-08, 2025-09-04")
                dates_input = input("\n날짜 입력: ").strip()

                if not dates_input:
                    print("날짜가 입력되지 않았습니다.\n")
                    continue

                try:
                    date_list = parse_dates(dates_input)

                    if not date_list:
                        print("유효한 날짜가 없습니다.\n")
                        continue

                    print(f"\n입력된 날짜: {len(date_list)}개")
                    for idx, d in enumerate(date_list, 1):
                        print(f"  {idx}. {d}")

                    confirm = input("\n분석을 시작하시겠습니까? (y/n): ").strip().lower()
                    if confirm == 'y':
                        # 커넥션 풀 재초기화
                        print("\n커넥션 풀 최적화 중...")
                        await db_manager.close()
                        db_manager = AsyncDatabaseManager()
                        await db_manager.initialize(min_size=10, max_size=30)

                        await analyze_all_stocks_specific_dates(db_manager, date_list)
                        print("\n프로그램 종료합니다.")
                        break
                    else:
                        print("취소되었습니다.\n")

                except ValueError as e:
                    print(f"잘못된 날짜 형식입니다. YYYY-MM-DD 형식을 사용하세요. (오류: {e})\n")
                except Exception as e:
                    print(f"오류 발생: {e}\n")

            else:
                print("잘못된 선택입니다. 다시 시도하세요.\n")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")

    except Exception as e:
        logger.error(f"오류 발생: {e}")
        print(f"\n오류 발생: {e}")

    finally:
        await db_manager.close()
        print("\n데이터베이스 연결을 종료했습니다.")


if __name__ == '__main__':
    asyncio.run(main())
