"""
US Quant System v2.0 - Main Integration Module

핵심 변경:
1. 시장 레짐 기반 동적 가중치
2. 섹터별 기준값 적용
3. 절대 점수 기반 등급 (섹터 상대 순위 아님)
4. 4개 Factor v2.0 통합

Usage:
    python us_main.py

File: us/us_main.py
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
from us_value_factor import calculate_value_score
from us_quality_factor import calculate_quality_score
from us_momentum_factor import calculate_momentum_score
from us_growth_factor import calculate_growth_score
from us_factor_interactions import USFactorInteractions
from us_agent_metrics import USAgentMetrics

# Phase 3.1 imports
from us_outlier_risk import USOutlierRisk
from us_exchange_optimizer import USExchangeOptimizer
from us_sector_dynamic_weights import USSectorDynamicWeights
from us_healthcare_optimizer import USHealthcareOptimizer
from us_nasdaq_optimizer import USNASDAQOptimizer
from us_event_engine import USEventEngine  # Phase 3.4 Event Engine
from us_volatility_adjustment import USVolatilityAdjustment  # Phase 3.4.2
from us_data_prefetcher import USDataPrefetcher  # Phase 3.4.3 Query Optimization
from us_alternative_matcher import USAlternativeStockMatcher  # Alternative stock matching

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
# 7단계 한글 등급 체계 (stock_agents/us와 통일)
GRADE_THRESHOLDS = {
    '강력 매수': 85,  # 85점 이상 = 최우수
    '매수':     75,  # 75-85점 = 우수
    '매수 고려': 65,  # 65-75점 = 양호
    '중립':     55,  # 55-65점 = 보통 이상
    '매도 고려': 45,  # 45-55점 = 보통
    '매도':     35,  # 35-45점 = 주의
    '강력 매도': 0    # 35점 미만 = 위험
}


class USQuantSystemV2:
    """US Quant System v3.0 (HMM Regime Detection Support)"""

    # 클래스 레벨 HMM 캐시 (모든 인스턴스 공유)
    _hmm_detector_cache = None
    _hmm_cache_initialized = False

    def __init__(
        self,
        db_manager: AsyncDatabaseManager,
        use_hmm: bool = False,
        use_static_weights: bool = False
    ):
        """
        Args:
            db_manager: AsyncDatabaseManager instance
            use_hmm: HMM 기반 레짐 감지 사용 (v2.0 학술 연구 기반)
            use_static_weights: 정적 가중치 사용 (동적 Macro 조정 비활성화)
        """
        self.db = db_manager
        self.sector_benchmarks = USSectorBenchmarks(db_manager)

        # Phase 3.1 modules
        self.outlier_risk = USOutlierRisk()
        self.exchange_optimizer = USExchangeOptimizer()
        self.sector_weights = USSectorDynamicWeights()
        self.healthcare_optimizer = USHealthcareOptimizer()
        self.nasdaq_optimizer = USNASDAQOptimizer()
        self.event_engine = USEventEngine(db_manager)  # Phase 3.4 Event Engine

        # Phase 3.5: HMM Regime Detection Options
        self.use_hmm = use_hmm
        self.use_static_weights = use_static_weights

        # Cache
        self.current_regime = None
        self.regime_weights = None
        self.regime_method = None  # 'HMM' or 'RULE_BASED'
        self.sector_cache = {}

    async def _init_hmm_detector(self):
        """
        HMM Detector 초기화 (시스템 레벨 캐싱)
        - 한 번만 학습하고 재사용
        - 모든 날짜에 동일한 학습된 모델 사용
        """
        if not USQuantSystemV2._hmm_cache_initialized and self.use_hmm:
            from us_market_regime import HMMRegimeDetector, HMM_AVAILABLE
            if HMM_AVAILABLE:
                logger.info("Initializing HMM Detector (one-time training)...")
                USQuantSystemV2._hmm_detector_cache = HMMRegimeDetector(n_regimes=3)
                await USQuantSystemV2._hmm_detector_cache.fit(self.db, lookback_days=504)
                USQuantSystemV2._hmm_cache_initialized = True
                logger.info("HMM Detector ready (cached for all dates)")

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
        """
        시장 레짐 감지 및 가중치 설정 (Phase 3.5: HMM 지원 + 캐싱)

        - use_hmm=True: HMM 기반 3-State 레짐 감지 (BULL/NEUTRAL/BEAR)
        - use_hmm=False: 기존 규칙 기반 5-State 레짐 감지
        - use_static_weights=True: 정적 가중치 (Factor Timing 최소화)

        효율성 최적화:
        - HMM 모델은 시스템 레벨에서 한 번만 학습
        - 여러 날짜 분석 시 동일한 학습된 모델 재사용
        """
        # HMM 모드: 캐시된 HMM Detector 사용
        if self.use_hmm and USQuantSystemV2._hmm_cache_initialized:
            regime_result = await self._detect_regime_with_cached_hmm(analysis_date)
        else:
            # 기존 방식 (규칙 기반 또는 HMM 미초기화 시)
            detector = USMarketRegimeDetector(
                self.db,
                analysis_date,
                use_hmm=self.use_hmm,
                use_static_weights=self.use_static_weights
            )
            regime_result = await detector.detect_regime()

            # 레짐 저장
            try:
                await detector.save_to_db(regime_result)
            except Exception as e:
                logger.warning(f"Regime save failed: {e}")

        self.current_regime = regime_result['regime']
        self.regime_weights = regime_result['weights']
        self.regime_data = regime_result
        self.macro_environment = regime_result.get('macro_environment', 'UNKNOWN')
        self.scenario_probability = regime_result.get('scenario_probability', {})
        self.regime_method = regime_result.get('method', 'UNKNOWN')

    async def _detect_regime_with_cached_hmm(self, analysis_date: date) -> Dict:
        """
        캐시된 HMM Detector를 사용한 레짐 감지 (효율성 최적화)
        - HMM 재학습 없이 예측만 수행
        - 기존 지표 조회는 별도 수행
        """
        from us_market_regime import REGIME_WEIGHTS_V2, MACRO_ENVIRONMENT

        # 캐시된 HMM으로 레짐 예측
        hmm_result = await USQuantSystemV2._hmm_detector_cache.predict_current_regime(
            self.db, analysis_date
        )

        regime = hmm_result['regime']
        weights = hmm_result['weights']

        # Macro Environment 매핑
        macro_env_map = {
            'BULL': 'SOFT_LANDING',
            'NEUTRAL': 'REFLATION',
            'BEAR': 'HARD_LANDING'
        }
        macro_env = macro_env_map.get(regime, 'SOFT_LANDING')

        # Scenario Probability
        transition_probs = hmm_result.get('transition_probs', {})
        scenario_prob = {
            '강세': transition_probs.get('BULL', 0.33),
            '횡보': transition_probs.get('NEUTRAL', 0.34),
            '약세': transition_probs.get('BEAR', 0.33)
        }

        return {
            'regime': regime,
            'weights': weights,
            'indicators': {},
            'description': REGIME_WEIGHTS_V2[regime]['description'],
            'composite_scores': {
                'liquidity_score': 50.0,
                'macro_score': 50.0,
                'volatility_score': 50.0,
                'risk_appetite_score': 50.0,
                'composite_score': 50.0
            },
            'macro_environment': macro_env,
            'macro_env_description': MACRO_ENVIRONMENT[macro_env]['description'],
            'scenario_probability': scenario_prob,
            'method': 'HMM_CACHED',
            'hmm_regime_id': hmm_result.get('regime_id', -1),
            'hmm_transition_probs': transition_probs
        }

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
        """개별 종목 분석 (Phase 3.1 통합, Phase 3.4.3 쿼리 최적화)"""

        # ============================================================
        # Phase 3.4.3: Prefetch all common data (5 parallel queries)
        # Reduces DB queries from ~35/stock to ~20/stock (43% reduction)
        # ============================================================
        prefetcher = USDataPrefetcher(self.db)
        prefetched = await prefetcher.prefetch_all(symbol, analysis_date)

        # Validate required data
        stock_basic = prefetched.get('stock_basic')
        price_data = prefetched.get('price_data')

        if not stock_basic:
            return None

        sector = stock_basic.get('sector')
        exchange = stock_basic.get('exchange', 'NYSE')
        if not sector:
            return None

        # Build stock_info from prefetched data
        stock_info = {
            'symbol': symbol,
            'stock_name': stock_basic.get('stock_name'),
            'sector': sector,
            'industry': stock_basic.get('industry'),
            'exchange': exchange,
            'market_cap': stock_basic.get('market_cap'),
            'beta': stock_basic.get('beta')
        }

        # 섹터 벤치마크 로드
        benchmarks = await self._get_sector_benchmarks(sector)

        # Phase 3.4.3: Create volatility engine from prefetched price data
        volatility_engine = self._create_volatility_engine_from_data(price_data)

        # ============================================================
        # Phase 3.4.3: Calculate factors with prefetched data
        # Factor modules will use prefetched_data instead of DB queries
        # ============================================================
        value_task = calculate_value_score(
            symbol, self.db, benchmarks, analysis_date,
            volatility_engine, prefetched_data=prefetched
        )
        quality_task = calculate_quality_score(
            symbol, self.db, benchmarks, analysis_date,
            volatility_engine, prefetched_data=prefetched
        )
        momentum_task = calculate_momentum_score(
            symbol, self.db, benchmarks, analysis_date, exchange,
            volatility_engine, prefetched_data=prefetched
        )
        growth_task = calculate_growth_score(
            symbol, self.db, benchmarks, analysis_date, exchange,
            volatility_engine, prefetched_data=prefetched
        )

        # Phase 3.4.3: Use _from_data methods for risk calculations
        var_cvar_result = self._calculate_var_cvar_from_data(price_data)
        rs_value = self._calculate_rs_value_from_data(price_data)
        ors_result = self._calculate_ors_from_data(price_data, stock_basic.get('market_cap'))
        risk_metrics = self._calculate_risk_metrics_from_data(price_data)

        # Event engine still needs individual queries
        event_task = self.event_engine.calculate_event_modifier(symbol, analysis_date)

        value_result, quality_result, momentum_result, growth_result, event_result = await asyncio.gather(
            value_task, quality_task, momentum_task, growth_task, event_task
        )

        # 개별 점수
        value_score = value_result['value_score']
        quality_score = quality_result['quality_score']
        momentum_score = momentum_result['momentum_score']
        growth_score = growth_result['growth_score']

        # Phase 3.1: ORS 결과
        outlier_risk_score = ors_result.get('ors', 0)
        risk_flag = ors_result.get('risk_flag', 'NORMAL')

        # Phase 3.3: Healthcare 최적화 (Growth-centric + Sub-sector differentiation)
        # Phase 3.2: Cash Runway 결과 초기화
        subsector = None
        cash_runway_result = None

        if sector == 'Healthcare':
            # Healthcare 서브섹터 분류
            subsector = await self.healthcare_optimizer.classify_healthcare_subsector(
                stock_info, symbol, self.db, analysis_date
            )

            # Healthcare 전용 가중치 적용 (0-1 범위 -> 0-100 스케일 변환)
            healthcare_weights = self.healthcare_optimizer.get_healthcare_weights(subsector)
            # healthcare_weights = {'growth': 0.60, 'quality': 0.25, ...} sum = 1.00
            adjusted_weights = {
                'growth': healthcare_weights['growth'] * 100,
                'momentum': healthcare_weights['momentum'] * 100,
                'quality': healthcare_weights['quality'] * 100,
                'value': healthcare_weights['value'] * 100
            }  # Sum = 100 (100-point scale maintained)

            # Validation: Weight sum must be 100
            weight_sum = sum(adjusted_weights.values())
            if abs(weight_sum - 100.0) > 0.01:
                logger.warning(
                    f"[Healthcare] Weight sum != 100: {weight_sum:.2f} for {symbol} "
                    f"(subsector: {subsector})"
                )

            # Phase 3.2: Cash Runway 평가 (Healthcare 섹터만)
            cash_runway_result = await self.healthcare_optimizer.evaluate_cash_runway_risk(
                symbol, self.db, analysis_date, subsector
            )

        # Phase 3.3: NASDAQ 최적화 (Momentum 축소 + Quality 강화 + 섹터별 차별화)
        elif exchange == 'NASDAQ':
            # NASDAQ 섹터별 가중치 적용
            # Technology/Healthcare는 Momentum 10%, 나머지는 기본 가중치
            nasdaq_weights = self.nasdaq_optimizer.get_nasdaq_sector_weights(sector)
            # nasdaq_weights = {'growth': 0.50, 'quality': 0.25, ...} sum = 1.00

            adjusted_weights = {
                'growth': nasdaq_weights['growth'] * 100,
                'momentum': nasdaq_weights['momentum'] * 100,
                'quality': nasdaq_weights['quality'] * 100,
                'value': nasdaq_weights['value'] * 100
            }  # Sum = 100 (100-point scale maintained)

            # Validation: Weight sum must be 100
            weight_sum = sum(adjusted_weights.values())
            if abs(weight_sum - 100.0) > 0.01:
                logger.warning(
                    f"[NASDAQ] Weight sum != 100: {weight_sum:.2f} for {symbol} "
                    f"(sector: {sector})"
                )

        else:
            # NYSE/기타 거래소: Phase 3.1 기존 로직
            base_weights = {
                'growth': self.regime_weights['growth'] * 100,
                'momentum': self.regime_weights['momentum'] * 100,
                'quality': self.regime_weights['quality'] * 100,
                'value': self.regime_weights['value'] * 100
            }
            adjusted_weights = self.sector_weights.get_combined_weights(base_weights, exchange, sector)

        # Base score (Phase 3.1: 조정된 가중치 사용)
        base_score = (
            growth_score * adjusted_weights['growth'] / 100 +
            momentum_score * adjusted_weights['momentum'] / 100 +
            quality_score * adjusted_weights['quality'] / 100 +
            value_score * adjusted_weights['value'] / 100
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

        # Phase 3.2: Cash Runway 패널티 적용 (Healthcare 섹터만)
        cash_runway_penalty = 0.0
        if cash_runway_result and cash_runway_result.get('penalty', 0) > 0:
            cash_runway_penalty = cash_runway_result['penalty']
            total_score = max(0, total_score - cash_runway_penalty)  # 최소 0점
            logger.info(
                f"[{symbol}] Cash Runway penalty applied: -{cash_runway_penalty:.1f} "
                f"({cash_runway_result.get('reason', 'N/A')})"
            )

        # Phase 3.4: Event Modifier 적용
        event_modifier = event_result.get('total_modifier', 0)
        if event_modifier != 0:
            total_score = max(0, min(100, total_score + event_modifier))  # 0-100 클램핑
            logger.debug(
                f"[{symbol}] Event modifier applied: {event_modifier:+.1f} "
                f"({event_result.get('reason', 'N/A')})"
            )

        # 등급 결정 (절대 점수 기반, 7단계 한글)
        grade = self._determine_grade(total_score)
        opinion = grade  # grade 자체가 한글 투자의견

        # Phase 3.4.3: confidence_score (factor alignment - lower std = higher confidence)
        factor_scores = [value_score, quality_score, momentum_score, growth_score]
        factor_std = np.std(factor_scores)
        confidence_score = round(max(0, 100 - (factor_std / 35 * 100)), 1)

        # Phase 3.4.3: factor_combination_bonus
        factor_combination_bonus = self._calculate_factor_combination_bonus(
            value_score, quality_score, momentum_score, growth_score
        )

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

        # Phase 3.4.3 Stage 3: Factor Momentum (4 columns)
        factor_momentum_result = await self._calculate_factor_momentum(
            symbol, analysis_date,
            {
                'value': value_score,
                'quality': quality_score,
                'momentum': momentum_score,
                'growth': growth_score
            }
        )

        # Phase 3.4.3 Stage 4: Sector Metrics (4 columns)
        sector_metrics_result = await self._calculate_sector_metrics(sector, analysis_date)

        # Phase 3.4.3 Stage 4: Text Generation (5 columns)
        text_metrics = {
            'volatility_annual': risk_metrics.get('volatility_annual'),
            'max_drawdown_1y': risk_metrics.get('max_drawdown_1y'),
            'beta': float(stock_info['beta']) if stock_info.get('beta') else None,
            'total_score': total_score,
            'score_trend_2w': agent_result.get('score_trend_2w'),
            'regime': self.current_regime
        }
        text_result = self._generate_analysis_texts(text_metrics)

        return {
            'symbol': symbol,
            'sector': sector,
            'exchange': exchange,  # Phase 3.1
            # Phase 3.4.3: stock_name, beta, industry, confidence_score, factor_combination_bonus
            'stock_name': stock_info.get('stock_name'),
            'beta': float(stock_info['beta']) if stock_info.get('beta') else None,
            'industry': stock_info.get('industry'),
            'confidence_score': confidence_score,
            'factor_combination_bonus': factor_combination_bonus,
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
            # Phase 3.1: ORS and Adjusted Weights
            'outlier_risk_score': outlier_risk_score,
            'risk_flag': risk_flag,
            'weight_growth': round(adjusted_weights['growth'], 1),
            'weight_momentum': round(adjusted_weights['momentum'], 1),
            'weight_quality': round(adjusted_weights['quality'], 1),
            'weight_value': round(adjusted_weights['value'], 1),
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
            'scenario_sample_count': agent_result.get('scenario_sample_count'),
            # Phase 3.2: Cash Runway (Healthcare/Biotech only)
            'healthcare_subsector': subsector,
            'cash_runway_months': cash_runway_result.get('runway_months') if cash_runway_result else None,
            'cash_runway_risk': cash_runway_result.get('risk_level') if cash_runway_result else None,
            'cash_runway_penalty': cash_runway_penalty,
            'is_cash_burning': cash_runway_result.get('is_burning') if cash_runway_result else None,
            # Phase 3.4: Event Engine
            'event_modifier': event_modifier,
            'earnings_modifier': event_result.get('earnings_modifier', 0),
            'options_modifier': event_result.get('options_modifier', 0),
            'insider_modifier': event_result.get('insider_modifier', 0),
            'event_reason': event_result.get('reason', 'N/A'),
            # Phase 3.4.3: Risk Metrics from prefetched price data
            'volatility_annual': risk_metrics.get('volatility_annual'),
            'max_drawdown_1y': risk_metrics.get('max_drawdown_1y'),
            'sharpe_ratio': risk_metrics.get('sharpe_ratio'),
            'sortino_ratio': risk_metrics.get('sortino_ratio'),
            'calmar_ratio': risk_metrics.get('calmar_ratio'),
            # Phase 3.4.3 Stage 3: Factor Momentum (4 columns)
            'value_momentum': factor_momentum_result.get('value_momentum'),
            'quality_momentum': factor_momentum_result.get('quality_momentum'),
            'momentum_momentum': factor_momentum_result.get('momentum_momentum'),
            'growth_momentum': factor_momentum_result.get('growth_momentum'),
            # Phase 3.4.3 Stage 4: Sector Metrics (4 columns)
            'sector_rotation_score': sector_metrics_result.get('sector_rotation_score'),
            'sector_momentum': sector_metrics_result.get('sector_momentum'),
            'sector_rank': sector_metrics_result.get('sector_rank'),
            'sector_percentile': sector_metrics_result.get('sector_percentile'),
            # Phase 3.4.3 Stage 4: Text Generation (5 columns)
            'risk_profile_text': text_result.get('risk_profile_text'),
            'risk_recommendation': text_result.get('risk_recommendation'),
            'time_series_text': text_result.get('time_series_text'),
            'signal_overall': text_result.get('signal_overall'),
            'market_state': text_result.get('market_state')
        }

    async def _get_stock_info(self, symbol: str) -> Optional[Dict]:
        """종목 기본 정보 조회 (Phase 3.1: exchange 추가, Phase 3.4.3: stock_name, beta, industry 추가)"""

        query = """
        SELECT symbol, sector, market_cap, exchange, stock_name, beta, industry
        FROM us_stock_basic
        WHERE symbol = $1
        """

        result = await self.db.execute_query(query, symbol)
        return dict(result[0]) if result else None

    async def _create_volatility_engine(self, symbol: str, analysis_date: date) -> Optional[USVolatilityAdjustment]:
        """
        Phase 3.4.2: Create shared volatility engine for all factors

        Loads price data once and creates a USVolatilityAdjustment instance
        that can be shared across all factor calculations.
        """
        try:
            # Load price data for HV calculation
            query = """
            SELECT date, close
            FROM us_daily
            WHERE symbol = $1
              AND ($2::DATE IS NULL OR date <= $2::DATE)
            ORDER BY date DESC
            LIMIT 30
            """

            result = await self.db.execute_query(query, symbol, analysis_date)

            if not result:
                return None

            price_data = [
                {'date': r['date'], 'close': float(r['close']) if r['close'] else None}
                for r in result
            ]

            # Create and initialize volatility engine
            vol_engine = USVolatilityAdjustment(self.db)
            await vol_engine.load_data(symbol, analysis_date, price_data)

            return vol_engine

        except Exception as e:
            logger.warning(f"{symbol}: Failed to create volatility engine - {e}")
            return None

    def _determine_grade(self, score: float) -> str:
        """절대 점수 기반 등급 결정 (7단계 한글)"""

        if score >= GRADE_THRESHOLDS['강력 매수']:
            return '강력 매수'
        elif score >= GRADE_THRESHOLDS['매수']:
            return '매수'
        elif score >= GRADE_THRESHOLDS['매수 고려']:
            return '매수 고려'
        elif score >= GRADE_THRESHOLDS['중립']:
            return '중립'
        elif score >= GRADE_THRESHOLDS['매도 고려']:
            return '매도 고려'
        elif score >= GRADE_THRESHOLDS['매도']:
            return '매도'
        else:
            return '강력 매도'

    def _calculate_factor_combination_bonus(self, value: float, quality: float,
                                             momentum: float, growth: float) -> int:
        """
        Phase 3.4.3: Calculate factor combination bonus/penalty

        Proven factor patterns from academic research:
        - Quality Compounder: High Q + High G = sustained growth
        - Deep Value Turnaround: High V + High Q + Low M = contrarian opportunity
        - GARP: High G + Reasonable V = growth at reasonable price
        - Momentum Speculation: Low V + High M = speculative risk

        Returns:
            int: Bonus/penalty in range [-20, +30]
        """
        bonus = 0

        # Pattern 1: Quality Compounder (Q>=70, G>=70) -> +15
        if quality >= 70 and growth >= 70:
            bonus += 15

        # Pattern 2: Deep Value Turnaround (V>=75, Q>=60, M<40) -> +20
        if value >= 75 and quality >= 60 and momentum < 40:
            bonus += 20

        # Pattern 3: GARP - Growth at Reasonable Price (G>=70, V>=50) -> +10
        if growth >= 70 and value >= 50:
            bonus += 10

        # Pattern 4: Momentum Speculation Penalty (V<30, M>=80) -> -10
        if value < 30 and momentum >= 80:
            bonus -= 10

        # Pattern 5: All-around Strong (all >= 65) -> +15
        if all(s >= 65 for s in [value, quality, momentum, growth]):
            bonus += 15

        # Clamp to [-20, +30]
        return max(-20, min(30, bonus))

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

    async def _calculate_ors(self, symbol: str, analysis_date: date) -> Dict:
        """
        Phase 3.1: Outlier Risk Score (ORS) 계산
        - price, avg_volume, volatility_252d, market_cap 기반
        - 점수 범위: 0-100
        """
        query = """
        SELECT
            b.market_cap,
            d.close as price,
            d.volume
        FROM us_stock_basic b
        LEFT JOIN us_daily d ON b.symbol = d.symbol AND d.date = $2
        WHERE b.symbol = $1
        """
        rows = await self.db.execute_query(query, symbol, analysis_date)

        if not rows or not rows[0].get('price'):
            return {'ors': 0, 'risk_flag': 'NORMAL'}

        r = rows[0]
        price = float(r['price']) if r['price'] else 0
        market_cap = float(r['market_cap']) if r['market_cap'] else 0

        # 평균 거래량 계산 (최근 20일)
        vol_query = """
        SELECT AVG(volume) as avg_volume
        FROM (
            SELECT volume
            FROM us_daily
            WHERE symbol = $1 AND date <= $2
            ORDER BY date DESC
            LIMIT 20
        ) sub
        """
        vol_rows = await self.db.execute_query(vol_query, symbol, analysis_date)
        avg_volume = float(vol_rows[0]['avg_volume']) if vol_rows and vol_rows[0].get('avg_volume') else 0

        # 변동성 계산 (연환산)
        vol_price_query = """
        SELECT close
        FROM us_daily
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT 253
        """
        vol_price_rows = await self.db.execute_query(vol_price_query, symbol, analysis_date)

        volatility = 0
        if vol_price_rows and len(vol_price_rows) > 20:
            prices = [float(r['close']) for r in vol_price_rows if r['close']]
            if len(prices) > 1:
                returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]
                import statistics
                volatility = statistics.stdev(returns) * (252 ** 0.5) * 100 if returns else 0

        # ORS 계산 (USOutlierRisk 모듈 사용)
        ors_result = self.outlier_risk.calculate_ors_single(
            price=price,
            avg_volume=avg_volume,
            volatility=volatility,
            market_cap=market_cap
        )

        return ors_result

    # ========================================================================
    # Phase 3.4.3: Prefetched Data Methods (Query Optimization)
    # ========================================================================

    def _calculate_var_cvar_from_data(self, price_data: Dict) -> Dict:
        """
        Calculate VaR 95% and CVaR 95% from prefetched price data.
        No DB query required.

        Args:
            price_data: Prefetched price data from USDataPrefetcher

        Returns:
            {'var_95': float, 'cvar_95': float}
        """
        if not price_data or 'closes' not in price_data:
            return {'var_95': None, 'cvar_95': None}

        closes = price_data['closes'][:253]  # Up to 252 trading days
        closes = [c for c in closes if c is not None]

        if len(closes) < 30:
            return {'var_95': None, 'cvar_95': None}

        # Calculate returns (oldest first for consistency)
        prices = closes[::-1]
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] * 100
                   for i in range(1, len(prices))]

        var_95 = float(np.percentile(returns, 5))  # 5th percentile (worst 5%)
        cvar_returns = [r for r in returns if r <= var_95]
        cvar_95 = float(np.mean(cvar_returns)) if cvar_returns else var_95

        return {
            'var_95': round(abs(var_95), 2),
            'cvar_95': round(abs(cvar_95), 2)
        }

    def _calculate_rs_value_from_data(self, price_data: Dict) -> Optional[float]:
        """
        Calculate IBD-style Relative Strength Value from prefetched price data.
        RS = 3M return * 40% + 6M return * 20% + 9M return * 20% + 12M return * 20%

        Args:
            price_data: Prefetched price data from USDataPrefetcher

        Returns:
            RS value or None
        """
        if not price_data or 'closes' not in price_data:
            return None

        closes = price_data['closes']
        if not closes or len(closes) < 253:
            return None

        current_price = closes[0]
        if current_price is None:
            return None

        def safe_return(days: int) -> float:
            if len(closes) > days and closes[days] is not None and closes[days] > 0:
                return (current_price / closes[days] - 1) * 100
            return 0

        ret_3m = safe_return(63)
        ret_6m = safe_return(126)
        ret_9m = safe_return(189)
        ret_12m = safe_return(252)

        rs_value = ret_3m * 0.4 + ret_6m * 0.2 + ret_9m * 0.2 + ret_12m * 0.2
        return round(rs_value, 2)

    def _calculate_ors_from_data(self, price_data: Dict, market_cap: float) -> Dict:
        """
        Calculate Outlier Risk Score from prefetched data.
        No DB query required.

        Args:
            price_data: Prefetched price data from USDataPrefetcher
            market_cap: Market cap from stock_basic

        Returns:
            {'ors': float, 'risk_flag': str}
        """
        if not price_data or 'closes' not in price_data:
            return {'ors': 0, 'risk_flag': 'NORMAL'}

        closes = price_data['closes']
        volumes = price_data.get('volumes', [])

        if not closes or closes[0] is None:
            return {'ors': 0, 'risk_flag': 'NORMAL'}

        price = closes[0]

        # Average volume (20 days)
        vol_20d = [v for v in volumes[:20] if v is not None]
        avg_volume = sum(vol_20d) / len(vol_20d) if vol_20d else 0

        # Volatility (annualized, 252 days)
        volatility = 0
        closes_clean = [c for c in closes[:253] if c is not None]
        if len(closes_clean) > 20:
            returns = [(closes_clean[i] - closes_clean[i + 1]) / closes_clean[i + 1]
                       for i in range(len(closes_clean) - 1)]
            if returns:
                import statistics
                volatility = statistics.stdev(returns) * (252 ** 0.5) * 100

        # Use outlier_risk module
        return self.outlier_risk.calculate_ors_single(
            price=price,
            avg_volume=avg_volume,
            volatility=volatility,
            market_cap=float(market_cap) if market_cap else 0
        )

    def _calculate_risk_metrics_from_data(self, price_data: Dict) -> Dict:
        """
        Calculate risk metrics from prefetched price data.
        Phase 3.4.3: volatility_annual, max_drawdown_1y, sharpe/sortino/calmar_ratio

        Args:
            price_data: Prefetched price data from USDataPrefetcher

        Returns:
            {
                'volatility_annual': float,  # Annualized volatility (60-day)
                'max_drawdown_1y': float,    # Maximum drawdown (252-day)
                'sharpe_ratio': float,       # Risk-adjusted return
                'sortino_ratio': float,      # Downside risk-adjusted return
                'calmar_ratio': float        # Return / Max Drawdown
            }
        """
        null_result = {
            'volatility_annual': None,
            'max_drawdown_1y': None,
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'calmar_ratio': None
        }

        if not price_data or 'closes' not in price_data:
            return null_result

        closes = price_data['closes']
        closes = [c for c in closes if c is not None]

        if len(closes) < 60:
            return null_result

        # Calculate returns (60 days for volatility)
        returns_60d = [(closes[i] - closes[i + 1]) / closes[i + 1]
                       for i in range(min(60, len(closes) - 1))]

        if not returns_60d:
            return null_result

        # 1. Volatility (annualized, 60-day)
        volatility_annual = np.std(returns_60d) * np.sqrt(252) * 100

        # 2. Max Drawdown (1 year)
        closes_1y = closes[:min(252, len(closes))][::-1]  # oldest first
        running_max = closes_1y[0]
        max_drawdown = 0
        for price in closes_1y:
            running_max = max(running_max, price)
            drawdown = (price - running_max) / running_max * 100
            max_drawdown = min(max_drawdown, drawdown)

        # 3. Risk-Free Rate (approximate 4.5% annual = 0.0178% daily)
        rf_daily = 0.045 / 252

        # 4. Sharpe Ratio (annualized)
        avg_return = np.mean(returns_60d)
        std_return = np.std(returns_60d)
        sharpe_ratio = None
        if std_return > 0:
            sharpe_ratio = (avg_return - rf_daily) / std_return * np.sqrt(252)

        # 5. Sortino Ratio (downside deviation)
        downside_returns = [r for r in returns_60d if r < 0]
        sortino_ratio = None
        if downside_returns:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                sortino_ratio = (avg_return - rf_daily) / downside_std * np.sqrt(252)

        # 6. Calmar Ratio (annual return / max drawdown)
        calmar_ratio = None
        annual_return = avg_return * 252 * 100
        if max_drawdown < 0:
            calmar_ratio = annual_return / abs(max_drawdown)

        return {
            'volatility_annual': round(volatility_annual, 2),
            'max_drawdown_1y': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
            'sortino_ratio': round(sortino_ratio, 4) if sortino_ratio is not None else None,
            'calmar_ratio': round(calmar_ratio, 4) if calmar_ratio is not None else None
        }

    def _create_volatility_engine_from_data(self, price_data: Dict) -> Optional[USVolatilityAdjustment]:
        """
        Create shared volatility engine from prefetched price data.
        No DB query required.

        Args:
            price_data: Prefetched price data from USDataPrefetcher

        Returns:
            USVolatilityAdjustment instance or None
        """
        if not price_data or 'raw' not in price_data:
            return None

        try:
            # Extract price data for volatility engine (30 days)
            raw_data = price_data['raw'][:30]
            engine_data = [
                {'date': r['date'], 'close': float(r['close']) if r.get('close') else None}
                for r in raw_data
            ]

            vol_engine = USVolatilityAdjustment(self.db)
            # Note: load_data_sync is a synchronous method that accepts pre-loaded data
            vol_engine.price_data = engine_data
            vol_engine.is_loaded = True

            return vol_engine

        except Exception as e:
            logger.warning(f"Failed to create volatility engine from data: {e}")
            return None

    # ========================================================================
    # Phase 3.4.3 Stage 3: Factor Momentum & Sector Metrics
    # ========================================================================

    async def _calculate_factor_momentum(self, symbol: str, analysis_date: date,
                                          current_scores: Dict) -> Dict[str, Optional[float]]:
        """
        Calculate factor score momentum (change vs previous month).
        Phase 3.4.3 Stage 3: 4 columns

        Args:
            symbol: Stock symbol
            analysis_date: Current analysis date
            current_scores: Dict with 'value', 'quality', 'momentum', 'growth' keys

        Returns:
            {
                'value_momentum': float,      # DECIMAL(5,1)
                'quality_momentum': float,
                'momentum_momentum': float,
                'growth_momentum': float
            }
        """
        null_result = {
            'value_momentum': None,
            'quality_momentum': None,
            'momentum_momentum': None,
            'growth_momentum': None
        }

        try:
            # Get scores from ~30 days ago
            prev_date = analysis_date - timedelta(days=30)
            query = """
                SELECT value_score, quality_score, momentum_score, growth_score
                FROM us_stock_grade
                WHERE symbol = $1 AND date <= $2
                ORDER BY date DESC LIMIT 1
            """
            result = await self.db.execute_query(query, symbol, prev_date)

            if not result:
                return null_result

            prev_row = result[0]

            # Calculate momentum (current - previous)
            def calc_momentum(current: float, prev_val) -> Optional[float]:
                if prev_val is None:
                    return None
                return round(current - float(prev_val), 1)

            return {
                'value_momentum': calc_momentum(current_scores['value'], prev_row.get('value_score')),
                'quality_momentum': calc_momentum(current_scores['quality'], prev_row.get('quality_score')),
                'momentum_momentum': calc_momentum(current_scores['momentum'], prev_row.get('momentum_score')),
                'growth_momentum': calc_momentum(current_scores['growth'], prev_row.get('growth_score'))
            }

        except Exception as e:
            logger.debug(f"{symbol}: Factor momentum calculation failed - {e}")
            return null_result

    async def _calculate_sector_metrics(self, sector: str, analysis_date: date) -> Dict:
        """
        Calculate sector-level metrics from materialized view.
        Phase 3.4.3 Stage 4: 4 columns

        Args:
            sector: Sector name (e.g., 'Technology')
            analysis_date: Analysis date

        Returns:
            {
                'sector_rotation_score': int,  # 0-100
                'sector_momentum': float,      # DECIMAL(5,2)
                'sector_rank': int,
                'sector_percentile': float     # DECIMAL(5,2)
            }
        """
        null_result = {
            'sector_rotation_score': None,
            'sector_momentum': None,
            'sector_rank': None,
            'sector_percentile': None
        }

        if not sector:
            return null_result

        try:
            # Query mv_us_sector_daily_performance
            query = """
                SELECT
                    avg_return_30d as momentum,
                    sector_rank,
                    stock_count,
                    (SELECT COUNT(DISTINCT sector_code)
                     FROM mv_us_sector_daily_performance
                     WHERE date = $1) as total_sectors
                FROM mv_us_sector_daily_performance
                WHERE date = $1 AND sector_code = $2
            """
            result = await self.db.execute_query(query, analysis_date, sector)

            if not result:
                # Try previous trading days if no data for exact date
                for days_back in [1, 2, 3]:
                    fallback_date = analysis_date - timedelta(days=days_back)
                    result = await self.db.execute_query(query, fallback_date, sector)
                    if result:
                        break

            if not result:
                return null_result

            row = result[0]
            momentum = float(row['momentum']) if row.get('momentum') is not None else 0
            sector_rank = int(row['sector_rank']) if row.get('sector_rank') else None
            total_sectors = int(row['total_sectors']) if row.get('total_sectors') else 11

            # Calculate percentile (lower rank = better = lower percentile)
            sector_percentile = None
            if sector_rank and total_sectors > 0:
                sector_percentile = round((sector_rank / total_sectors) * 100, 2)

            # Calculate rotation score (0-100)
            # Higher momentum = higher score, lower rank = higher score
            rotation_score = None
            if sector_rank is not None:
                # Momentum contribution: +/-30 points based on momentum
                momentum_component = min(30, max(-30, momentum * 3))
                # Rank contribution: 70 points for rank 1, decreasing
                rank_component = max(0, 70 - (sector_rank - 1) * 7)
                rotation_score = int(min(100, max(0, 50 + momentum_component + (rank_component - 35))))

            return {
                'sector_rotation_score': rotation_score,
                'sector_momentum': round(momentum, 2),
                'sector_rank': sector_rank,
                'sector_percentile': sector_percentile
            }

        except Exception as e:
            logger.debug(f"Sector metrics calculation failed for {sector}: {e}")
            return null_result

    def _generate_analysis_texts(self, metrics: Dict) -> Dict[str, str]:
        """
        Generate human-readable analysis text.
        Phase 3.4.3 Stage 4: 5 columns

        Args:
            metrics: Dict containing all calculated metrics

        Returns:
            {
                'risk_profile_text': str,
                'risk_recommendation': str,
                'time_series_text': str,
                'signal_overall': str,
                'market_state': str
            }
        """
        # Extract metrics with defaults
        vol = metrics.get('volatility_annual') or 30
        mdd = metrics.get('max_drawdown_1y') or -15
        beta = metrics.get('beta') or 1.0
        final_score = metrics.get('total_score') or 50
        score_trend = metrics.get('score_trend_2w') or 0
        regime = metrics.get('regime') or 'NEUTRAL'
        vix_level = 20  # Default VIX estimate

        # 1. risk_profile_text
        if vol > 50 or mdd < -40 or (beta and beta > 1.8):
            risk_profile = "High Risk - Elevated volatility with significant drawdown potential"
        elif vol > 35 or mdd < -25 or (beta and beta > 1.3):
            risk_profile = "Moderate-High Risk - Above average market volatility exposure"
        elif vol > 25 or mdd < -15:
            risk_profile = "Moderate Risk - Average market volatility exposure"
        else:
            risk_profile = "Low Risk - Below average volatility and controlled drawdown"

        # 2. risk_recommendation
        if vol > 45:
            risk_rec = "Suitable for aggressive investors with high risk tolerance and long time horizon"
        elif vol > 30:
            risk_rec = "Suitable for balanced investors seeking growth with moderate risk acceptance"
        else:
            risk_rec = "Suitable for conservative investors prioritizing capital preservation"

        # 3. time_series_text
        if score_trend and score_trend > 5:
            time_series = "Improving - Score trending up over past 2 weeks"
        elif score_trend and score_trend < -5:
            time_series = "Declining - Score trending down over past 2 weeks"
        else:
            time_series = "Stable - Score relatively unchanged over past 2 weeks"

        # 4. signal_overall
        if final_score >= 80:
            signal = "Strong Buy - Exceptional multi-factor alignment across all dimensions"
        elif final_score >= 70:
            signal = "Buy - Strong fundamentals with favorable risk-reward profile"
        elif final_score >= 60:
            signal = "Moderate Buy - Above average metrics with some positive catalysts"
        elif final_score >= 50:
            signal = "Hold - Neutral outlook with balanced risk-reward"
        elif final_score >= 40:
            signal = "Reduce - Below average fundamentals, consider position sizing"
        elif final_score >= 30:
            signal = "Sell - Weak metrics suggest downside risk"
        else:
            signal = "Strong Sell - Significant fundamental deterioration"

        # 5. market_state (regime + context)
        regime_map = {
            'BULL': 'Bullish',
            'NEUTRAL': 'Neutral',
            'BEAR': 'Bearish',
            'EXPANSION': 'Expansion',
            'CONTRACTION': 'Contraction'
        }
        regime_text = regime_map.get(regime, regime)
        market_state = f"{regime_text}-Market"

        return {
            'risk_profile_text': risk_profile,
            'risk_recommendation': risk_rec,
            'time_series_text': time_series,
            'signal_overall': signal,
            'market_state': market_state
        }

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
            date, symbol, stock_name, beta, confidence_score, factor_combination_bonus,
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
            iv_percentile, insider_signal,
            outlier_risk_score, risk_flag,
            weight_growth, weight_momentum, weight_quality, weight_value,
            volatility_annual, max_drawdown_1y, sharpe_ratio, sortino_ratio, calmar_ratio,
            value_momentum, quality_momentum, momentum_momentum, growth_momentum,
            sector_rotation_score, sector_momentum, sector_rank, sector_percentile,
            risk_profile_text, risk_recommendation, time_series_text, signal_overall, market_state
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8,
            $9, $10, $11, $12,
            $13::jsonb, $14::jsonb, $15::jsonb, $16::jsonb,
            $17, $18, $19,
            $20, $21,
            $22, $23, $24,
            $25, $26, $27, $28, $29,
            $30, $31, $32,
            $33, $34, $35,
            $36,
            $37::jsonb, $38::jsonb, $39::jsonb,
            $40, $41,
            $42, $43,
            $44, $45, $46, $47,
            $48, $49, $50, $51, $52,
            $53, $54, $55, $56,
            $57, $58, $59, $60,
            $61, $62, $63, $64, $65
        )
        ON CONFLICT (date, symbol) DO UPDATE SET
            stock_name = EXCLUDED.stock_name,
            beta = EXCLUDED.beta,
            confidence_score = EXCLUDED.confidence_score,
            factor_combination_bonus = EXCLUDED.factor_combination_bonus,
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
            outlier_risk_score = EXCLUDED.outlier_risk_score,
            risk_flag = EXCLUDED.risk_flag,
            weight_growth = EXCLUDED.weight_growth,
            weight_momentum = EXCLUDED.weight_momentum,
            weight_quality = EXCLUDED.weight_quality,
            weight_value = EXCLUDED.weight_value,
            volatility_annual = EXCLUDED.volatility_annual,
            max_drawdown_1y = EXCLUDED.max_drawdown_1y,
            sharpe_ratio = EXCLUDED.sharpe_ratio,
            sortino_ratio = EXCLUDED.sortino_ratio,
            calmar_ratio = EXCLUDED.calmar_ratio,
            value_momentum = EXCLUDED.value_momentum,
            quality_momentum = EXCLUDED.quality_momentum,
            momentum_momentum = EXCLUDED.momentum_momentum,
            growth_momentum = EXCLUDED.growth_momentum,
            sector_rotation_score = EXCLUDED.sector_rotation_score,
            sector_momentum = EXCLUDED.sector_momentum,
            sector_rank = EXCLUDED.sector_rank,
            sector_percentile = EXCLUDED.sector_percentile,
            risk_profile_text = EXCLUDED.risk_profile_text,
            risk_recommendation = EXCLUDED.risk_recommendation,
            time_series_text = EXCLUDED.time_series_text,
            signal_overall = EXCLUDED.signal_overall,
            market_state = EXCLUDED.market_state,
            updated_at = CURRENT_TIMESTAMP
        """

        try:
            await self.db.execute_query(
                query,
                analysis_date,                                      # $1
                result['symbol'],                                   # $2
                result.get('stock_name'),                           # $3 (Phase 3.4.3)
                result.get('beta'),                                 # $4 (Phase 3.4.3)
                result.get('confidence_score'),                     # $5 (Phase 3.4.3)
                result.get('factor_combination_bonus'),             # $6 (Phase 3.4.3)
                result['total_score'],                              # $7
                result['grade'],                                    # $8
                result['value_score'],                              # $9
                result['quality_score'],                            # $10
                result['momentum_score'],                           # $11
                result['growth_score'],                             # $12
                serialize_detail(result['value_detail']),           # $13
                serialize_detail(result['quality_detail']),         # $14
                serialize_detail(result['momentum_detail']),        # $15
                serialize_detail(result['growth_detail']),          # $16
                result.get('var_95'),                               # $17
                result.get('cvar_95'),                              # $18
                result.get('rs_value'),                             # $19
                result.get('interaction_score'),                    # $20
                result.get('conviction_score'),                     # $21
                result.get('entry_timing_score'),                   # $22
                result.get('score_trend_2w'),                       # $23
                result.get('price_position_52w'),                   # $24
                result.get('atr_pct'),                              # $25
                result.get('stop_loss_pct'),                        # $26
                result.get('take_profit_pct'),                      # $27
                result.get('risk_reward_ratio'),                    # $28
                result.get('position_size_pct'),                    # $29
                result.get('scenario_bullish_prob'),                # $30
                result.get('scenario_sideways_prob'),               # $31
                result.get('scenario_bearish_prob'),                # $32
                result.get('scenario_bullish_return'),              # $33
                result.get('scenario_sideways_return'),             # $34
                result.get('scenario_bearish_return'),              # $35
                result.get('scenario_sample_count'),                # $36
                serialize_triggers(result.get('buy_triggers')),     # $37
                serialize_triggers(result.get('sell_triggers')),    # $38
                serialize_triggers(result.get('hold_triggers')),    # $39
                result.get('iv_percentile'),                        # $40
                result.get('insider_signal'),                       # $41
                # Phase 3.1 fields
                result.get('outlier_risk_score'),                   # $42
                result.get('risk_flag'),                            # $43
                result.get('weight_growth'),                        # $44
                result.get('weight_momentum'),                      # $45
                result.get('weight_quality'),                       # $46
                result.get('weight_value'),                         # $47
                # Phase 3.4.3: Risk Metrics
                result.get('volatility_annual'),                    # $48
                result.get('max_drawdown_1y'),                      # $49
                result.get('sharpe_ratio'),                         # $50
                result.get('sortino_ratio'),                        # $51
                result.get('calmar_ratio'),                         # $52
                # Phase 3.4.3 Stage 3: Factor Momentum
                result.get('value_momentum'),                       # $53
                result.get('quality_momentum'),                     # $54
                result.get('momentum_momentum'),                    # $55
                result.get('growth_momentum'),                      # $56
                # Phase 3.4.3 Stage 4: Sector Metrics
                result.get('sector_rotation_score'),                # $57
                result.get('sector_momentum'),                      # $58
                result.get('sector_rank'),                          # $59
                result.get('sector_percentile'),                    # $60
                # Phase 3.4.3 Stage 4: Text Generation
                result.get('risk_profile_text'),                    # $61
                result.get('risk_recommendation'),                  # $62
                result.get('time_series_text'),                     # $63
                result.get('signal_overall'),                       # $64
                result.get('market_state')                          # $65
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
            print(f"  레짐: {system.current_regime} / Macro: {getattr(system, 'macro_environment', 'N/A')}")
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
            print(f"  시가총액: ${stock_info.get('market_cap') or 0:,.0f}\n")

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
        # Phase 3.2: NASDAQ이면 전용 Momentum 가중치 사용
        exchange = stock_info.get('exchange', 'NYSE')
        momentum_task = calculate_momentum_score(symbol, db_manager, benchmarks, analysis_date, exchange)
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

        # Step 6: 등급 결정 (7단계 한글)
        grade = system._determine_grade(total_score)
        opinion = grade  # grade 자체가 한글 투자의견

        if verbose:
            print("Step 6: 등급 결정...")
            print(f"  등급: {grade}")
            print(f"  기준: 강력 매수(85+), 매수(75+), 매수 고려(65+), 중립(55+), 매도 고려(45+), 매도(35+), 강력 매도(<35)\n")

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
            print(f"  등급:    {grade}")
            print("="*80 + "\n")

        # Alternative stock matching for sell-grade stocks (Option 1)
        if grade in ('매도 고려', '매도', '강력 매도'):
            if verbose:
                print("Alternative Stock 매칭 중...")
            matcher = USAlternativeStockMatcher(db_manager)
            match_success = await matcher.match_single_symbol(symbol, analysis_date)
            if verbose and match_success:
                # Query and display matched alternative
                alt_query = """
                    SELECT alt_symbol, alt_stock_name, alt_final_grade, alt_final_score,
                           alt_match_type, alt_reasons
                    FROM us_stock_grade
                    WHERE symbol = $1 AND date = $2
                """
                alt_result = await db_manager.execute_query(alt_query, symbol, analysis_date)
                if alt_result and alt_result[0]['alt_symbol']:
                    alt = alt_result[0]
                    print(f"  대체 종목: {alt['alt_symbol']} {alt['alt_stock_name']}")
                    print(f"  등급/점수: {alt['alt_final_grade']} ({alt['alt_final_score']}점)")
                    print(f"  매칭 유형: {alt['alt_match_type']}")
                    if alt['alt_reasons']:
                        reasons = json.loads(alt['alt_reasons']) if isinstance(alt['alt_reasons'], str) else alt['alt_reasons']
                        print(f"  추천 이유:")
                        for i, reason in enumerate(reasons, 1):
                            print(f"    {i}. {reason['text']}")
                    print()

        return result

    except Exception as e:
        logger.error(f"{symbol} 분석 실패: {e}", exc_info=True)
        if verbose:
            print(f"\n[오류] {symbol} 분석 실패")
            print(f"이유: {str(e)}\n")
        return None


async def analyze_all_stocks_specific_dates(
    db_manager,
    date_list: List[date],
    use_hmm: bool = False,
    use_static_weights: bool = False
) -> Dict:
    """
    전체 종목 특정 날짜 분석 - 디버깅 로그 포함

    Args:
        db_manager: AsyncDatabaseManager instance
        date_list: 분석 날짜 리스트
        use_hmm: HMM 기반 레짐 감지 사용 (v2.0)
        use_static_weights: 정적 가중치 사용
    """
    hmm_mode = "HMM (v2.0)" if use_hmm else "규칙 기반 (Legacy)"

    print("\n" + "="*80)
    print("미국 주식 v3.0 - 전체 종목 특정 날짜 분석")
    print("="*80)
    print(f"분석 날짜: {len(date_list)}개")
    for idx, d in enumerate(date_list, 1):
        print(f"  {idx}. {d}")
    print(f"레짐 감지: {hmm_mode}")
    print(f"가중치 모드: {'정적 (v2.0)' if use_static_weights else '동적 Macro 조정'}")
    print(f"DB 커넥션 풀: min_size=10, max_size=30")
    print(f"배치 크기: 25개 종목 (동시 처리)")
    print("="*80 + "\n")

    # Suppress verbose logging
    original_levels = {}
    loggers_to_suppress = [
        'root', '__main__', 'us_main',
        'us_market_regime', 'us_sector_benchmarks',
        'us_value_factor', 'us_quality_factor',
        'us_momentum_factor', 'us_growth_factor'
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

    system = USQuantSystemV2(
        db_manager,
        use_hmm=use_hmm,
        use_static_weights=use_static_weights
    )

    # HMM 모드: 한 번만 학습 (효율성 최적화)
    if use_hmm:
        print("HMM 모델 초기화 중 (2년 데이터 학습, 1회만 실행)...")
        await system._init_hmm_detector()
        print("HMM 모델 준비 완료\n")

    for date_idx, analysis_date in enumerate(date_list, 1):
        date_start_time = time.time()

        print(f"[날짜 {date_idx}/{len(date_list)}] {analysis_date} - 분석 시작...")

        # Sector/Industry MV 데이터 생성 (해당 날짜 데이터 없으면 INSERT)
        try:
            await db_manager.refresh_sector_performance_for_date(analysis_date)
            await db_manager.refresh_industry_performance_for_date(analysis_date)
        except Exception as mv_err:
            logger.warning(f"MV refresh failed for {analysis_date}: {mv_err}")

        # 레짐 감지
        try:
            await system._detect_regime(analysis_date)
            method_str = f"[{system.regime_method}]" if system.regime_method else ""
            print(f"  레짐: {system.current_regime} {method_str} / Macro: {getattr(system, 'macro_environment', 'N/A')}")
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
                    batch_errors.append(f"{symbol}: {type(result).__name__}: {str(result)[:100]}")
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
                # 첫 번째 에러만 출력
                print(f"    [실패] {batch_errors[0]}")
                print(f"    [실패] ... 외 {len(batch_errors)-1}개 종목")

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

        # Phase 3.4.3 Stage 3: Industry Ranking 배치 계산
        print(f"  Industry Rank 계산 중...")
        industry_rank_query = """
        WITH industry_rankings AS (
            SELECT
                g.symbol,
                b.industry,
                RANK() OVER (PARTITION BY b.industry ORDER BY g.final_score DESC) as industry_rank,
                ROUND(
                    (PERCENT_RANK() OVER (PARTITION BY b.industry ORDER BY g.final_score DESC) * 100)::NUMERIC,
                    1
                ) as industry_percentile
            FROM us_stock_grade g
            JOIN us_stock_basic b ON g.symbol = b.symbol
            WHERE g.date = $1
              AND b.industry IS NOT NULL
              AND b.industry != ''
        )
        UPDATE us_stock_grade g
        SET
            industry_rank = r.industry_rank,
            industry_percentile = r.industry_percentile
        FROM industry_rankings r
        WHERE g.symbol = r.symbol AND g.date = $1
        """
        try:
            await db_manager.execute_query(industry_rank_query, analysis_date)
            print(f"  Industry Rank 계산 완료")
        except Exception as e:
            print(f"  [경고] Industry Rank 계산 실패: {e}")

        # Alternative stock matching for this date
        print(f"  Alternative Stock 매칭 중...")
        matcher = USAlternativeStockMatcher(db_manager)
        match_result = await matcher.match_for_date(analysis_date)
        print(f"  Alternative Stock 매칭 완료: {match_result['matched']}개 성공, {match_result['failed']}개 실패")
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
                print("\n전체 종목 특정 날짜 분석 (v3.0)")
                print("-" * 50)

                # HMM 모드 선택
                print("\n[레짐 감지 모드 선택]")
                print("  1. 규칙 기반 (Legacy) - 5-State: AI_BULL/TIGHTENING/RECOVERY/CRISIS/NEUTRAL")
                print("  2. HMM 기반 (v2.0)   - 3-State: BULL/NEUTRAL/BEAR (학술 연구 기반)")
                hmm_choice = input("\n선택 (1/2, 기본=1): ").strip()
                use_hmm = hmm_choice == '2'

                if use_hmm:
                    print("\n[HMM 모드 활성화]")
                    print("  - 3-State Hidden Markov Model (BULL/NEUTRAL/BEAR)")
                    print("  - Factor Timing 최소화 (IC 변동성 고려)")
                    print("  - 정적 가중치 (분기별 리밸런싱 권고)")
                    use_static_weights = True
                else:
                    print("\n[규칙 기반 모드]")
                    print("  - 5-State 레짐 분류")
                    print("  - 동적 Macro 조정")
                    use_static_weights = False

                print("\n" + "-" * 50)
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
                    print(f"레짐 모드: {'HMM (v2.0)' if use_hmm else '규칙 기반 (Legacy)'}")

                    confirm = input("\n분석을 시작하시겠습니까? (y/n): ").strip().lower()
                    if confirm == 'y':
                        # 커넥션 풀 재초기화
                        print("\n커넥션 풀 최적화 중...")
                        await db_manager.close()
                        db_manager = AsyncDatabaseManager()
                        await db_manager.initialize(min_size=10, max_size=30)

                        await analyze_all_stocks_specific_dates(
                            db_manager,
                            date_list,
                            use_hmm=use_hmm,
                            use_static_weights=use_static_weights
                        )
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
