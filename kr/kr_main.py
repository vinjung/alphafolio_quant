"""
Korean Stock Main Analysis System
메인 분석 시스템 - 단일 종목 또는 전체 종목 분석

Usage:
    python kr_main.py
"""

import os
import asyncio
import logging
import time
import traceback
import json
from datetime import date, datetime
from dotenv import load_dotenv

# Import async modules
from db_async import AsyncDatabaseManager, save_to_kr_stock_grade, batch_save_to_kr_stock_grade
from kr_value_factor import ValueFactorCalculator
from kr_quality_factor import QualityFactorCalculator
from kr_momentum_factor import MomentumFactorCalculator
from kr_growth_factor import GrowthFactorCalculator
from weight import ConditionAnalyzer, WeightCalculator
from kr_additional_metrics import AdditionalMetricsCalculator
from kr_interpretation import InterpretationEngine
from batch_weight import AsyncBatchWeightCalculator

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def analyze_single_stock(symbol, db_manager, analysis_date=None, verbose=True, precomputed_weights=None, precomputed_conditions=None):
    """
    단일 종목 분석 (Option 1)

    Args:
        symbol: 종목 코드
        db_manager: AsyncDatabaseManager 인스턴스
        analysis_date: 분석 날짜 (None이면 자동 조회)
        verbose: 상세 로그 출력 여부 (기본값: True)
        precomputed_weights: 사전 계산된 가중치 (batch processing용)
        precomputed_conditions: 사전 계산된 조건 (batch processing용)

    Returns:
        bool: 성공 여부
    """
    try:
        if verbose:
            print(f"\n{'='*80}")
            print(f"종목 분석 시작: {symbol}")
            print(f"{'='*80}\n")

        # Step 1: 조건 분석 및 시장 상태 분류
        if precomputed_weights is not None and precomputed_conditions is not None:
            # Use precomputed values from batch processing
            final_weights = precomputed_weights
            conditions = precomputed_conditions
            market_state = conditions.get('market_state', '기타')
            if verbose:
                print(f"시장 상태: {market_state} (배치 처리 사용)\n")
        else:
            # Calculate weights individually (Option 1)
            if verbose:
                print("Step 1: 조건 분석 및 시장 상태 분류...")
            analyzer = ConditionAnalyzer(symbol, db_manager)
            conditions, condition_weights = await analyzer.analyze()
            market_state = conditions.get('market_state', '기타')

            # Calculate dynamic weights using WeightCalculator
            calculator = WeightCalculator()
            final_weights = calculator.calculate_final_weights(condition_weights)

            if verbose:
                print(f"시장 상태: {market_state}\n")

        # Step 2: 4개 팩터 점수 계산 (병렬 처리 - Pro 플랜 최적화)
        if verbose:
            print("Step 2: 팩터 점수 계산 (병렬 처리)...")

        # Create factor calculator instances
        value_calc = ValueFactorCalculator(symbol, db_manager, market_state=market_state, analysis_date=analysis_date)
        quality_calc = QualityFactorCalculator(symbol, db_manager, market_state=market_state, analysis_date=analysis_date)
        momentum_calc = MomentumFactorCalculator(symbol, db_manager, market_state=market_state, analysis_date=analysis_date)
        growth_calc = GrowthFactorCalculator(symbol, db_manager, market_state=market_state, analysis_date=analysis_date)

        # Execute in parallel using asyncio.gather
        results = await asyncio.gather(
            value_calc.calculate_weighted_score(),
            quality_calc.calculate_weighted_score(),
            momentum_calc.calculate_weighted_score(),
            growth_calc.calculate_weighted_score(),
            return_exceptions=True
        )

        # Unpack results
        value_result = results[0] if not isinstance(results[0], Exception) else None
        quality_result = results[1] if not isinstance(results[1], Exception) else None
        momentum_result = results[2] if not isinstance(results[2], Exception) else None
        growth_result = results[3] if not isinstance(results[3], Exception) else None

        # Extract scores
        value_score = value_result['weighted_score'] if value_result else 0
        quality_score = quality_result['weighted_score'] if quality_result else 0
        momentum_score = momentum_result['weighted_score'] if momentum_result else 0
        growth_score = growth_result['weighted_score'] if growth_result else 0

        # Serialize strategies_scores to JSON for v2_detail columns
        def serialize_strategies(calc):
            """Convert strategies_scores dict to JSON string"""
            if hasattr(calc, 'strategies_scores') and calc.strategies_scores:
                clean_scores = {}
                for k, v in calc.strategies_scores.items():
                    if v is not None:
                        clean_scores[k] = round(float(v), 2)
                return json.dumps(clean_scores) if clean_scores else None
            return None

        value_v2_detail = serialize_strategies(value_calc)
        quality_v2_detail = serialize_strategies(quality_calc)
        momentum_v2_detail = serialize_strategies(momentum_calc)
        growth_v2_detail = serialize_strategies(growth_calc)

        # Print individual strategy scores (after parallel execution)
        if verbose:
            print("  팩터 점수 계산 완료 (병렬 처리)")

            if hasattr(value_calc, 'strategies_scores') and value_calc.strategies_scores:
                print("    [Value 개별 전략 점수]")
                for strategy_name, score in value_calc.strategies_scores.items():
                    if score is not None:
                        print(f"      {strategy_name}: {score:.2f}")
                    else:
                        print(f"      {strategy_name}: None")

            if hasattr(quality_calc, 'strategies_scores') and quality_calc.strategies_scores:
                print("    [Quality 개별 전략 점수]")
                for strategy_name, score in quality_calc.strategies_scores.items():
                    if score is not None:
                        print(f"      {strategy_name}: {score:.2f}")
                    else:
                        print(f"      {strategy_name}: None")

            if hasattr(momentum_calc, 'strategies_scores') and momentum_calc.strategies_scores:
                print("    [Momentum 개별 전략 점수]")
                for strategy_name, score in momentum_calc.strategies_scores.items():
                    if score is not None:
                        print(f"      {strategy_name}: {score:.2f}")
                    else:
                        print(f"      {strategy_name}: None")

            if hasattr(growth_calc, 'strategies_scores') and growth_calc.strategies_scores:
                print("    [Growth 개별 전략 점수]")
                for strategy_name, score in growth_calc.strategies_scores.items():
                    if score is not None:
                        print(f"      {strategy_name}: {score:.2f}")
                    else:
                        print(f"      {strategy_name}: None")

            print(f"  Value: {value_score:.2f}, Quality: {quality_score:.2f}, "
                  f"Momentum: {momentum_score:.2f}, Growth: {growth_score:.2f}\n")

        # Step 3: 종목명 조회
        if verbose:
            print("Step 3: 종목명 조회...")
        stock_name_query = "SELECT stock_name FROM kr_stock_basic WHERE symbol = $1"
        stock_name_result = await db_manager.execute_query(stock_name_query, symbol)
        stock_name = stock_name_result[0]['stock_name'] if stock_name_result else None

        # Step 4: 추가 정량 지표 계산 (Phase 1: 먼저 계산하여 smart_money_score, volatility_context_score 획득)
        if verbose:
            print("Step 4: 추가 정량 지표 계산...")

        # Prepare factor_scores dict for AdditionalMetricsCalculator
        factor_scores_dict = {
            'value': {
                'weighted_result': value_result,
                'strategies': value_result.get('strategies', {}) if value_result else {}
            },
            'quality': {
                'weighted_result': quality_result,
                'strategies': quality_result.get('strategies', {}) if quality_result else {}
            },
            'momentum': {
                'weighted_result': momentum_result,
                'strategies': momentum_result.get('strategies', {}) if momentum_result else {}
            },
            'growth': {
                'weighted_result': growth_result,
                'strategies': growth_result.get('strategies', {}) if growth_result else {}
            }
        }

        metrics_calc = AdditionalMetricsCalculator(symbol, db_manager, factor_scores_dict, analysis_date=analysis_date)
        additional_metrics = await metrics_calc.calculate_all_metrics()

        # Step 4.5: 멀티 에이전트 AI용 신규 지표 계산 (Phase Agent)
        if verbose:
            print("Step 4.5: 에이전트용 신규 지표 계산...")

        # ATR 기반 손절/익절
        atr_metrics = await metrics_calc.calculate_atr_stop_take_profit()

        # Step 5: 최종 점수 계산 (Option 3: IC-based weights)
        if verbose:
            print("Step 5: 최종 점수 계산 (Option 3: IC 기반 가중치)...")

        # Base Factor Score (동적 가중치 적용)
        base_factor_score = (
            value_score * final_weights['value'] +
            quality_score * final_weights['quality'] +
            momentum_score * final_weights['momentum'] +
            growth_score * final_weights['growth']
        )

        # Get helpful signal scores from additional_metrics (Option 3: IC > 0 only)
        # Removed harmful metrics (IC < 0):
        # - smart_money_signal_score (IC -0.0167)
        # - volatility_context_score (IC -0.0194)
        # - smart_money_momentum_score (IC -0.0110)
        # - score_momentum_signal (IC -0.0329)
        factor_combo_bonus = additional_metrics.get('factor_combination_bonus', 0)
        sector_rotation_score = additional_metrics.get('sector_rotation_score', 50)

        # Final Score (Option 3: IC-based weights - Only helpful metrics)
        # All components are 0-100 scale for consistency
        # Weights based on IC contribution:
        # - base_factor_score: 36% (IC +0.0463, range 0-100)
        # - sector_rotation_score: 25% (IC +0.0321, range 0-100)
        # - factor_combination_bonus: 39% (IC +0.0503, range 0-100)
        # Validation result: IC improved from +0.0038 to +0.0600 (+1,479%)
        # Result: final_score is 0-100 scale (100-point system)
        final_score = (
            base_factor_score * 0.36 +           # 0-100 x 0.36
            sector_rotation_score * 0.25 +       # 0-100 x 0.25
            factor_combo_bonus * 0.39            # 0-100 x 0.39 = 0-100 total
        )
        # Round to 1 decimal place for DECIMAL(5,1) format
        final_score = round(final_score, 1)

        # Step 5.5: 나머지 에이전트용 지표 계산 (final_score 필요)
        # Entry Timing Score
        entry_timing = await metrics_calc.calculate_entry_timing_score(final_score, momentum_score)

        # Position Size
        position_size = await metrics_calc.calculate_position_size(additional_metrics.get('volatility_annual'))

        # 업종 정보 조회 (scenario_probability용)
        industry_query = "SELECT industry FROM kr_stock_detail WHERE symbol = $1 LIMIT 1"
        industry_result = await db_manager.execute_query(industry_query, symbol)
        industry = industry_result[0]['industry'] if industry_result and industry_result[0]['industry'] else 'Unknown'

        # Scenario Probability
        scenario = await metrics_calc.calculate_scenario_probability(final_score, industry)

        # Action Triggers
        stop_loss_pct = atr_metrics.get('stop_loss_pct', -10) if atr_metrics else -10
        take_profit_pct = atr_metrics.get('take_profit_pct', 15) if atr_metrics else 15
        sector_percentile = additional_metrics.get('sector_percentile', 50) or 50
        triggers = metrics_calc.generate_action_triggers(final_score, sector_percentile, stop_loss_pct, take_profit_pct)

        if verbose:
            print(f"  Base Factor Score: {base_factor_score:.2f} "
                  f"(Value: {final_weights['value']*100:.1f}%, "
                  f"Quality: {final_weights['quality']*100:.1f}%, "
                  f"Momentum: {final_weights['momentum']*100:.1f}%, "
                  f"Growth: {final_weights['growth']*100:.1f}%)")
            print(f"  Sector Rotation Score: {sector_rotation_score}/100")
            print(f"  Factor Combo Bonus: {factor_combo_bonus}/100")
            print(f"  Final Score: {final_score:.2f} (Option 3: 36%+25%+39%, IC +0.0600)\n")

        # Step 6: 등급 결정 (Phase 3.7: 시장 타이밍 필터 적용)
        # Check 1: 모든 팩터가 0점인지 확인 (데이터 부족)
        if value_score == 0 and quality_score == 0 and momentum_score == 0 and growth_score == 0:
            final_grade = '평가 불가'
            final_score = None
            filters_passed = 0
        # Check 2: 신뢰도 기반 평가 불가
        elif additional_metrics.get('confidence_score', 0) < 30:
            final_grade = '평가 불가'
            final_score = None
            filters_passed = 0
        # Check 3: final_score가 0점
        elif final_score == 0:
            final_grade = '평가 불가'
            filters_passed = 0
        else:
            # Phase 3.9: Market Timing Filters (Decile-based thresholds)
            # Get market regime thresholds (based on Decile analysis: Top Decile 43.6+)
            market_sentiment = conditions.get('market_sentiment', 'NEUTRAL')
            regime_thresholds = {
                'PANIC': (50, 25),      # Bear: conservative BUY threshold
                'FEAR': (48, 23),       # Bear: conservative
                'NEUTRAL': (45, 21),    # Sideways: default (Decile 9-10 based)
                'GREED': (43, 21),      # Bull: aggressive
                'OVERHEATED': (48, 25)  # Overheated: conservative
            }
            buy_threshold, sell_threshold = regime_thresholds.get(market_sentiment, (45, 21))

            # Filter 1: Momentum confirmation (min 35 for buy)
            momentum_pass = momentum_score >= 35
            # Filter 2: Trend confirmation (SuperTrend)
            supertrend = additional_metrics.get('supertrend_signal', '보유')
            trend_pass = supertrend in ['매수', '보유']
            # Filter 3: Relative Strength (min -10% for buy)
            rs_20d = additional_metrics.get('rs_20d', 0) or 0
            rs_pass = rs_20d >= -10
            # Filter 4: RSI overbought filter (RSI > 70 blocks BUY)
            rsi = additional_metrics.get('rsi', 50) or 50
            rsi_not_overbought = rsi <= 70
            # Filter 5: Entry Timing (Phase 3.10 - High Entry >= 70 for 70%+ win rate)
            entry_timing_score = entry_timing.get('entry_timing_score', 0) if entry_timing else 0
            entry_timing_pass = entry_timing_score >= 70

            filters_passed = sum([momentum_pass, trend_pass, rs_pass, rsi_not_overbought, entry_timing_pass])

            # Phase 3.9: Theme-based penalty for low IC themes
            theme = conditions.get('theme', '')
            low_ic_themes = ['Telecom_Media', 'IT_Software']
            theme_penalty = 5 if theme in low_ic_themes else 0
            adjusted_buy_threshold = buy_threshold + theme_penalty

            # Grade determination with timing filters (5 filters, require 4+ for BUY)
            # Phase 3.10: Entry Timing Score added (High Entry + High Score = 70%+ win rate)
            if final_score >= adjusted_buy_threshold and filters_passed >= 4:
                final_grade = '매수'
            elif final_score >= adjusted_buy_threshold - 5 and filters_passed >= 3:
                final_grade = '매수 고려'
            elif final_score < sell_threshold or momentum_score < 25:
                if final_score < sell_threshold - 10:
                    final_grade = '매도'
                else:
                    final_grade = '매도 고려'
            else:
                final_grade = '중립'

        if verbose:
            print(f"Step 6: 최종 결과 (Phase 3.9: 시장 타이밍 필터)")
            if final_grade != '평가 불가' and final_score is not None:
                print(f"  최종 점수: {final_score:.2f}")
                print(f"  시장 레짐: {conditions.get('market_sentiment', 'NEUTRAL')} (Buy >= {adjusted_buy_threshold}, Sell < {sell_threshold})")
                print(f"  테마 페널티: {theme_penalty} ({theme})")
                print(f"  필터 통과: {filters_passed}/5 (Momentum: {momentum_pass}, Trend: {trend_pass}, RS: {rs_pass}, RSI: {rsi_not_overbought}, EntryTiming: {entry_timing_pass})")
            else:
                print(f"  최종 점수: N/A (데이터 부족)")
            print(f"  투자 등급: {final_grade}\n")

        # Step 7: 해석 및 통찰 생성
        if verbose:
            print("Step 7: 해석 및 통찰 생성...")
        interpretation = InterpretationEngine(
            symbol,
            factor_scores_dict,
            additional_metrics,
            {'value': value_score, 'quality': quality_score, 'momentum': momentum_score, 'growth': growth_score},
            final_score,
            final_grade
        )
        interpretations = interpretation.generate_all_interpretations()

        # Step 8: 데이터베이스 저장
        if verbose:
            print("Step 8: kr_stock_grade 테이블에 저장 중...")

        # Get analysis date from latest data if not provided
        if analysis_date is None:
            date_query = """
            SELECT MAX(date) as latest_date
            FROM kr_intraday_total
            WHERE symbol = $1
            """
            date_result = await db_manager.execute_query(date_query, symbol)
            analysis_date = date_result[0]['latest_date'] if date_result and date_result[0]['latest_date'] else date.today()
            if verbose:
                print(f"  분석 기준일: {analysis_date} (kr_intraday_total 최신 데이터 기준)\n")

        # Note: All values are stored directly in additional_metrics dict, not nested

        # Extract signal_lights for overall assessment
        signal_lights = interpretations.get('signal_lights', {})

        grade_data = {
            'stock_name': stock_name,
            'final_grade': final_grade,
            'final_score': final_score,
            'value_score': value_score,
            'quality_score': quality_score,
            'momentum_score': momentum_score,
            'growth_score': growth_score,
            'confidence_score': additional_metrics.get('confidence_score'),
            # 리스크 지표
            'var_95': additional_metrics.get('var_95'),
            'cvar_95': additional_metrics.get('cvar_95'),
            'volatility_annual': additional_metrics.get('volatility_annual'),
            'max_drawdown_1y': additional_metrics.get('max_drawdown_1y'),
            'beta': additional_metrics.get('beta'),
            # 수급 지표
            'inst_net_30d': additional_metrics.get('inst_net_30d'),
            'foreign_net_30d': additional_metrics.get('foreign_net_30d'),
            # 팩터 모멘텀
            'value_momentum': additional_metrics.get('value_momentum'),
            'quality_momentum': additional_metrics.get('quality_momentum'),
            'momentum_momentum': additional_metrics.get('momentum_momentum'),
            'growth_momentum': additional_metrics.get('growth_momentum'),
            # 업종 분석
            'industry_rank': additional_metrics.get('industry_rank'),
            'industry_percentile': additional_metrics.get('industry_percentile'),
            # 상대강도 (유지)
            'rs_value': additional_metrics.get('rs_value'),
            'rs_rank': additional_metrics.get('rs_rank'),
            # Phase 2: Signal Scores
            'factor_combination_bonus': additional_metrics.get('factor_combination_bonus'),
            # Sector Rotation
            'sector_rotation_score': additional_metrics.get('sector_rotation_score'),
            'sector_momentum': additional_metrics.get('sector_momentum'),
            'sector_rank': additional_metrics.get('sector_rank'),
            'sector_percentile': additional_metrics.get('sector_percentile'),
            # ========== Phase Agent: 멀티 에이전트 AI용 신규 지표 ==========
            # 진입 타이밍
            'entry_timing_score': entry_timing.get('entry_timing_score') if entry_timing else None,
            'score_trend_2w': entry_timing.get('score_trend_2w') if entry_timing else None,
            'price_position_52w': entry_timing.get('price_position_52w') if entry_timing else None,
            # 리스크 관리 (ATR 기반)
            'atr_pct': atr_metrics.get('atr_pct') if atr_metrics else None,
            'stop_loss_pct': atr_metrics.get('stop_loss_pct') if atr_metrics else None,
            'take_profit_pct': atr_metrics.get('take_profit_pct') if atr_metrics else None,
            'risk_reward_ratio': atr_metrics.get('risk_reward_ratio') if atr_metrics else None,
            'position_size_pct': position_size,
            # 시나리오 분석
            'scenario_bullish_prob': scenario.get('scenario_bullish_prob') if scenario else None,
            'scenario_sideways_prob': scenario.get('scenario_sideways_prob') if scenario else None,
            'scenario_bearish_prob': scenario.get('scenario_bearish_prob') if scenario else None,
            'scenario_bullish_return': scenario.get('scenario_bullish_return') if scenario else None,
            'scenario_sideways_return': scenario.get('scenario_sideways_return') if scenario else None,
            'scenario_bearish_return': scenario.get('scenario_bearish_return') if scenario else None,
            'scenario_sample_count': scenario.get('scenario_sample_count') if scenario else None,
            # 행동 트리거 (JSON)
            'buy_triggers': json.dumps(triggers.get('buy_triggers', []), ensure_ascii=False) if triggers else None,
            'sell_triggers': json.dumps(triggers.get('sell_triggers', []), ensure_ascii=False) if triggers else None,
            'hold_triggers': json.dumps(triggers.get('hold_triggers', []), ensure_ascii=False) if triggers else None,
            # Interpretation & Signal Lights
            'risk_profile_text': interpretations.get('risk_profile', {}).get('interpretation'),
            'risk_recommendation': interpretations.get('risk_profile', {}).get('recommendation'),
            'time_series_text': interpretations.get('time_series_trend', {}).get('interpretation'),
            'signal_overall': signal_lights.get('overall_assessment'),
            'market_state': market_state,
            # v2_detail columns - individual strategy scores as JSON
            'value_v2_detail': value_v2_detail,
            'quality_v2_detail': quality_v2_detail,
            'momentum_v2_detail': momentum_v2_detail,
            'growth_v2_detail': growth_v2_detail,
            # Phase 3.9: Risk-Adjusted Returns
            'sharpe_ratio': additional_metrics.get('sharpe_ratio'),
            'sortino_ratio': additional_metrics.get('sortino_ratio'),
            'calmar_ratio': additional_metrics.get('calmar_ratio')
        }

        success = await save_to_kr_stock_grade(db_manager, symbol, analysis_date, grade_data)

        if success and verbose:
            print(f"[OK] 저장 완료: {symbol}\n")

            # Display saved data
            print(f"{'='*80}")
            print(f"저장된 분석 결과 상세 내역")
            print(f"{'='*80}\n")

            # Section 1: Basic Info
            print(f"[1] 기본 정보")
            print(f"  종목코드: {symbol}")
            print(f"  종목명: {stock_name or 'N/A'}")
            print(f"  분석일자: {analysis_date}")
            print(f"  시장상태: {market_state}\n")

            # Section 2: Final Grade & Score
            print(f"[2] 최종 투자 등급")
            print(f"  등급: {final_grade}")
            if final_score is not None:
                print(f"  점수: {final_score:.2f}점")
            else:
                print(f"  점수: N/A (데이터 부족)")
            print(f"  신뢰도: {additional_metrics.get('confidence_score', 0):.1f}%\n")

            # Section 3: Factor Scores
            print(f"[3] 팩터별 점수")
            print(f"  Value Factor:    {value_score:.2f}점")
            print(f"  Quality Factor:  {quality_score:.2f}점")
            print(f"  Momentum Factor: {momentum_score:.2f}점")
            print(f"  Growth Factor:   {growth_score:.2f}점\n")

            # Section 4: 에이전트용 투자 전략 지표 (Phase Agent)
            print(f"[4] 투자 전략 지표 (AI 에이전트용)")

            # 4-1. 진입 타이밍
            print(f"  [진입 타이밍]")
            ets = entry_timing.get('entry_timing_score') if entry_timing else None
            st2w = entry_timing.get('score_trend_2w') if entry_timing else None
            pp52 = entry_timing.get('price_position_52w') if entry_timing else None
            print(f"    타이밍 점수: {ets:.1f}/100" if ets else "    타이밍 점수: N/A")
            print(f"    점수 추세(2주): {st2w:+.1f}점" if st2w is not None else "    점수 추세(2주): N/A")
            print(f"    52주 위치: {pp52:.1f}% (0%=저점, 100%=고점)" if pp52 else "    52주 위치: N/A")

            # 4-2. 리스크 관리
            print(f"  [리스크 관리]")
            atr = atr_metrics.get('atr_pct') if atr_metrics else None
            sl = atr_metrics.get('stop_loss_pct') if atr_metrics else None
            tp = atr_metrics.get('take_profit_pct') if atr_metrics else None
            rr = atr_metrics.get('risk_reward_ratio') if atr_metrics else None
            ps = position_size
            print(f"    ATR%: {atr:.2f}%" if atr else "    ATR%: N/A")
            print(f"    손절선: {sl:.1f}%" if sl else "    손절선: N/A")
            print(f"    익절선: +{tp:.1f}%" if tp else "    익절선: N/A")
            print(f"    손익비: {rr:.2f}" if rr else "    손익비: N/A")
            print(f"    권장 비중: {ps:.1f}%" if ps else "    권장 비중: N/A")

            # 4-3. 시나리오 분석
            print(f"  [시나리오 분석]")
            bp = scenario.get('scenario_bullish_prob') if scenario else None
            sp = scenario.get('scenario_sideways_prob') if scenario else None
            bear_p = scenario.get('scenario_bearish_prob') if scenario else None
            br = scenario.get('scenario_bullish_return') if scenario else None
            sr = scenario.get('scenario_sideways_return') if scenario else None
            bear_r = scenario.get('scenario_bearish_return') if scenario else None
            sc = scenario.get('scenario_sample_count') if scenario else None
            print(f"    강세: {bp}% ({br})" if bp and br else "    강세: N/A")
            print(f"    횡보: {sp}% ({sr})" if sp and sr else "    횡보: N/A")
            print(f"    약세: {bear_p}% ({bear_r})" if bear_p and bear_r else "    약세: N/A")
            print(f"    분석 샘플: {sc}건" if sc else "    분석 샘플: N/A")

            # 4-4. 행동 트리거
            print(f"  [행동 트리거]")
            buy_t = triggers.get('buy_triggers', []) if triggers else []
            sell_t = triggers.get('sell_triggers', []) if triggers else []
            print(f"    매수: {', '.join(buy_t[:2])}" if buy_t else "    매수: N/A")
            print(f"    매도: {', '.join(sell_t[:2])}" if sell_t else "    매도: N/A")
            print()

            # Section 5: VaR/CVaR (Risk Metric)
            print(f"[5] 리스크 지표 (VaR/CVaR)")
            var_95 = additional_metrics.get('var_95')
            cvar_95 = additional_metrics.get('cvar_95')
            if var_95 is not None:
                print(f"  VaR(95%):  {var_95:.2f}%")
            else:
                print(f"  VaR(95%):  N/A")
            if cvar_95 is not None:
                print(f"  CVaR(95%): {cvar_95:.2f}%")
            else:
                print(f"  CVaR(95%): N/A")
            print()

            # Section 6: Investor Flow (30-day)
            print(f"[6] 투자자별 순매수 (30일)")
            inst_net = additional_metrics.get('inst_net_30d')
            foreign_net = additional_metrics.get('foreign_net_30d')

            if inst_net is not None:
                print(f"  기관: {inst_net:,.0f}주")
            else:
                print(f"  기관: N/A")

            if foreign_net is not None:
                print(f"  외국인: {foreign_net:,.0f}주")
            else:
                print(f"  외국인: N/A")
            print()

            # Section 7: Factor Momentum
            print(f"[7] 팩터별 모멘텀")
            val_mom = additional_metrics.get('value_momentum')
            qual_mom = additional_metrics.get('quality_momentum')
            mom_mom = additional_metrics.get('momentum_momentum')
            grow_mom = additional_metrics.get('growth_momentum')

            print(f"  Value Momentum:    {val_mom:.2f}점" if val_mom is not None else "  Value Momentum:    N/A")
            print(f"  Quality Momentum:  {qual_mom:.2f}점" if qual_mom is not None else "  Quality Momentum:  N/A")
            print(f"  Momentum Momentum: {mom_mom:.2f}점" if mom_mom is not None else "  Momentum Momentum: N/A")
            print(f"  Growth Momentum:   {grow_mom:.2f}점" if grow_mom is not None else "  Growth Momentum:   N/A")
            print()

            # Section 8: Industry Rank
            print(f"[8] 업종 내 순위")
            ind_rank = additional_metrics.get('industry_rank')
            ind_pct = additional_metrics.get('industry_percentile')

            if ind_rank is not None:
                print(f"  순위: {ind_rank}위")
            else:
                print(f"  순위: N/A")

            if ind_pct is not None:
                print(f"  백분위: 상위 {ind_pct:.1f}%")
            else:
                print(f"  백분위: N/A")
            print()

            # Section 9: Beta & Volatility
            print(f"[9] 베타 및 변동성")
            beta = additional_metrics.get('beta')
            volatility = additional_metrics.get('volatility_annual')

            if beta is not None:
                print(f"  Beta: {beta:.3f}")
            else:
                print(f"  Beta: N/A")

            if volatility is not None:
                print(f"  연간 변동성: {volatility:.2f}%")
            else:
                print(f"  연간 변동성: N/A")
            print()

            # Section 10: Max Drawdown
            print(f"[10] 최대 낙폭 (1년)")
            mdd = additional_metrics.get('max_drawdown_1y')
            if mdd is not None:
                print(f"  MDD: {mdd:.2f}%")
            else:
                print(f"  MDD: N/A")
            print()

            # Section 11: 상대강도
            print(f"[11] 상대강도 (RS)")
            rs_value = additional_metrics.get('rs_value')
            rs_rank = additional_metrics.get('rs_rank')
            print(f"  RS 값: {rs_value:+.2f}%" if rs_value is not None else "  RS 값: N/A")
            print(f"  RS 등급: {rs_rank}" if rs_rank else "  RS 등급: N/A")
            print()

            # Section 12: Risk Profile
            print(f"[12] 리스크 프로파일")
            risk_text = interpretations.get('risk_profile', {}).get('interpretation')
            risk_rec = interpretations.get('risk_profile', {}).get('recommendation')

            if risk_text:
                print(f"  분석: {risk_text}")
            else:
                print(f"  분석: N/A")

            if risk_rec:
                print(f"  권장: {risk_rec}")
            else:
                print(f"  권장: N/A")
            print()

            # Section 13: Time Series Trend
            print(f"[13] 시계열 트렌드")
            ts_text = interpretations.get('time_series_trend', {}).get('interpretation')
            if ts_text:
                print(f"  {ts_text}")
            else:
                print(f"  N/A")
            print()

            # Section 14: Signal Lights
            print(f"[14] 신호등 분석")
            overall_signal = signal_lights.get('overall_assessment')
            buy_count = signal_lights.get('buy_count', 0)
            neutral_count = signal_lights.get('neutral_count', 0)
            sell_count = signal_lights.get('sell_count', 0)

            if overall_signal:
                print(f"  종합 평가: {overall_signal}")
            else:
                print(f"  종합 평가: N/A")

            print(f"  매수 신호: {buy_count}개")
            print(f"  중립 신호: {neutral_count}개")
            print(f"  매도 신호: {sell_count}개")
            print()

            print(f"{'='*80}\n")

        if success:
            return True
        else:
            if verbose:
                print(f"[ERROR] 저장 실패: {symbol}\n")
                print(f"{'='*80}\n")
            return False

    except Exception as e:
        logger.error(f"종목 분석 실패 ({symbol}): {e}", exc_info=True)
        if verbose:
            print(f"\n[ERROR] 종목 분석 중 오류 발생: {e}\n")
        return False


async def analyze_all_stocks(db_manager):
    """
    전체 종목 분석 (Option 2) - batch_weight.py 사용

    Args:
        db_manager: AsyncDatabaseManager 인스턴스

    Returns:
        dict: 성공/실패 통계
    """
    try:
        # Get analysis date from latest data
        date_query = """
        SELECT MAX(date) as latest_date
        FROM kr_intraday_total
        """
        date_result = await db_manager.execute_query(date_query)
        analysis_date = date_result[0]['latest_date'] if date_result and date_result[0]['latest_date'] else date.today()

        # Get symbols that have data on the analysis date
        query = """
        SELECT DISTINCT symbol
        FROM kr_intraday_total
        WHERE date = $1
        ORDER BY symbol
        """

        result = await db_manager.execute_query(query, analysis_date)
        symbols = [row['symbol'] for row in result]
        total_count = len(symbols)

        BATCH_SIZE = 20

        print("\n" + "="*80)
        print(f"전체 종목 분석 시작 (Batch Weight 최적화)")
        print("="*80)
        print(f"분석 기준일: {analysis_date}")
        print(f"총 종목 수: {total_count:,}개")
        print(f"배치 크기: {BATCH_SIZE}개 (동시 처리)")
        print("="*80 + "\n")

        # Suppress verbose logging during batch processing
        original_levels = {}
        loggers_to_suppress = [
            'root',
            '__main__',
            'kr_main',
            'weight',
            'kr_value_factor',
            'kr_quality_factor',
            'kr_momentum_factor',
            'kr_growth_factor',
            'kr_additional_metrics',
            'kr_interpretation',
            'batch_weight'
        ]

        for logger_name in loggers_to_suppress:
            target_logger = logging.getLogger(logger_name)
            original_levels[logger_name] = target_logger.level
            target_logger.setLevel(logging.ERROR)

        start_time = time.time()

        # Step 1: Use AsyncBatchWeightCalculator to compute weights and conditions
        print("Step 1: 공통 데이터 및 가중치 일괄 계산 (AsyncBatchWeightCalculator)...")
        batch_calc = AsyncBatchWeightCalculator()
        await batch_calc.initialize()

        batch_results = await batch_calc.process_all_stocks(symbols)

        await batch_calc.close()

        print(f"가중치 계산 완료: {len(batch_results):,}개 종목\n")

        # Step 2: Process each stock with precomputed weights
        print("Step 2: 각 종목별 팩터 점수 계산 및 저장...")

        batches = [symbols[i:i + BATCH_SIZE] for i in range(0, total_count, BATCH_SIZE)]
        total_batches = len(batches)

        success_count = 0
        fail_count = 0
        batch_errors = []

        for batch_idx, batch in enumerate(batches, 1):
            batch_start = time.time()

            # Process batch concurrently with precomputed weights
            tasks = []
            for symbol in batch:
                if symbol in batch_results:
                    precomputed_weights = batch_results[symbol]['weights']
                    precomputed_conditions = batch_results[symbol]['conditions']
                    tasks.append(
                        analyze_single_stock(
                            symbol,
                            db_manager,
                            analysis_date,
                            verbose=False,
                            precomputed_weights=precomputed_weights,
                            precomputed_conditions=precomputed_conditions
                        )
                    )
                else:
                    # Fallback: no precomputed data
                    tasks.append(analyze_single_stock(symbol, db_manager, analysis_date, verbose=False))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successes and failures
            batch_success = 0
            batch_fail = 0

            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    batch_fail += 1
                    batch_errors.append(f"{batch[idx]}: {str(result)}")
                elif isinstance(result, bool) and result:
                    batch_success += 1
                else:
                    batch_fail += 1
                    batch_errors.append(f"{batch[idx]}: Unknown error")

            success_count += batch_success
            fail_count += batch_fail

            # Batch progress log (concise)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] [배치 {batch_idx}/{total_batches}] 완료: {batch_success}개 | 실패: {batch_fail}개")

            # Small delay between batches (Windows semaphore stabilization)
            if batch_idx < total_batches:
                await asyncio.sleep(0.3)

        total_time = time.time() - start_time

        # Restore original log levels
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)

        print("\n" + "="*80)
        print("전체 분석 완료")
        print("="*80)
        print(f"총 종목: {total_count:,}개")
        print(f"성공: {success_count:,}개")
        print(f"실패: {fail_count:,}개")
        print(f"성공률: {(success_count/total_count*100):.1f}%")
        print(f"총 소요시간: {total_time/60:.1f}분 ({total_time:.0f}초)")
        print("="*80 + "\n")

        return {
            'total': total_count,
            'success': success_count,
            'fail': fail_count
        }

    except Exception as e:
        logger.error(f"전체 종목 분석 실패: {e}", exc_info=True)
        return None


async def analyze_all_stocks_date_range(db_manager, start_date, end_date):
    """
    전체 종목 날짜 범위 분석 (Historical Backfill)

    Args:
        db_manager: AsyncDatabaseManager 인스턴스
        start_date: 시작 날짜 (date object)
        end_date: 종료 날짜 (date object)

    Returns:
        dict: 날짜별 성공/실패 통계
    """
    try:
        from datetime import timedelta

        # Get all dates with data in the range
        date_query = """
        SELECT DISTINCT date
        FROM kr_intraday_total
        WHERE date >= $1 AND date <= $2
        ORDER BY date
        """
        date_result = await db_manager.execute_query(date_query, start_date, end_date)
        dates = [row['date'] for row in date_result]

        if not dates:
            print(f"\n[ERROR] {start_date} ~ {end_date} 기간의 데이터가 없습니다.\n")
            return None

        print("\n" + "="*80)
        print("날짜 범위 분석 시작")
        print("="*80)
        print(f"기간: {start_date} ~ {end_date}")
        print(f"분석 대상 날짜: {len(dates)}일")
        print(f"배치 크기: 40개 (동시 처리)")
        print("="*80 + "\n")

        # Suppress verbose logging
        original_levels = {}
        loggers_to_suppress = [
            'root',
            '__main__',
            'kr_main',
            'weight',
            'kr_value_factor',
            'kr_quality_factor',
            'kr_momentum_factor',
            'kr_growth_factor',
            'kr_additional_metrics',
            'kr_interpretation',
            'batch_weight'
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

        for date_idx, analysis_date in enumerate(dates, 1):
            date_start_time = time.time()

            # Get symbols that have data on this date
            symbol_query = """
            SELECT DISTINCT symbol
            FROM kr_intraday_total
            WHERE date = $1
            ORDER BY symbol
            """
            symbol_result = await db_manager.execute_query(symbol_query, analysis_date)
            symbols = [row['symbol'] for row in symbol_result]
            symbol_count = len(symbols)

            # Step 1: Use AsyncBatchWeightCalculator for this date
            print(f"[날짜 {date_idx}/{len(dates)}] {analysis_date} - 가중치 계산 중...")
            batch_calc = AsyncBatchWeightCalculator()
            await batch_calc.initialize()

            batch_results = await batch_calc.process_all_stocks(symbols)

            await batch_calc.close()

            # Step 2: Process stocks with precomputed weights
            BATCH_SIZE = 20
            batches = [symbols[i:i + BATCH_SIZE] for i in range(0, symbol_count, BATCH_SIZE)]

            date_success = 0
            date_fail = 0

            for batch_idx, batch in enumerate(batches, 1):
                # Process batch concurrently with precomputed weights
                tasks = []
                for symbol in batch:
                    if symbol in batch_results:
                        precomputed_weights = batch_results[symbol]['weights']
                        precomputed_conditions = batch_results[symbol]['conditions']
                        tasks.append(
                            analyze_single_stock(
                                symbol,
                                db_manager,
                                analysis_date,
                                verbose=False,
                                precomputed_weights=precomputed_weights,
                                precomputed_conditions=precomputed_conditions
                            )
                        )
                    else:
                        # Fallback: no precomputed data
                        tasks.append(analyze_single_stock(symbol, db_manager, analysis_date, verbose=False))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Count successes and failures
                batch_success = 0
                batch_fail = 0
                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        date_fail += 1
                        batch_fail += 1
                    elif isinstance(result, bool) and result:
                        date_success += 1
                        batch_success += 1
                    else:
                        date_fail += 1
                        batch_fail += 1

                # Print batch progress
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"  [{current_time}] [배치 {batch_idx}/{len(batches)}] 완료: {batch_success}개 | 실패: {batch_fail}개")

                # Small delay between batches (Windows semaphore stabilization)
                if batch_idx < len(batches):
                    await asyncio.sleep(0.3)

            # Log date progress (concise)
            print(f"[날짜 {date_idx}/{len(dates)}] {analysis_date} | "
                  f"종목: {symbol_count:,}개 | "
                  f"성공: {date_success}개 | 실패: {date_fail}개")

            date_results[str(analysis_date)] = {
                'success': date_success,
                'fail': date_fail,
                'total': symbol_count
            }

            total_success += date_success
            total_fail += date_fail
            total_processed += symbol_count

        total_time = time.time() - total_start_time

        # Restore original log levels
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)

        print("\n" + "="*80)
        print("날짜 범위 분석 완료")
        print("="*80)
        print(f"분석 기간: {start_date} ~ {end_date}")
        print(f"전체 날짜: {len(dates)}일")
        print(f"전체 분석: {total_processed:,}건")
        print(f"성공: {total_success:,}개 ({(total_success/total_processed*100):.1f}%)")
        print(f"실패: {total_fail:,}개")
        print(f"총 소요시간: {total_time/60:.1f}분 ({total_time:.0f}초)")
        print(f"평균 시간: {total_time/len(dates):.1f}초/일")
        print("="*80 + "\n")

        return {
            'dates': len(dates),
            'total_processed': total_processed,
            'total_success': total_success,
            'total_fail': total_fail,
            'total_time': total_time,
            'date_results': date_results
        }

    except Exception as e:
        logger.error(f"날짜 범위 분석 실패: {e}", exc_info=True)
        return None


async def analyze_sample_stocks_specific_dates(db_manager, sample_file='sample_symbols.txt'):
    """
    샘플 종목 특정 날짜 분석 (Option 3 Validation)

    Args:
        db_manager: AsyncDatabaseManager 인스턴스
        sample_file: 샘플 종목코드 파일 경로

    Returns:
        dict: 날짜별 성공/실패 통계
    """
    try:
        import os

        # Specific dates to analyze
        analysis_dates = [
            date(2025, 8, 1),
            date(2025, 8, 6),
            date(2025, 8, 12),
            date(2025, 8, 21),
            date(2025, 8, 27),
            date(2025, 9, 1),
            date(2025, 9, 5),
            date(2025, 9, 10),
            date(2025, 9, 16)
        ]

        # Get the absolute path (kr_main.py is in kr/ directory, sample file is in parent directory)
        if not os.path.isabs(sample_file):
            # If relative path, look in parent directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sample_file = os.path.join(os.path.dirname(current_dir), sample_file)

        # Read sample symbols from file
        if not os.path.exists(sample_file):
            print(f"\n[ERROR] 샘플 파일을 찾을 수 없습니다: {sample_file}\n")
            return None

        with open(sample_file, 'r', encoding='utf-8') as f:
            symbols = [line.strip() for line in f if line.strip()]

        if not symbols:
            print(f"\n[ERROR] 샘플 파일에 종목이 없습니다: {sample_file}\n")
            return None

        print("\n" + "="*80)
        print("샘플 종목 특정 날짜 분석 (Option 3 Validation)")
        print("="*80)
        print(f"샘플 파일: {sample_file}")
        print(f"샘플 종목: {len(symbols):,}개")
        print(f"분석 날짜: {len(analysis_dates)}일")
        print(f"총 분석 건수: {len(symbols) * len(analysis_dates):,}건")
        print(f"DB 연결 풀: max_size=55 (miss.py와 공유 고려)")
        print(f"배치 크기: 45개 (동시 처리)")
        print("="*80 + "\n")

        # Suppress verbose logging
        original_levels = {}
        loggers_to_suppress = [
            'root',
            '__main__',
            'kr_main',
            'weight',
            'kr_value_factor',
            'kr_quality_factor',
            'kr_momentum_factor',
            'kr_growth_factor',
            'kr_additional_metrics',
            'kr_interpretation',
            'batch_weight'
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

        for date_idx, analysis_date in enumerate(analysis_dates, 1):
            date_start_time = time.time()

            # Step 1: Use AsyncBatchWeightCalculator for this date
            print(f"[날짜 {date_idx}/{len(analysis_dates)}] {analysis_date} - 가중치 계산 중...")
            batch_calc = AsyncBatchWeightCalculator()
            await batch_calc.initialize()

            batch_results = await batch_calc.process_all_stocks(symbols)

            await batch_calc.close()

            # Step 2: Process stocks with precomputed weights
            # Optimized for DB max_connections=100 (miss.py uses 30, kr_main uses 45)
            # Each analyze_single_stock uses ~3-5 connections sequentially
            # BATCH_SIZE=40 for stable throughput (45 pool / ~1.1 connections per stock)
            BATCH_SIZE = 20
            batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]

            date_success = 0
            date_fail = 0

            for batch_idx, batch in enumerate(batches, 1):
                # Process batch concurrently with precomputed weights
                tasks = []
                for symbol in batch:
                    if symbol in batch_results:
                        precomputed_weights = batch_results[symbol]['weights']
                        precomputed_conditions = batch_results[symbol]['conditions']
                        tasks.append(
                            analyze_single_stock(
                                symbol,
                                db_manager,
                                analysis_date,
                                verbose=False,
                                precomputed_weights=precomputed_weights,
                                precomputed_conditions=precomputed_conditions
                            )
                        )
                    else:
                        # Fallback: no precomputed data
                        tasks.append(analyze_single_stock(symbol, db_manager, analysis_date, verbose=False))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Count successes and failures
                batch_success = 0
                batch_fail = 0
                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        date_fail += 1
                        batch_fail += 1
                    elif isinstance(result, bool) and result:
                        date_success += 1
                        batch_success += 1
                    else:
                        date_fail += 1
                        batch_fail += 1

                # Print batch progress
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"  [{current_time}] [배치 {batch_idx}/{len(batches)}] 완료: {batch_success}개 | 실패: {batch_fail}개")

                # Small delay between batches (Windows semaphore stabilization)
                if batch_idx < len(batches):
                    await asyncio.sleep(0.3)

            # Log date progress (concise)
            date_time = time.time() - date_start_time
            print(f"[날짜 {date_idx}/{len(analysis_dates)}] {analysis_date} | "
                  f"종목: {len(symbols):,}개 | "
                  f"성공: {date_success}개 | 실패: {date_fail}개 | "
                  f"소요: {date_time:.1f}초")

            date_results[str(analysis_date)] = {
                'success': date_success,
                'fail': date_fail,
                'total': len(symbols)
            }

            total_success += date_success
            total_fail += date_fail
            total_processed += len(symbols)

        total_time = time.time() - total_start_time

        # Restore original log levels
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)

        print("\n" + "="*80)
        print("샘플 종목 분석 완료")
        print("="*80)
        print(f"샘플 종목: {len(symbols):,}개")
        print(f"분석 날짜: {len(analysis_dates)}일")
        print(f"전체 분석: {total_processed:,}건")
        print(f"성공: {total_success:,}개 ({(total_success/total_processed*100):.1f}%)")
        print(f"실패: {total_fail:,}개")
        print(f"총 소요시간: {total_time/60:.1f}분 ({total_time:.0f}초)")
        print(f"평균 시간: {total_time/len(analysis_dates):.1f}초/일")
        print("="*80 + "\n")

        return {
            'dates': len(analysis_dates),
            'symbols': len(symbols),
            'total_processed': total_processed,
            'total_success': total_success,
            'total_fail': total_fail,
            'total_time': total_time,
            'date_results': date_results
        }

    except Exception as e:
        logger.error(f"샘플 종목 분석 실패: {e}", exc_info=True)
        return None


async def analyze_all_stocks_specific_dates(db_manager, date_list):
    """
    전체 종목 특정 날짜 분석 (Option 5 - Phase 3.4)

    Args:
        db_manager: AsyncDatabaseManager 인스턴스
        date_list: 분석할 날짜 리스트 (date objects)

    Returns:
        dict: 날짜별 성공/실패 통계
    """
    try:
        print("\n" + "="*80)
        print("전체 종목 특정 날짜 분석 (Option 5)")
        print("="*80)
        print(f"분석 날짜: {len(date_list)}일")
        for idx, d in enumerate(date_list, 1):
            print(f"  {idx}. {d}")
        print(f"DB 연결 풀: max_size=45 (Pro 플랜 최적화)")
        print(f"배치 크기: 40개 (동시 처리)")
        print("="*80 + "\n")

        # Suppress verbose logging
        original_levels = {}
        loggers_to_suppress = [
            'root',
            '__main__',
            'kr_main',
            'weight',
            'kr_value_factor',
            'kr_quality_factor',
            'kr_momentum_factor',
            'kr_growth_factor',
            'kr_additional_metrics',
            'kr_interpretation',
            'batch_weight'
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
        total_symbols_all_dates = 0

        for date_idx, analysis_date in enumerate(date_list, 1):
            date_start_time = time.time()

            # Refresh sector performance data for this date (MV optimization)
            # Prevents memory overflow in calculate_sector_rotation_signal()
            # Memory: 5GB -> 10MB (500x reduction), Speed: 5-10s -> 0.01s (1000x faster)
            print(f"[날짜 {date_idx}/{len(date_list)}] {analysis_date} - 섹터 데이터 갱신 중...")
            await db_manager.refresh_sector_performance_for_date(analysis_date)

            # Get all symbols that have data on this date
            print(f"[날짜 {date_idx}/{len(date_list)}] {analysis_date} - 종목 조회 중...")
            symbol_query = """
            SELECT DISTINCT symbol
            FROM kr_intraday_total
            WHERE date = $1
            ORDER BY symbol
            """
            symbol_result = await db_manager.execute_query(symbol_query, analysis_date)
            symbols = [row['symbol'] for row in symbol_result]

            if not symbols:
                print(f"  [WARNING] {analysis_date}에 데이터가 있는 종목이 없습니다. 건너뜁니다.\n")
                continue

            print(f"  -> {len(symbols):,}개 종목 발견")
            total_symbols_all_dates += len(symbols)

            # Step 1: Use AsyncBatchWeightCalculator for this date
            print(f"[날짜 {date_idx}/{len(date_list)}] {analysis_date} - 가중치 계산 중...")
            batch_calc = AsyncBatchWeightCalculator()
            await batch_calc.initialize()

            batch_results = await batch_calc.process_all_stocks(symbols)

            await batch_calc.close()

            # Step 2: Process stocks with precomputed weights
            # Optimized for Pro plan: DB max_connections=100, batch size=40
            BATCH_SIZE = 20
            batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]

            date_success = 0
            date_fail = 0

            for batch_idx, batch in enumerate(batches, 1):
                # Process batch concurrently with precomputed weights
                tasks = []
                for symbol in batch:
                    if symbol in batch_results:
                        precomputed_weights = batch_results[symbol]['weights']
                        precomputed_conditions = batch_results[symbol]['conditions']
                        tasks.append(
                            analyze_single_stock(
                                symbol,
                                db_manager,
                                analysis_date,
                                verbose=False,
                                precomputed_weights=precomputed_weights,
                                precomputed_conditions=precomputed_conditions
                            )
                        )
                    else:
                        # Fallback: no precomputed data
                        tasks.append(analyze_single_stock(symbol, db_manager, analysis_date, verbose=False))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Count successes and failures
                batch_success = 0
                batch_fail = 0
                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        date_fail += 1
                        batch_fail += 1
                    elif isinstance(result, bool) and result:
                        date_success += 1
                        batch_success += 1
                    else:
                        date_fail += 1
                        batch_fail += 1

                # Print batch progress
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"  [{current_time}] [배치 {batch_idx}/{len(batches)}] 완료: {batch_success}개 | 실패: {batch_fail}개")

                # Small delay between batches (Windows semaphore stabilization)
                if batch_idx < len(batches):
                    await asyncio.sleep(0.3)

            # Log date progress
            date_time = time.time() - date_start_time
            print(f"[날짜 {date_idx}/{len(date_list)}] {analysis_date} | "
                  f"종목: {len(symbols):,}개 | "
                  f"성공: {date_success}개 | 실패: {date_fail}개 | "
                  f"소요: {date_time:.1f}초\n")

            date_results[str(analysis_date)] = {
                'success': date_success,
                'fail': date_fail,
                'total': len(symbols)
            }

            total_success += date_success
            total_fail += date_fail
            total_processed += len(symbols)

        total_time = time.time() - total_start_time

        # Restore original log levels
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)

        print("\n" + "="*80)
        print("전체 종목 특정 날짜 분석 완료")
        print("="*80)
        print(f"분석 날짜: {len(date_list)}일")
        print(f"전체 종목: {total_symbols_all_dates:,}개 (중복 포함)")
        print(f"전체 분석: {total_processed:,}건")
        print(f"성공: {total_success:,}개 ({(total_success/total_processed*100) if total_processed > 0 else 0:.1f}%)")
        print(f"실패: {total_fail:,}개")
        print(f"총 소요시간: {total_time/60:.1f}분 ({total_time:.0f}초)")
        print(f"평균 시간: {total_time/len(date_list):.1f}초/일" if len(date_list) > 0 else "N/A")
        print("="*80 + "\n")

        return {
            'dates': len(date_list),
            'total_processed': total_processed,
            'total_success': total_success,
            'total_fail': total_fail,
            'total_time': total_time,
            'date_results': date_results
        }

    except Exception as e:
        logger.error(f"전체 종목 특정 날짜 분석 실패: {e}", exc_info=True)
        return None


async def main():
    """메인 실행 함수"""
    print("\n" + "="*80)
    print("한국 주식 퀀트 분석 시스템 (Korean Stock Quant Analysis System)")
    print("="*80 + "\n")

    # Initialize database connection
    db_manager = AsyncDatabaseManager()
    await db_manager.initialize()

    try:
        while True:
            print("분석 옵션을 선택하세요:")
            print("  1. 단일 종목 분석")
            print("  2. 전체 종목 분석 (최신 날짜)")
            print("  3. 전체 종목 날짜 범위 분석 (예: 시작일 2025-01-01, 종료일 2025-01-31 입력)")
            print("  4. 샘플 종목 특정 날짜 분석 (sample_symbols.txt, 9개 날짜)")
            print("  5. 전체 종목 특정 날짜 분석 (날짜 직접 입력)")
            print()

            choice = input("선택 (1-5): ").strip()

            if choice == '1':
                symbol = input("\n종목코드를 입력하세요: ").strip()
                if not symbol:
                    print("종목코드가 입력되지 않았습니다.\n")
                    continue

                print("\n분석할 날짜를 입력하세요")
                print("사용법:")
                print("  - 개별 날짜: 2025-08-01, 2025-08-05")
                print("  - 날짜 범위: 2025-08-01~2025-08-05 (1일부터 5일까지)")
                print("  - 혼합: 2025-08-01, 2025-08-05~2025-08-08, 2025-09-04")
                print("  - 엔터만 입력하면 최신 날짜로 분석합니다.")
                dates_input = input("\n날짜 입력: ").strip()

                # 날짜 입력이 없으면 기존 방식대로 최신 날짜로 분석
                if not dates_input:
                    await analyze_single_stock(symbol, db_manager)
                    print("\n분석이 완료되었습니다. 프로그램을 종료합니다.")
                    break

                # 날짜 파싱 (개별 날짜 + 범위 지원)
                try:
                    from datetime import datetime, timedelta

                    # 쉼표로 분리
                    segments = [seg.strip() for seg in dates_input.split(',')]
                    date_list = []

                    for segment in segments:
                        # 범위 표시(~)가 있는지 확인
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

                            # 범위 내 모든 날짜 추가
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

                    # 날짜 정렬
                    date_list.sort()

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

                    # 각 날짜별로 분석 실행
                    print("\n" + "="*80)
                    print(f"종목 {symbol} 다중 날짜 분석 시작")
                    print("="*80)

                    success_count = 0
                    for idx, analysis_date in enumerate(date_list, 1):
                        print(f"\n[{idx}/{len(date_list)}] {analysis_date} 분석 중...")
                        print("-"*80)

                        try:
                            success = await analyze_single_stock(
                                symbol,
                                db_manager,
                                analysis_date=analysis_date,
                                verbose=True
                            )
                            if success:
                                success_count += 1
                        except Exception as e:
                            print(f"  ❌ 오류 발생: {str(e)}")
                            continue

                    print("\n" + "="*80)
                    print(f"분석 완료: {success_count}/{len(date_list)} 성공")
                    print("="*80)
                    print("\n프로그램을 종료합니다.")
                    break

                except ValueError as e:
                    print(f"잘못된 날짜 형식입니다. YYYY-MM-DD 형식으로 입력해주세요. (오류: {str(e)})\n")
                    continue

            elif choice == '2':
                confirm = input("\n전체 종목 분석을 시작하시겠습니까? (y/n): ").strip().lower()
                if confirm == 'y':
                    await analyze_all_stocks(db_manager)
                    break
                else:
                    print("취소되었습니다.\n")

            elif choice == '3':
                print("\n날짜 범위 입력 (형식: YYYY-MM-DD)")
                start_date_str = input("시작 날짜: ").strip()
                end_date_str = input("종료 날짜: ").strip()

                try:
                    from datetime import datetime
                    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

                    if start_date > end_date:
                        print("시작 날짜가 종료 날짜보다 늦습니다.\n")
                        continue

                    await analyze_all_stocks_date_range(db_manager, start_date, end_date)
                    break
                except ValueError:
                    print("잘못된 날짜 형식입니다. YYYY-MM-DD 형식으로 입력해주세요.\n")

            elif choice == '4':
                print("\n샘플 종목 특정 날짜 분석")
                print("분석 날짜: 2025-08-01, 08-06, 08-12, 08-21, 08-27, 09-01, 09-05, 09-10, 09-16")
                confirm = input("샘플 종목 분석을 시작하시겠습니까? (y/n): ").strip().lower()
                if confirm == 'y':
                    # Optimize connection pool for maximum performance
                    # DB max_connections=100, miss.py uses 30 connections
                    # Allocate 55 connections for kr_main.py (45 reserved: miss.py 30 + system 15)
                    print("\n연결 풀 최적화 중 (max_connections=45, Railway Pro 최적화)...")
                    print("us_main.py와 동시 실행 지원: kr_main=45, us_main=45 (총 90/100)")
                    await db_manager.close()
                    db_manager = AsyncDatabaseManager()
                    await db_manager.initialize(min_size=15, max_size=45)

                    await analyze_sample_stocks_specific_dates(db_manager)
                    break
                else:
                    print("취소되었습니다.\n")

            elif choice == '5':
                print("\n전체 종목 특정 날짜 분석")
                print("분석할 날짜를 입력하세요")
                print("사용법:")
                print("  - 개별 날짜: 2025-08-01, 2025-08-05")
                print("  - 날짜 범위: 2025-08-01~2025-08-05 (1일부터 5일까지)")
                print("  - 혼합: 2025-08-01, 2025-08-05~2025-08-08, 2025-09-04")
                dates_input = input("\n날짜 입력: ").strip()

                if not dates_input:
                    print("날짜가 입력되지 않았습니다.\n")
                    continue

                # 날짜 파싱 (개별 날짜 + 범위 지원)
                try:
                    from datetime import datetime, timedelta

                    # 쉼표로 분리
                    segments = [seg.strip() for seg in dates_input.split(',')]
                    date_list = []

                    for segment in segments:
                        # 범위 표시(~)가 있는지 확인
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

                            # 범위 내 모든 날짜 추가
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

                    # 날짜 정렬
                    date_list.sort()

                    if not date_list:
                        print("유효한 날짜가 없습니다.\n")
                        continue

                    print(f"\n입력된 날짜: {len(date_list)}일")
                    for idx, d in enumerate(date_list, 1):
                        print(f"  {idx}. {d}")

                    confirm = input("\n분석을 시작하시겠습니까? (y/n): ").strip().lower()
                    if confirm == 'y':
                        # Optimize connection pool for maximum performance (same as option 4)
                        # DB max_connections=100, miss.py uses 30 connections
                        # Allocate 55 connections for kr_main.py
                        print("\n연결 풀 최적화 중 (max_connections=45, Railway Pro 최적화)...")
                        print("us_main.py와 동시 실행 지원: kr_main=45, us_main=45 (총 90/100)")
                        await db_manager.close()
                        db_manager = AsyncDatabaseManager()
                        await db_manager.initialize(min_size=15, max_size=45)

                        await analyze_all_stocks_specific_dates(db_manager, date_list)
                        break
                    else:
                        print("취소되었습니다.\n")

                except ValueError as e:
                    print(f"잘못된 날짜 형식입니다. YYYY-MM-DD 형식으로 입력해주세요. (오류: {e})\n")
                except Exception as e:
                    print(f"오류 발생: {e}\n")

            else:
                print("잘못된 선택입니다. 다시 선택해주세요.\n")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")

    except Exception as e:
        logger.error(f"오류 발생: {e}")
        print(f"\n오류 발생: {e}")

    finally:
        await db_manager.close()
        print("\n데이터베이스 연결이 종료되었습니다.")


if __name__ == "__main__":
    asyncio.run(main())
