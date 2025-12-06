"""
Scenario Probability Calibrator for Korean Stock Analysis (Phase 3.9)

Purpose: Fix systematic underprediction of Bullish/Bearish scenarios
- Bullish underprediction: +5.5~13.8%p error correction
- Bearish severe underprediction: +20.1%p error correction

Based on phase3_9 analysis results (2025-08-04 ~ 2025-09-25)

File: kr/kr_probability_calibrator.py
"""

import logging
from typing import Dict, Optional
from datetime import date

logger = logging.getLogger(__name__)


class ScenarioCalibrator:
    """
    Calibrate scenario probabilities based on empirical analysis

    Methodology:
    1. Apply empirical calibration map (from phase3_9 analysis)
    2. Adjust by market regime (GREED/NEUTRAL/FEAR/PANIC)
    3. Add theme risk premium (high-risk themes)
    4. Apply bearish probability floor (score-based)
    5. Normalize to 100%
    """

    # Empirical calibration mapping table (from phase3_9 CSV analysis)
    CALIBRATION_MAP = {
        'bullish': {
            (0, 20): 1.55,    # 10% predicted -> 15.5% actual
            (20, 40): 1.32,   # 30% predicted -> 39.6% actual
            (40, 60): 1.28,   # 50% predicted -> 63.8% actual
            (60, 80): 1.23,   # 70% predicted -> 85.9% actual
            (80, 100): 1.15
        },
        'bearish': {
            (0, 20): 3.01,    # 10% predicted -> 30.1% actual (3x!)
            (20, 40): 1.48,   # 30% predicted -> 44.3% actual
            (40, 60): 1.39,   # 50% predicted -> 69.7% actual
            (60, 80): 1.26,   # 70% predicted -> 88.5% actual
            (80, 100): 1.20
        }
    }

    # Market regime adjustment coefficients
    REGIME_ADJUSTMENT = {
        'OVERHEATED': {'bullish_boost': 1.2, 'bearish_reduce': 0.85},
        'GREED': {'bullish_boost': 1.3, 'bearish_reduce': 0.8},
        'NEUTRAL': {'bullish_boost': 1.0, 'bearish_reduce': 1.0},
        'FEAR': {'bullish_boost': 0.7, 'bearish_reduce': 1.2},
        'PANIC': {'bullish_boost': 0.5, 'bearish_reduce': 1.4}
    }

    # High-risk themes (from phase3_9 theme analysis)
    HIGH_RISK_THEMES = ['Telecom_Media', 'IT_Software', 'Consumer_Goods']
    THEME_RISK_PREMIUM = 15  # bearish +15%p

    # Bearish probability floor (score-based)
    BEARISH_FLOOR = {
        (0, 20): 25,     # Low score: minimum 25% bearish probability
        (20, 35): 20,
        (35, 45): 15,
        (45, 60): 12,
        (60, 100): 8     # High score: minimum 8% bearish probability
    }

    def __init__(self, db_manager=None):
        """
        Initialize scenario calibrator

        Args:
            db_manager: Database manager instance (optional, for market regime lookup)
        """
        self.db_manager = db_manager

    def calibrate_probabilities(
        self,
        raw_bullish: int,
        raw_bearish: int,
        raw_sideways: int,
        final_score: float,
        theme: str
    ) -> Dict[str, int]:
        """
        Calibrate raw probabilities to corrected probabilities

        Args:
            raw_bullish: Raw bullish probability (0-100)
            raw_bearish: Raw bearish probability (0-100)
            raw_sideways: Raw sideways probability (0-100)
            final_score: Stock final score (0-100)
            theme: Stock theme classification

        Returns:
            Dict with calibrated probabilities: {'bullish': int, 'sideways': int, 'bearish': int}
        """
        try:
            # Step 1: Apply empirical calibration map
            cal_bullish = self._apply_calibration_map('bullish', raw_bullish)
            cal_bearish = self._apply_calibration_map('bearish', raw_bearish)

            # Step 2: Get market regime and apply adjustment
            regime = self._get_market_regime()
            cal_bullish, cal_bearish = self._apply_regime_adjustment(
                cal_bullish, cal_bearish, regime
            )

            # Step 3: Apply theme risk premium (bearish only)
            cal_bearish = self._apply_theme_risk(cal_bearish, theme)

            # Step 4: Apply bearish floor
            cal_bearish = self._apply_bearish_floor(cal_bearish, final_score)

            # Step 5: Normalize to 100%
            calibrated = self._normalize(cal_bullish, raw_sideways, cal_bearish)

            logger.debug(f"Calibration: raw=({raw_bullish}, {raw_sideways}, {raw_bearish}) -> "
                        f"cal=({calibrated['bullish']}, {calibrated['sideways']}, {calibrated['bearish']}) "
                        f"[regime={regime}, theme={theme}]")

            return calibrated

        except Exception as e:
            logger.error(f"Calibration failed: {e}, returning raw probabilities")
            return {
                'bullish': raw_bullish,
                'sideways': raw_sideways,
                'bearish': raw_bearish
            }

    def _apply_calibration_map(self, scenario: str, raw_prob: int) -> float:
        """
        Apply empirical calibration multiplier

        Args:
            scenario: 'bullish' or 'bearish'
            raw_prob: Raw probability (0-100)

        Returns:
            Calibrated probability (float)
        """
        if scenario not in self.CALIBRATION_MAP:
            return float(raw_prob)

        mapping = self.CALIBRATION_MAP[scenario]

        for (low, high), multiplier in mapping.items():
            if low <= raw_prob < high:
                return raw_prob * multiplier

        # Default: no adjustment
        return float(raw_prob)

    def _get_market_regime(self) -> str:
        """
        Get current market regime from market sentiment analysis

        Returns:
            Regime: 'OVERHEATED', 'GREED', 'NEUTRAL', 'FEAR', 'PANIC'

        Note:
            Uses KOSPI sentiment as proxy
            Fallback to 'NEUTRAL' if data unavailable
        """
        if not self.db_manager:
            return 'NEUTRAL'

        try:
            # Query market_index for KOSPI recent performance
            # Classify into regime based on change_rate
            query = """
            SELECT change_rate
            FROM market_index
            WHERE exchange = 'KOSPI'
            ORDER BY date DESC
            LIMIT 5
            """

            result = self.db_manager.execute_query_sync(query) if hasattr(self.db_manager, 'execute_query_sync') else None

            if not result or len(result) < 5:
                return 'NEUTRAL'

            # Calculate 5-day average change
            avg_change = sum(float(r['change_rate']) for r in result) / len(result)

            # Classify regime
            if avg_change > 2.0:
                return 'OVERHEATED'
            elif avg_change > 0.5:
                return 'GREED'
            elif avg_change > -0.5:
                return 'NEUTRAL'
            elif avg_change > -2.0:
                return 'FEAR'
            else:
                return 'PANIC'

        except Exception as e:
            logger.warning(f"Failed to get market regime: {e}, using NEUTRAL")
            return 'NEUTRAL'

    def _apply_regime_adjustment(
        self,
        bullish: float,
        bearish: float,
        regime: str
    ) -> tuple:
        """
        Apply market regime adjustment

        Args:
            bullish: Calibrated bullish probability
            bearish: Calibrated bearish probability
            regime: Market regime

        Returns:
            Tuple of (adjusted_bullish, adjusted_bearish)
        """
        adjustment = self.REGIME_ADJUSTMENT.get(regime, {'bullish_boost': 1.0, 'bearish_reduce': 1.0})

        adj_bullish = bullish * adjustment['bullish_boost']
        adj_bearish = bearish * adjustment['bearish_reduce']

        return adj_bullish, adj_bearish

    def _apply_theme_risk(self, bearish: float, theme: str) -> float:
        """
        Apply theme risk premium to bearish probability

        Args:
            bearish: Bearish probability
            theme: Stock theme

        Returns:
            Adjusted bearish probability
        """
        if theme in self.HIGH_RISK_THEMES:
            return bearish + self.THEME_RISK_PREMIUM
        return bearish

    def _apply_bearish_floor(self, bearish: float, final_score: float) -> float:
        """
        Apply minimum bearish probability floor

        Args:
            bearish: Bearish probability
            final_score: Stock final score

        Returns:
            Bearish probability with floor applied
        """
        for (low, high), floor in self.BEARISH_FLOOR.items():
            if low <= final_score < high:
                return max(bearish, floor)

        # Default floor for high scores
        return max(bearish, 8)

    def _normalize(self, bullish: float, sideways: float, bearish: float) -> Dict[str, int]:
        """
        Normalize probabilities to sum to 100%

        Args:
            bullish: Bullish probability
            sideways: Sideways probability (raw)
            bearish: Bearish probability

        Returns:
            Dict with normalized integer probabilities
        """
        total = bullish + sideways + bearish

        if total == 0:
            return {'bullish': 33, 'sideways': 34, 'bearish': 33}

        # Normalize
        norm_bullish = round(bullish / total * 100)
        norm_bearish = round(bearish / total * 100)
        norm_sideways = 100 - norm_bullish - norm_bearish

        # Ensure sideways is non-negative
        if norm_sideways < 0:
            norm_sideways = 0
            norm_bullish = round(bullish / (bullish + bearish) * 100)
            norm_bearish = 100 - norm_bullish

        return {
            'bullish': norm_bullish,
            'sideways': norm_sideways,
            'bearish': norm_bearish
        }


# Validation test
if __name__ == '__main__':
    print("Scenario Calibrator Validation Test")
    print("=" * 80)

    calibrator = ScenarioCalibrator()

    # Test case 1: Low score, high-risk theme
    print("\nTest 1: Low score (15), high-risk theme (Telecom_Media)")
    result = calibrator.calibrate_probabilities(
        raw_bullish=10,
        raw_bearish=10,
        raw_sideways=80,
        final_score=15,
        theme='Telecom_Media'
    )
    print(f"  Raw: Bullish=10%, Sideways=80%, Bearish=10%")
    print(f"  Calibrated: Bullish={result['bullish']}%, Sideways={result['sideways']}%, Bearish={result['bearish']}%")
    print(f"  Expected: Bearish significantly increased (floor 25% + theme risk 15% + calibration 3x)")

    # Test case 2: High score, normal theme
    print("\nTest 2: High score (70), normal theme (Semiconductor)")
    result = calibrator.calibrate_probabilities(
        raw_bullish=50,
        raw_bearish=10,
        raw_sideways=40,
        final_score=70,
        theme='Semiconductor'
    )
    print(f"  Raw: Bullish=50%, Sideways=40%, Bearish=10%")
    print(f"  Calibrated: Bullish={result['bullish']}%, Sideways={result['sideways']}%, Bearish={result['bearish']}%")
    print(f"  Expected: Bullish increased (~64%), Bearish minimum floor (8%)")

    # Test case 3: Medium score, medium probabilities
    print("\nTest 3: Medium score (40), normal theme")
    result = calibrator.calibrate_probabilities(
        raw_bullish=30,
        raw_bearish=30,
        raw_sideways=40,
        final_score=40,
        theme='Electronics'
    )
    print(f"  Raw: Bullish=30%, Sideways=40%, Bearish=30%")
    print(f"  Calibrated: Bullish={result['bullish']}%, Sideways={result['sideways']}%, Bearish={result['bearish']}%")
    print(f"  Expected: Both adjusted by calibration map")

    print("\n" + "=" * 80)
    print("Validation complete. Check if calibrated probabilities are reasonable.")
