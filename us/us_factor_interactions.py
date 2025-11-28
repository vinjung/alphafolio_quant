# -*- coding: utf-8 -*-
"""
US Factor Interactions - Non-linear Factor Combination
======================================================

Phase 2 Analysis Result:
- Pearson IC: -0.097 (linear relationship negative)
- Spearman IC: +0.174 (rank relationship positive)
- Gap: 0.271

Conclusion: Non-linear relationship exists between score and return
Solution: Factor Interaction Model to capture non-linear effects

Interaction Terms:
- I1: Growth x Quality (growth + profitability synergy)
- I2: Growth x Momentum (growth + trend confirmation)
- I3: Quality x Value (high quality undervalued)
- I4: Momentum x Quality (trend backed by fundamentals)
- I5: All-Factor Agreement (Conviction Score)

File: us/us_factor_interactions.py
"""

import numpy as np
from typing import Dict, Optional


# Interaction Weights
INTERACTION_WEIGHTS = {
    'I1_growth_quality': 0.30,
    'I2_growth_momentum': 0.25,
    'I3_quality_value': 0.20,
    'I4_momentum_quality': 0.15,
    'I5_all_agreement': 0.10
}


class USFactorInteractions:
    """Factor Interaction Calculator for Non-linear Factor Combination"""

    def __init__(self, factor_scores: Dict[str, float]):
        """
        Initialize with factor scores

        Args:
            factor_scores: {
                'value_score': float (0-100),
                'quality_score': float (0-100),
                'momentum_score': float (0-100),
                'growth_score': float (0-100)
            }
        """
        self.scores = factor_scores
        self.value = float(factor_scores.get('value_score', 50) or 50)
        self.quality = float(factor_scores.get('quality_score', 50) or 50)
        self.momentum = float(factor_scores.get('momentum_score', 50) or 50)
        self.growth = float(factor_scores.get('growth_score', 50) or 50)

    def calculate(self) -> Dict:
        """
        Calculate Interaction Score and Conviction Score

        Returns:
            {
                'interaction_score': float (0-100),
                'conviction_score': float (0-100),
                'i1_growth_quality': float,
                'i2_growth_momentum': float,
                'i3_quality_value': float,
                'i4_momentum_quality': float,
                'i5_conviction': float,
                'interactions': Dict (detailed breakdown)
            }
        """
        interactions = {}

        # I1: Growth x Quality
        interactions['I1'] = self._calc_i1_growth_quality()

        # I2: Growth x Momentum
        interactions['I2'] = self._calc_i2_growth_momentum()

        # I3: Quality x Value
        interactions['I3'] = self._calc_i3_quality_value()

        # I4: Momentum x Quality
        interactions['I4'] = self._calc_i4_momentum_quality()

        # I5: All-Factor Agreement (Conviction Score)
        interactions['I5'] = self._calc_i5_all_agreement()

        # Weighted sum for interaction_score
        interaction_score = self._weighted_sum(interactions)

        return {
            'interaction_score': round(interaction_score, 2),
            'conviction_score': round(interactions['I5']['score'], 2),
            'i1_growth_quality': round(interactions['I1']['score'], 2),
            'i2_growth_momentum': round(interactions['I2']['score'], 2),
            'i3_quality_value': round(interactions['I3']['score'], 2),
            'i4_momentum_quality': round(interactions['I4']['score'], 2),
            'i5_conviction': round(interactions['I5']['score'], 2),
            'interactions': interactions
        }

    def _calc_i1_growth_quality(self) -> Dict:
        """
        I1: Growth x Quality Interaction

        Logic: Companies with both high growth and high profitability
        - Both high: Synergy bonus
        - One high: Neutral
        - Both low: Penalty

        Calculation: Geometric mean + synergy bonus
        """
        # Normalize to 0-1 range
        g_norm = self.growth / 100
        q_norm = self.quality / 100

        # Geometric mean (both need to be high for high score)
        geometric_mean = np.sqrt(g_norm * q_norm)

        # Synergy bonus: Both >= 70 gives additional score
        synergy_bonus = 0
        if self.growth >= 70 and self.quality >= 70:
            synergy_bonus = 0.1 * min((self.growth - 70) / 30, (self.quality - 70) / 30)

        raw_score = geometric_mean + synergy_bonus
        score = min(100, raw_score * 100)

        return {
            'score': round(score, 2),
            'raw': round(raw_score, 4),
            'growth': self.growth,
            'quality': self.quality,
            'synergy_bonus': round(synergy_bonus * 100, 2)
        }

    def _calc_i2_growth_momentum(self) -> Dict:
        """
        I2: Growth x Momentum Interaction

        Logic: Growth companies with price momentum
        - Fundamental growth + market recognition = strong signal

        Calculation: Harmonic mean (balanced consideration)
        """
        g_norm = self.growth / 100
        m_norm = self.momentum / 100

        # Harmonic mean (requires both to be reasonably high)
        if g_norm + m_norm > 0:
            harmonic_mean = 2 * g_norm * m_norm / (g_norm + m_norm)
        else:
            harmonic_mean = 0

        score = harmonic_mean * 100

        return {
            'score': round(score, 2),
            'raw': round(harmonic_mean, 4),
            'growth': self.growth,
            'momentum': self.momentum
        }

    def _calc_i3_quality_value(self) -> Dict:
        """
        I3: Quality x Value Interaction

        Logic: High quality but undervalued (traditional value investing)

        Calculation: Arithmetic mean + bonus if both >= 60
        """
        q_norm = self.quality / 100
        v_norm = self.value / 100

        # Arithmetic mean (either being high has meaning)
        arithmetic_mean = (q_norm + v_norm) / 2

        # Bonus if both >= 60
        bonus = 0
        if self.quality >= 60 and self.value >= 60:
            bonus = 0.05

        score = min(100, (arithmetic_mean + bonus) * 100)

        return {
            'score': round(score, 2),
            'raw': round(arithmetic_mean, 4),
            'quality': self.quality,
            'value': self.value,
            'bonus': round(bonus * 100, 2)
        }

    def _calc_i4_momentum_quality(self) -> Dict:
        """
        I4: Momentum x Quality Interaction

        Logic: Price momentum backed by fundamentals
        - High momentum + low quality: Risky (bubble potential) -> penalty
        - High momentum + high quality: Healthy rise -> reward

        Calculation: Geometric mean with penalty for momentum without quality
        """
        m_norm = self.momentum / 100
        q_norm = self.quality / 100

        # Penalty if momentum high but quality low (potential bubble)
        penalty = 0
        if self.momentum >= 70 and self.quality < 50:
            penalty = 0.1

        geometric_mean = np.sqrt(m_norm * q_norm)
        score = max(0, (geometric_mean - penalty) * 100)

        return {
            'score': round(score, 2),
            'raw': round(geometric_mean, 4),
            'momentum': self.momentum,
            'quality': self.quality,
            'penalty': round(penalty * 100, 2)
        }

    def _calc_i5_all_agreement(self) -> Dict:
        """
        I5: All-Factor Agreement (Conviction Score)

        Logic: Degree of agreement among all 4 factors
        - Lower std deviation = higher conviction (factors agree)
        - Higher average with low std = strong conviction for buy
        - Lower average with low std = strong conviction for sell

        Calculation:
        - Base conviction from std: lower std = higher conviction
        - Adjusted by average score direction
        """
        scores = [self.value, self.quality, self.momentum, self.growth]

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Conviction from std: lower std = higher conviction
        # std=0 -> 100, std=25 -> 0
        conviction_from_std = max(0, 100 - std_score * 4)

        # Average bonus: higher average = better for long conviction
        # avg 50 -> 0 bonus, avg 75 -> 12.5 bonus
        avg_bonus = max(0, (mean_score - 50) / 2)

        # Final conviction score
        # 70% from std agreement, 30% from average level
        conviction = min(100, conviction_from_std * 0.7 + avg_bonus * 0.3 + mean_score * 0.3)

        return {
            'score': round(conviction, 2),
            'raw': round(std_score, 4),
            'mean': round(mean_score, 2),
            'std': round(std_score, 2),
            'factor_scores': [round(s, 2) for s in scores]
        }

    def _weighted_sum(self, interactions: Dict) -> float:
        """Calculate weighted sum of interaction scores"""
        total = 0
        weights = {
            'I1': INTERACTION_WEIGHTS['I1_growth_quality'],
            'I2': INTERACTION_WEIGHTS['I2_growth_momentum'],
            'I3': INTERACTION_WEIGHTS['I3_quality_value'],
            'I4': INTERACTION_WEIGHTS['I4_momentum_quality'],
            'I5': INTERACTION_WEIGHTS['I5_all_agreement']
        }

        for key, weight in weights.items():
            if key in interactions and interactions[key]['score'] is not None:
                total += interactions[key]['score'] * weight

        return total


def calculate_conviction_score(value: float, quality: float,
                                momentum: float, growth: float) -> float:
    """
    Standalone function to calculate Conviction Score

    Args:
        value, quality, momentum, growth: Factor scores (0-100)

    Returns:
        conviction_score: 0-100
    """
    scores = [value, quality, momentum, growth]

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Base conviction from std
    base_conviction = max(0, 100 - std_score * 4)

    # Direction bonus: extreme average (high or low) gets bonus
    direction_bonus = abs(mean_score - 50) * 0.5

    conviction = min(100, base_conviction * 0.8 + direction_bonus)

    return round(conviction, 2)


# Test function
if __name__ == '__main__':
    print("=" * 60)
    print("US Factor Interactions - Test Cases")
    print("=" * 60)

    # Test case 1: All high scores (high conviction, high interaction)
    test1 = {
        'value_score': 75,
        'quality_score': 80,
        'momentum_score': 70,
        'growth_score': 85
    }

    calc1 = USFactorInteractions(test1)
    result1 = calc1.calculate()
    print("\nTest 1 - All High Scores:")
    print(f"  Input: V={test1['value_score']}, Q={test1['quality_score']}, "
          f"M={test1['momentum_score']}, G={test1['growth_score']}")
    print(f"  Interaction Score: {result1['interaction_score']}")
    print(f"  Conviction Score: {result1['conviction_score']}")
    print(f"  I1 (Growth x Quality): {result1['i1_growth_quality']}")
    print(f"  I2 (Growth x Momentum): {result1['i2_growth_momentum']}")
    print(f"  I3 (Quality x Value): {result1['i3_quality_value']}")
    print(f"  I4 (Momentum x Quality): {result1['i4_momentum_quality']}")

    # Test case 2: Mixed scores (low conviction)
    test2 = {
        'value_score': 30,
        'quality_score': 80,
        'momentum_score': 40,
        'growth_score': 90
    }

    calc2 = USFactorInteractions(test2)
    result2 = calc2.calculate()
    print("\nTest 2 - Mixed Scores (Low Conviction):")
    print(f"  Input: V={test2['value_score']}, Q={test2['quality_score']}, "
          f"M={test2['momentum_score']}, G={test2['growth_score']}")
    print(f"  Interaction Score: {result2['interaction_score']}")
    print(f"  Conviction Score: {result2['conviction_score']}")
    print(f"  Std Dev: {result2['interactions']['I5']['std']}")

    # Test case 3: High momentum, low quality (penalty case)
    test3 = {
        'value_score': 50,
        'quality_score': 35,
        'momentum_score': 85,
        'growth_score': 60
    }

    calc3 = USFactorInteractions(test3)
    result3 = calc3.calculate()
    print("\nTest 3 - High Momentum, Low Quality (Penalty):")
    print(f"  Input: V={test3['value_score']}, Q={test3['quality_score']}, "
          f"M={test3['momentum_score']}, G={test3['growth_score']}")
    print(f"  Interaction Score: {result3['interaction_score']}")
    print(f"  I4 (Momentum x Quality): {result3['i4_momentum_quality']}")
    print(f"  I4 Penalty Applied: {result3['interactions']['I4']['penalty']}")

    # Test case 4: All low scores (sell signal with high conviction)
    test4 = {
        'value_score': 30,
        'quality_score': 35,
        'momentum_score': 25,
        'growth_score': 40
    }

    calc4 = USFactorInteractions(test4)
    result4 = calc4.calculate()
    print("\nTest 4 - All Low Scores (Sell Signal):")
    print(f"  Input: V={test4['value_score']}, Q={test4['quality_score']}, "
          f"M={test4['momentum_score']}, G={test4['growth_score']}")
    print(f"  Interaction Score: {result4['interaction_score']}")
    print(f"  Conviction Score: {result4['conviction_score']}")
    print(f"  Mean: {result4['interactions']['I5']['mean']}, "
          f"Std: {result4['interactions']['I5']['std']}")

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)
