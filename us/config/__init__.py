"""
US Quant Configuration Package

Phase 3.1 configuration modules for weight adjustments and risk management.
"""

from .weight_adjustments import (
    EXCHANGE_ADJUSTMENTS,
    SECTOR_ADJUSTMENTS,
    ORS_THRESHOLDS,
    ORS_RISK_LEVELS,
    ORS_PRICE_SCORES,
    ORS_VOLUME_SCORES,
    ORS_VOLATILITY_SCORES,
    ORS_MKTCAP_SCORES,
    get_exchange_adjustment,
    get_sector_adjustment,
    get_risk_level,
    get_position_multiplier,
    calculate_adjusted_weights,
)

__all__ = [
    'EXCHANGE_ADJUSTMENTS',
    'SECTOR_ADJUSTMENTS',
    'ORS_THRESHOLDS',
    'ORS_RISK_LEVELS',
    'ORS_PRICE_SCORES',
    'ORS_VOLUME_SCORES',
    'ORS_VOLATILITY_SCORES',
    'ORS_MKTCAP_SCORES',
    'get_exchange_adjustment',
    'get_sector_adjustment',
    'get_risk_level',
    'get_position_multiplier',
    'calculate_adjusted_weights',
]
