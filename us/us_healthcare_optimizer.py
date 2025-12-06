"""
US Healthcare Sector Optimizer

Data-driven optimization for Healthcare sector based on us_phase2_healthcare_*.csv analysis.

Key findings:
- Healthcare IC: 0.119 (40% below average)
- Biotech IC: -0.001 (55% of samples, ineffective)
- Medical Services IC: 0.114+ (traditional factors work)
- Growth factor: only effective factor (IC 0.019)

Strategy:
- Growth-centric approach
- Sub-sector differentiation (Big Pharma / Biotech / Medical Services / Medical Devices)
- Adjust factor weights by sub-sector

Target: Healthcare IC 0.119 -> 0.160 (+34% improvement)

File: us/us_healthcare_optimizer.py
"""

import logging
from typing import Dict, Optional, Tuple
from decimal import Decimal
from datetime import date

logger = logging.getLogger(__name__)


class USHealthcareOptimizer:
    """
    Healthcare sector optimization

    Approach:
    1. Classify healthcare stocks into 4 sub-sectors
    2. Apply differentiated factor weights per sub-sector
    3. Growth-centric strategy (only effective factor in healthcare)
    """

    def __init__(self):
        # SIC code definitions
        self.MEDICAL_SERVICES_SIC = [8000, 8011, 8050, 8051, 8060, 8062, 8071, 5122]
        self.MEDICAL_DEVICES_SIC = [3841, 3842, 3843, 3844, 3845]

        # Classification thresholds
        self.BIG_PHARMA_MARKET_CAP = 50_000_000_000  # 50B USD
        self.BIG_PHARMA_RD_RATIO = 0.20  # 20%
        self.BIOTECH_REVENUE = 500_000_000  # 500M USD

    async def classify_healthcare_subsector(
        self,
        stock_info: Dict,
        symbol: str,
        db,
        analysis_date: date
    ) -> str:
        """
        Classify healthcare stock into sub-sector

        Logic:
        1. SIC code based: Medical Services (8000, 8011, 8050, 8051, 8060, 8062, 8071, 5122)
        2. SIC code based: Medical Devices (3841-3845)
        3. Market cap > 50B & R&D ratio < 20% -> Big Pharma
        4. R&D ratio >= 20% OR revenue < 500M -> Biotech
        5. Otherwise -> Unknown

        Args:
            stock_info: Stock basic information (market_cap, sic, etc.)
            symbol: Stock symbol
            db: Database connection
            analysis_date: Analysis date

        Returns:
            Sub-sector: 'Medical Services' / 'Medical Devices' / 'Big Pharma' / 'Biotech' / 'Unknown'
        """

        # Get SIC code and market cap from stock_info
        sic = stock_info.get('sic')
        market_cap = float(stock_info.get('market_cap', 0)) if stock_info.get('market_cap') else 0

        # 1. Medical Services (high IC: 0.114+)
        if sic in self.MEDICAL_SERVICES_SIC:
            logger.debug(f"[{symbol}] Classified as Medical Services (SIC: {sic})")
            return 'Medical Services'

        # 2. Medical Devices (IC: 0.015)
        if sic in self.MEDICAL_DEVICES_SIC:
            logger.debug(f"[{symbol}] Classified as Medical Devices (SIC: {sic})")
            return 'Medical Devices'

        # Fetch financial data for R&D and revenue
        try:
            query = """
                SELECT revenue, research_development, total_assets
                FROM us_financial_quarterly
                WHERE symbol = $1 AND fiscal_date <= $2
                ORDER BY fiscal_date DESC
                LIMIT 1
            """
            result = await db.execute_query(query, symbol, analysis_date)

            if not result or len(result) == 0:
                logger.debug(f"[{symbol}] No financial data found, classified as Unknown")
                return 'Unknown'

            row = result[0]
            revenue = float(row['revenue']) if row['revenue'] else 0
            rd = float(row['research_development']) if row['research_development'] else 0

            # Calculate R&D ratio
            rd_ratio = rd / revenue if revenue > 0 else 0

            # 3. Big Pharma (IC: -0.021, need Growth boost)
            # Large market cap + low R&D ratio = mature pharma company
            if market_cap > self.BIG_PHARMA_MARKET_CAP and rd_ratio < self.BIG_PHARMA_RD_RATIO:
                logger.debug(f"[{symbol}] Classified as Big Pharma (Market cap: {market_cap/1e9:.1f}B, R&D: {rd_ratio:.1%})")
                return 'Big Pharma'

            # 4. Biotech (IC: -0.001, need maximum Growth weight)
            # High R&D ratio OR small revenue = biotech company
            if rd_ratio >= self.BIG_PHARMA_RD_RATIO or revenue < self.BIOTECH_REVENUE:
                logger.debug(f"[{symbol}] Classified as Biotech (R&D: {rd_ratio:.1%}, Revenue: {revenue/1e6:.1f}M)")
                return 'Biotech'

            # 5. Unknown (fallback)
            logger.debug(f"[{symbol}] Classified as Unknown (fallback)")
            return 'Unknown'

        except Exception as e:
            logger.warning(f"[{symbol}] Error classifying healthcare subsector: {e}")
            return 'Unknown'

    def get_healthcare_weights(self, subsector: str) -> Dict[str, float]:
        """
        Get factor weights by healthcare sub-sector

        Data evidence (us_phase2_healthcare_by_industry.csv):
        - Medical Services IC: 0.114 (traditional factors work)
        - Medical Devices IC: 0.015 (Growth-centric)
        - Big Pharma IC: -0.021 (Growth boost needed)
        - Biotech IC: -0.001 (Growth maximization needed)

        CRITICAL: Weight sum MUST be 1.0 to maintain 100-point scale

        Args:
            subsector: Sub-sector name

        Returns:
            Dict[str, float]: Factor weights in 0-1 range, sum = 1.0
            Example: {'growth': 0.60, 'quality': 0.25, 'momentum': 0.10, 'value': 0.05}
        """

        weights = {
            # Medical Services: IC 0.114 (traditional factors work)
            'Medical Services': {
                'growth': 0.35,      # Healthcare Growth IC: 0.019 (only effective factor)
                'quality': 0.30,     # Stable business model
                'momentum': 0.20,    # Medium weight
                'value': 0.15        # Traditional valuation works
            },  # Sum: 1.00

            # Medical Devices: IC 0.015 (Growth-centric)
            'Medical Devices': {
                'growth': 0.40,      # Boost Growth
                'quality': 0.30,     # Innovation quality matters
                'momentum': 0.15,    # Lower weight
                'value': 0.15        # Traditional valuation
            },  # Sum: 1.00

            # Big Pharma: IC -0.021 (Growth boost needed)
            'Big Pharma': {
                'growth': 0.50,      # Maximum Growth (traditional factors ineffective)
                'quality': 0.30,     # Pipeline quality
                'momentum': 0.10,    # Minimize (low IC)
                'value': 0.10        # Minimize (low IC)
            },  # Sum: 1.00

            # Biotech: IC -0.001 (Growth maximization)
            # Growth factor is the ONLY effective factor in biotech
            'Biotech': {
                'growth': 0.60,      # Maximize Growth (60% weight)
                'quality': 0.25,     # Cash burn stability
                'momentum': 0.10,    # Minimize (ineffective)
                'value': 0.05        # Minimize (ineffective)
            },  # Sum: 1.00

            # Unknown: Conservative approach
            'Unknown': {
                'growth': 0.50,      # Growth-centric (safe default)
                'quality': 0.25,     # Balance
                'momentum': 0.15,    # Lower weight
                'value': 0.10        # Lower weight
            }  # Sum: 1.00
        }

        selected_weights = weights.get(subsector, weights['Unknown'])

        # Validation: Weight sum must be 1.0 (critical for 100-point scale)
        weight_sum = sum(selected_weights.values())
        assert abs(weight_sum - 1.0) < 0.001, \
            f"Weight sum must be 1.0, got {weight_sum} for subsector '{subsector}'"

        return selected_weights

    def calculate_healthcare_score(
        self,
        factors: Dict[str, float],
        subsector: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate healthcare optimized score

        Args:
            factors: Factor scores dict
                {'growth_score': 80, 'quality_score': 70, 'momentum_score': 60, 'value_score': 50}
            subsector: Healthcare sub-sector

        Returns:
            Tuple[float, Dict[str, float]]: (score, weights_used)
                score: 0-100 range
                weights_used: Applied weights dict
        """

        weights = self.get_healthcare_weights(subsector)

        # Calculate weighted score
        score = (
            factors['growth_score'] * weights['growth'] +
            factors['quality_score'] * weights['quality'] +
            factors['momentum_score'] * weights['momentum'] +
            factors['value_score'] * weights['value']
        )

        # score should be in 0-100 range (since weights sum to 1.0 and factor scores are 0-100)

        return score, weights


# Validation test (run when module is imported)
if __name__ == '__main__':
    optimizer = USHealthcareOptimizer()

    print("Healthcare Weight Validation:")
    print("=" * 80)

    subsectors = ['Medical Services', 'Medical Devices', 'Big Pharma', 'Biotech', 'Unknown']

    for subsector in subsectors:
        weights = optimizer.get_healthcare_weights(subsector)
        weight_sum = sum(weights.values())

        print(f"\n[{subsector}]")
        print(f"  Growth:   {weights['growth']:.2f}")
        print(f"  Quality:  {weights['quality']:.2f}")
        print(f"  Momentum: {weights['momentum']:.2f}")
        print(f"  Value:    {weights['value']:.2f}")
        print(f"  Sum:      {weight_sum:.2f} {'OK' if abs(weight_sum - 1.0) < 0.001 else 'FAIL'}")

    print("\n" + "=" * 80)
    print("All weights sum to 1.0 - 100 point scale maintained [OK]")
