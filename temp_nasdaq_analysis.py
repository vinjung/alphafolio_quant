"""
NASDAQ Data Analysis for Model Improvement
Temporary script for data exploration
"""
import asyncio
import asyncpg

async def main():
    db_url = "postgresql://postgres:KoHtrdVEltzXlLVcgYtnRbEBQIRfEhMv@switchback.proxy.rlwy.net:28289/railway"
    conn = await asyncpg.connect(db_url)

    # 1. NASDAQ vs NYSE Distribution
    print("=" * 80)
    print("1. NASDAQ vs NYSE Stock Distribution")
    print("=" * 80)
    result = await conn.fetch("""
        SELECT exchange, sector, COUNT(*) as cnt,
               AVG(market_cap) as avg_mcap,
               AVG(beta) as avg_beta
        FROM us_stock_basic
        WHERE exchange IN ('NASDAQ', 'NYSE') AND sector IS NOT NULL
        GROUP BY exchange, sector
        ORDER BY exchange, cnt DESC
    """)

    current_exchange = ''
    for row in result:
        if row['exchange'] != current_exchange:
            current_exchange = row['exchange']
            print(f"\n[{current_exchange}]")
        avg_mcap_b = float(row['avg_mcap'])/1e9 if row['avg_mcap'] else 0
        avg_beta = float(row['avg_beta']) if row['avg_beta'] else 0
        print(f"  {row['sector']:<25}: {row['cnt']:>4} stocks, MCap: ${avg_mcap_b:>6.1f}B, Beta: {avg_beta:.2f}")

    # 2. NASDAQ Market Cap Distribution
    print("\n" + "=" * 80)
    print("2. NASDAQ Market Cap Distribution")
    print("=" * 80)
    result2 = await conn.fetch("""
        SELECT
            CASE
                WHEN market_cap >= 200000000000 THEN '1_Mega (>$200B)'
                WHEN market_cap >= 10000000000 THEN '2_Large ($10-200B)'
                WHEN market_cap >= 2000000000 THEN '3_Mid ($2-10B)'
                WHEN market_cap >= 300000000 THEN '4_Small ($300M-2B)'
                ELSE '5_Micro (<$300M)'
            END as size_tier,
            COUNT(*) as cnt,
            AVG(beta) as avg_beta
        FROM us_stock_basic
        WHERE exchange = 'NASDAQ' AND market_cap IS NOT NULL
        GROUP BY size_tier
        ORDER BY size_tier
    """)
    for row in result2:
        tier = row['size_tier'][2:]  # Remove prefix
        avg_beta = float(row['avg_beta']) if row['avg_beta'] else 0
        print(f"  {tier:<20}: {row['cnt']:>5} stocks, Avg Beta: {avg_beta:.2f}")

    # 3. IV Data Availability
    print("\n" + "=" * 80)
    print("3. Options IV Data Availability (NASDAQ)")
    print("=" * 80)
    result3 = await conn.fetch("""
        SELECT
            COUNT(DISTINCT o.symbol) as symbols_with_iv,
            COUNT(*) as total_records,
            MIN(o.date) as min_date,
            MAX(o.date) as max_date,
            AVG(o.avg_implied_volatility) as avg_iv
        FROM us_option_daily_summary o
        JOIN us_stock_basic b ON o.symbol = b.symbol
        WHERE b.exchange = 'NASDAQ'
    """)
    for row in result3:
        print(f"  Symbols with IV data: {row['symbols_with_iv']}")
        print(f"  Total IV records: {row['total_records']:,}")
        print(f"  Date range: {row['min_date']} ~ {row['max_date']}")
        avg_iv = float(row['avg_iv'])*100 if row['avg_iv'] else 0
        print(f"  Average IV: {avg_iv:.1f}%")

    # 4. 52-Week High Data
    print("\n" + "=" * 80)
    print("4. 52-Week High/Low Data Availability")
    print("=" * 80)
    result4 = await conn.fetch("""
        SELECT
            exchange,
            COUNT(*) as total,
            COUNT(week52high) as has_52w_high,
            COUNT(week52low) as has_52w_low,
            COUNT(beta) as has_beta
        FROM us_stock_basic
        WHERE exchange IN ('NASDAQ', 'NYSE')
        GROUP BY exchange
    """)
    for row in result4:
        print(f"  [{row['exchange']}] Total: {row['total']}, 52W High: {row['has_52w_high']}, 52W Low: {row['has_52w_low']}, Beta: {row['has_beta']}")

    # 5. NASDAQ Sector Beta Analysis
    print("\n" + "=" * 80)
    print("5. NASDAQ Beta Distribution by Sector")
    print("=" * 80)
    result5 = await conn.fetch("""
        SELECT
            sector,
            COUNT(*) as cnt,
            AVG(beta) as avg_beta,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY beta) as p25_beta,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY beta) as p75_beta
        FROM us_stock_basic
        WHERE exchange = 'NASDAQ' AND beta IS NOT NULL AND sector IS NOT NULL
        GROUP BY sector
        ORDER BY avg_beta DESC
    """)
    print(f"  {'Sector':<25} {'N':>5} {'Avg':>8} {'P25':>8} {'P75':>8}")
    print("  " + "-" * 60)
    for row in result5:
        avg_b = float(row['avg_beta']) if row['avg_beta'] else 0
        p25_b = float(row['p25_beta']) if row['p25_beta'] else 0
        p75_b = float(row['p75_beta']) if row['p75_beta'] else 0
        print(f"  {row['sector']:<25} {row['cnt']:>5} {avg_b:>7.2f} {p25_b:>7.2f} {p75_b:>7.2f}")

    # 6. Recent us_stock_grade data check
    print("\n" + "=" * 80)
    print("6. Recent NASDAQ Stock Grade Performance")
    print("=" * 80)
    result6 = await conn.fetch("""
        SELECT
            b.sector,
            COUNT(*) as cnt,
            AVG(g.value_score) as avg_value,
            AVG(g.quality_score) as avg_quality,
            AVG(g.momentum_score) as avg_momentum,
            AVG(g.growth_score) as avg_growth,
            AVG(g.final_score) as avg_final
        FROM us_stock_grade g
        JOIN us_stock_basic b ON g.symbol = b.symbol
        WHERE b.exchange = 'NASDAQ'
          AND g.date = (SELECT MAX(date) FROM us_stock_grade)
          AND b.sector IS NOT NULL
        GROUP BY b.sector
        ORDER BY avg_final DESC
    """)
    if result6:
        print(f"  {'Sector':<25} {'N':>4} {'Val':>6} {'Qual':>6} {'Mom':>6} {'Grow':>6} {'Final':>6}")
        print("  " + "-" * 65)
        for row in result6:
            print(f"  {row['sector']:<25} {row['cnt']:>4} {float(row['avg_value'] or 0):>5.1f} {float(row['avg_quality'] or 0):>5.1f} {float(row['avg_momentum'] or 0):>5.1f} {float(row['avg_growth'] or 0):>5.1f} {float(row['avg_final'] or 0):>5.1f}")
    else:
        print("  No recent grade data found")

    # 7. Daily data availability for volatility calculation
    print("\n" + "=" * 80)
    print("7. Daily Price Data Availability (for Volatility Calculation)")
    print("=" * 80)
    result7 = await conn.fetch("""
        SELECT
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(DISTINCT symbol) as symbol_count,
            COUNT(*) as total_records
        FROM us_daily
    """)
    for row in result7:
        print(f"  Date range: {row['min_date']} ~ {row['max_date']}")
        print(f"  Symbols: {row['symbol_count']:,}")
        print(f"  Total records: {row['total_records']:,}")

    await conn.close()
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
