import asyncio
import asyncpg
import sys
sys.stdout.reconfigure(encoding='utf-8')

async def main():
    conn = await asyncpg.connect('postgresql://postgres:KoHtrdVEltzXlLVcgYtnRbEBQIRfEhMv@switchback.proxy.rlwy.net:28289/railway')

    # 1. 문제 종목들의 상세 분석 (phase3_10에서 언급된 종목들)
    problem_stocks = ['310200', '263700', '194480', '035900', '047820']

    r1 = await conn.fetch('''
        WITH stock_data AS (
            SELECT
                t.symbol,
                t.date,
                t.close,
                t.change_rate,
                t.volume,
                i.rsi,
                i.macd,
                i.obv,
                i.real_upper_band as bb_upper,
                i.real_middle_band as bb_mid,
                i.atr,
                LAG(t.close, 20) OVER (PARTITION BY t.symbol ORDER BY t.date) as close_20d_ago,
                AVG(t.close) OVER (PARTITION BY t.symbol ORDER BY t.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20,
                STDDEV(t.change_rate) OVER (PARTITION BY t.symbol ORDER BY t.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as volatility_20d
            FROM kr_intraday_total t
            JOIN kr_indicators i ON t.symbol = i.symbol AND t.date = i.date
            WHERE t.symbol = ANY($1::text[])
            AND t.date >= '2025-08-01' AND t.date <= '2025-09-30'
        )
        SELECT
            symbol,
            date,
            close,
            change_rate,
            rsi,
            ROUND((close / NULLIF(ma20, 0) - 1) * 100, 2) as ma20_deviation,
            ROUND((close / NULLIF(close_20d_ago, 0) - 1) * 100, 2) as return_20d,
            ROUND(volatility_20d, 2) as vol_20d,
            CASE WHEN close > bb_upper THEN 'ABOVE' WHEN close < bb_mid THEN 'BELOW' ELSE 'WITHIN' END as bb_position
        FROM stock_data
        WHERE date IN ('2025-08-11', '2025-08-22', '2025-09-09', '2025-09-12')
        ORDER BY symbol, date
    ''', problem_stocks)

    print('=== Problem Stocks Deep Analysis ===')
    print(f"{'Symbol':<8} {'Date':<12} {'Price':<10} {'RSI':<6} {'MA20Dev':<10} {'Ret20d':<10} {'Vol20d':<8} {'BB':<6}")
    for row in r1:
        print(f"{row['symbol']:<8} {str(row['date']):<12} {row['close']:<10} {row['rsi'] or 0:<6.1f} {row['ma20_deviation'] or 0:<10.1f}% {row['return_20d'] or 0:<10.1f}% {row['vol_20d'] or 0:<8.2f} {row['bb_position'] or 'N/A':<6}")

    # 2. Regime Detection: 시장 변동성 기반 모멘텀 크래시 예측
    r2 = await conn.fetch('''
        WITH market_vol AS (
            SELECT
                date,
                exchange,
                change_rate,
                STDDEV(change_rate) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as market_vol_20d,
                AVG(ABS(change_rate)) OVER (ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as avg_abs_ret_5d
            FROM market_index
            WHERE exchange = 'KOSPI'
            AND date >= '2025-07-01'
        )
        SELECT
            date,
            change_rate as market_return,
            ROUND(market_vol_20d, 3) as market_vol,
            ROUND(avg_abs_ret_5d, 3) as recent_turbulence,
            CASE
                WHEN market_vol_20d > 1.5 THEN 'HIGH_VOL'
                WHEN market_vol_20d > 1.0 THEN 'MED_VOL'
                ELSE 'LOW_VOL'
            END as vol_regime
        FROM market_vol
        WHERE date >= '2025-08-01'
        ORDER BY date DESC
        LIMIT 30
    ''')

    print('\n=== Market Volatility Regime (KOSPI) ===')
    print(f"{'Date':<12} {'Return':<10} {'Vol20d':<10} {'Turbulence':<12} {'Regime':<10}")
    for row in r2:
        print(f"{str(row['date']):<12} {row['market_return']:<10.2f}% {row['market_vol']:<10.3f} {row['recent_turbulence']:<12.3f} {row['vol_regime']:<10}")

    # 3. Volatility Scaling 효과 시뮬레이션
    r3 = await conn.fetch('''
        WITH stock_momentum AS (
            SELECT
                t.symbol,
                d.theme,
                AVG(t.change_rate) as avg_return,
                STDDEV(t.change_rate) as volatility,
                COUNT(*) as days
            FROM kr_intraday_total t
            JOIN kr_stock_detail d ON t.symbol = d.symbol
            WHERE t.date >= '2025-08-01' AND t.date <= '2025-09-30'
            AND d.theme IN ('Telecom_Media', 'Semiconductor', 'Bio_DrugRD')
            GROUP BY t.symbol, d.theme
            HAVING COUNT(*) >= 30 AND STDDEV(t.change_rate) > 0
        )
        SELECT
            theme,
            COUNT(*) as stock_count,
            ROUND(AVG(avg_return), 4) as avg_raw_momentum,
            ROUND(AVG(avg_return / volatility * 2), 4) as avg_vol_scaled_momentum,
            ROUND(AVG(volatility), 4) as avg_volatility,
            ROUND(STDDEV(avg_return), 4) as momentum_dispersion
        FROM stock_momentum
        GROUP BY theme
        ORDER BY avg_raw_momentum DESC
    ''')

    print('\n=== Volatility Scaling Effect by Theme ===')
    print(f"{'Theme':<20} {'Stocks':<8} {'Raw Mom':<12} {'VolScaled':<12} {'AvgVol':<10} {'Dispersion':<12}")
    for row in r3:
        print(f"{row['theme']:<20} {row['stock_count']:<8} {row['avg_raw_momentum']:<12.4f} {row['avg_vol_scaled_momentum']:<12.4f} {row['avg_volatility']:<10.4f} {row['momentum_dispersion']:<12.4f}")

    # 4. 기관/외국인 순매도 패턴 분석
    r4 = await conn.fetch('''
        SELECT
            t.symbol,
            d.theme,
            SUM(CASE WHEN i.inst_net_value < 0 THEN 1 ELSE 0 END) as inst_sell_days,
            SUM(CASE WHEN i.foreign_net_value < 0 THEN 1 ELSE 0 END) as foreign_sell_days,
            COUNT(*) as total_days,
            SUM(i.inst_net_value) as total_inst_flow,
            SUM(i.foreign_net_value) as total_foreign_flow
        FROM kr_individual_investor_daily_trading i
        JOIN kr_stock_detail d ON i.symbol = d.symbol
        JOIN kr_intraday_total t ON i.symbol = t.symbol AND i.date = t.date
        WHERE d.theme = 'Telecom_Media'
        AND i.date >= '2025-08-01' AND i.date <= '2025-09-30'
        GROUP BY t.symbol, d.theme
        HAVING COUNT(*) >= 20
        ORDER BY total_inst_flow + total_foreign_flow
        LIMIT 10
    ''')

    print('\n=== Institutional/Foreign Flow Analysis (Telecom_Media) ===')
    print(f"{'Symbol':<10} {'InstSellDays':<14} {'ForeignSellDays':<16} {'TotalDays':<12} {'InstFlow':<15} {'ForeignFlow':<15}")
    for row in r4:
        print(f"{row['symbol']:<10} {row['inst_sell_days']:<14} {row['foreign_sell_days']:<16} {row['total_days']:<12} {row['total_inst_flow'] or 0:<15,} {row['total_foreign_flow'] or 0:<15,}")

    await conn.close()
    print('\n=== Query Complete ===')

asyncio.run(main())
