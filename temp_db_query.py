import asyncio
import asyncpg
import sys
sys.stdout.reconfigure(encoding='utf-8')

async def main():
    conn = await asyncpg.connect('postgresql://postgres:KoHtrdVEltzXlLVcgYtnRbEBQIRfEhMv@switchback.proxy.rlwy.net:28289/railway')

    # 1. market_index 데이터 범위 확인
    r1 = await conn.fetch('''
        SELECT exchange, MIN(date) as min_dt, MAX(date) as max_dt, COUNT(*) as cnt
        FROM market_index
        GROUP BY exchange
        ORDER BY exchange
    ''')
    print('=== Market Index Data ===')
    for row in r1:
        print(f"  {row['exchange']}: {row['min_dt']} ~ {row['max_dt']}, {row['cnt']} records")

    # 2. kr_stock_detail.theme 값 분포 확인
    r2 = await conn.fetch('''
        SELECT theme, COUNT(*) as cnt
        FROM kr_stock_detail
        WHERE theme IS NOT NULL AND theme != ''
        GROUP BY theme
        ORDER BY cnt DESC
        LIMIT 20
    ''')
    print('\n=== Theme Distribution ===')
    for row in r2:
        print(f"  {row['theme']}: {row['cnt']} stocks")

    # 3. 문제 종목 분석: 통신/미디어 섹터의 변동성 패턴
    r3 = await conn.fetch('''
        WITH telecom_stocks AS (
            SELECT symbol FROM kr_stock_detail
            WHERE theme ILIKE '%Telecom%' OR theme ILIKE '%Media%'
        ),
        volatility_data AS (
            SELECT
                t.symbol,
                i.date,
                i.rsi,
                t2.change_rate,
                t2.close,
                AVG(t2.close) OVER (PARTITION BY t.symbol ORDER BY i.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20
            FROM telecom_stocks t
            JOIN kr_indicators i ON t.symbol = i.symbol
            JOIN kr_intraday_total t2 ON t.symbol = t2.symbol AND i.date = t2.date
            WHERE i.date >= '2025-08-01'
        )
        SELECT
            symbol,
            AVG(rsi) as avg_rsi,
            STDDEV(change_rate) as volatility,
            AVG(CASE WHEN close > ma20 THEN 1 ELSE 0 END) as pct_above_ma20,
            COUNT(*) as days
        FROM volatility_data
        GROUP BY symbol
        HAVING COUNT(*) >= 20
        ORDER BY volatility DESC
        LIMIT 10
    ''')
    print('\n=== Telecom/Media Volatility Analysis ===')
    for row in r3:
        print(f"  {row['symbol']}: RSI={row['avg_rsi']:.1f}, Vol={row['volatility']:.2f}%, Above MA20={row['pct_above_ma20']:.1%}")

    # 4. Idiosyncratic Momentum 계산 샘플 (시장+섹터 효과 제거)
    r4 = await conn.fetch('''
        WITH stock_returns AS (
            SELECT
                s.symbol,
                s.date,
                s.change_rate as stock_return,
                d.theme,
                d.exchange
            FROM kr_intraday_total s
            JOIN kr_stock_detail d ON s.symbol = d.symbol
            WHERE s.date >= '2025-08-01' AND s.date <= '2025-09-30'
        ),
        market_returns AS (
            SELECT date, exchange, change_rate as market_return
            FROM market_index
            WHERE date >= '2025-08-01' AND date <= '2025-09-30'
        ),
        theme_returns AS (
            SELECT date, theme, AVG(stock_return) as theme_return
            FROM stock_returns
            GROUP BY date, theme
        ),
        idiosyncratic AS (
            SELECT
                sr.symbol,
                sr.theme,
                AVG(sr.stock_return - COALESCE(mr.market_return, 0) - COALESCE(tr.theme_return, 0) + sr.stock_return) as idio_momentum,
                AVG(sr.stock_return) as raw_momentum,
                STDDEV(sr.stock_return) as volatility
            FROM stock_returns sr
            LEFT JOIN market_returns mr ON sr.date = mr.date AND sr.exchange = mr.exchange
            LEFT JOIN theme_returns tr ON sr.date = tr.date AND sr.theme = tr.theme
            GROUP BY sr.symbol, sr.theme
            HAVING COUNT(*) >= 30
        )
        SELECT symbol, theme, idio_momentum, raw_momentum, volatility
        FROM idiosyncratic
        WHERE theme ILIKE '%Telecom%' OR theme ILIKE '%Media%'
        ORDER BY raw_momentum DESC
        LIMIT 10
    ''')
    print('\n=== Idiosyncratic vs Raw Momentum (Telecom/Media) ===')
    print(f"  {'Symbol':<10} {'Theme':<20} {'Raw Mom':<10} {'Idio Mom':<10} {'Vol':<8}")
    for row in r4:
        theme_short = (row['theme'] or 'N/A')[:18]
        print(f"  {row['symbol']:<10} {theme_short:<20} {row['raw_momentum']:>8.2f}% {row['idio_momentum']:>8.2f}% {row['volatility']:>6.2f}%")

    await conn.close()
    print('\n=== Query Complete ===')

asyncio.run(main())
