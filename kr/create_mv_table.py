"""
Execute create_mv_sector_performance.sql to create Materialized View
"""
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

async def create_mv_table():
    """Create mv_sector_daily_performance Materialized View"""
    try:
        # Get database URL
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL not found in environment variables")

        # Convert to asyncpg format if needed
        db_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

        # Connect to database
        conn = await asyncpg.connect(db_url)
        print("[OK] Connected to database")

        # Read SQL file
        sql_file_path = r"C:\project\alpha\quant\kr\sql\create_mv_sector_performance.sql"
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql = f.read()

        print(f"[OK] Read SQL file: {sql_file_path}")

        # Execute SQL
        print("\nExecuting SQL...")
        await conn.execute(sql)
        print("[OK] Materialized View created successfully!")

        # Verify creation
        verify_query = """
        SELECT COUNT(*) as table_exists
        FROM information_schema.tables
        WHERE table_name = 'mv_sector_daily_performance'
        """
        result = await conn.fetchrow(verify_query)

        if result['table_exists'] > 0:
            print("\n[OK] Verified: mv_sector_daily_performance table exists")
        else:
            print("\n[WARNING] Table not found after creation")

        # Check indexes
        index_query = """
        SELECT indexname
        FROM pg_indexes
        WHERE tablename = 'mv_sector_daily_performance'
        ORDER BY indexname
        """
        indexes = await conn.fetch(index_query)

        if indexes:
            print(f"\n[OK] Indexes created ({len(indexes)} total):")
            for idx in indexes:
                print(f"  - {idx['indexname']}")
        else:
            print("\n[WARNING] No indexes found")

        # Close connection
        await conn.close()
        print("\n[OK] Database connection closed")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise

if __name__ == "__main__":
    asyncio.run(create_mv_table())
