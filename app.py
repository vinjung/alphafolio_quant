"""
Quant Analysis API Server
Railway deployment FastAPI server for quant analysis system.
Provides HTTP endpoints for KR/US stock analysis triggered by upstream data service.
"""
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse

load_dotenv()

# Configure sys.path for kr/, us/ submodule imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KR_DIR = os.path.join(BASE_DIR, "kr")
US_DIR = os.path.join(BASE_DIR, "us")

for path in [BASE_DIR, KR_DIR, US_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ========================================
# API Key Authentication
# ========================================
def verify_api_key(x_api_key: str = Header(None, alias="X-API-KEY")) -> str:
    """Verify X-API-KEY header (same pattern as data service)"""
    expected_key = os.getenv("API_SECRET_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="API_SECRET_KEY not configured on server")
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-KEY header is required")
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ========================================
# FastAPI App
# ========================================
app = FastAPI(
    title="Quant Analysis API",
    description="Quant analysis system API for Railway deployment",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Quant Analysis API Server Starting")
    logger.info(f"Environment: {'Railway' if os.getenv('RAILWAY_ENVIRONMENT') else 'Local'}")
    logger.info(f"API Key configured: {bool(os.getenv('API_SECRET_KEY'))}")
    logger.info(f"Portfolio URL configured: {bool(os.getenv('PORTFOLIO_SERVICE_URL'))}")
    logger.info("=" * 60)


# ========================================
# Health Check
# ========================================
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "quant",
        "timestamp": datetime.now().isoformat()
    }


# ========================================
# KR Analysis Endpoint
# ========================================
@app.post("/kr/run")
async def kr_run(api_key: str = Depends(verify_api_key)):
    """
    Korean stock full analysis (kr_main.py option 1).
    - Analyzes all stocks for today's date
    - Runs KR Prediction Collector
    - Chains to portfolio service on success
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("POST /kr/run - Started")
    logger.info("=" * 60)

    try:
        from kr.kr_main import run_option1
        result = await run_option1()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"KR analysis completed in {duration:.2f}s")

        # Chain to portfolio service
        chain_result = await _call_portfolio_service("KR")

        return {
            "status": "success",
            "message": "KR analysis completed",
            "duration_seconds": duration,
            "analysis_result": result,
            "chain_portfolio": chain_result,
            "timestamp": end_time.isoformat()
        }

    except Exception as e:
        logger.error(f"KR analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"KR analysis failed: {str(e)}")


# ========================================
# US Analysis Endpoint
# ========================================
@app.post("/us/run")
async def us_run(api_key: str = Depends(verify_api_key)):
    """
    US stock full analysis (us_main.py option 1).
    - Auto-detects latest data date from us_daily table
    - Analyzes all stocks
    - Runs US Prediction Collector
    - Chains to portfolio service on success
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("POST /us/run - Started")
    logger.info("=" * 60)

    try:
        from us.us_main import run_option1
        result = await run_option1()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"US analysis completed in {duration:.2f}s")

        # Chain to portfolio service
        chain_result = await _call_portfolio_service("US")

        return {
            "status": "success",
            "message": "US analysis completed",
            "duration_seconds": duration,
            "analysis_result": result,
            "chain_portfolio": chain_result,
            "timestamp": end_time.isoformat()
        }

    except Exception as e:
        logger.error(f"US analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"US analysis failed: {str(e)}")


# ========================================
# Chaining Helper
# ========================================
async def _call_portfolio_service(country: str) -> dict:
    """Call portfolio service /recommend/daily endpoint"""
    portfolio_url = os.getenv("PORTFOLIO_SERVICE_URL")
    api_key = os.getenv("API_SECRET_KEY")

    if not portfolio_url:
        logger.warning("PORTFOLIO_SERVICE_URL not configured, skipping chain")
        return {"status": "skipped", "reason": "PORTFOLIO_SERVICE_URL not configured"}

    try:
        import httpx
        endpoint = f"{portfolio_url}/recommend/daily"
        logger.info(f"Chaining to portfolio service: {endpoint} (country={country})")

        async with httpx.AsyncClient(timeout=3600) as client:
            response = await client.post(
                endpoint,
                json={"country": country},
                headers={"X-API-KEY": api_key} if api_key else {}
            )
            response.raise_for_status()
            logger.info(f"Portfolio service ({country}) called successfully")
            return {"status": "success", "response": response.json()}

    except Exception as e:
        logger.error(f"Portfolio service call failed: {e}")
        return {"status": "failed", "error": str(e)}


# ========================================
# Entry Point
# ========================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
