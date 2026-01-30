"""
Unified Phishing Detection System - FastAPI Backend
Main application entry point with comprehensive phishing detection capabilities.
"""

from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import List, Optional
import asyncio
from datetime import datetime

# Internal imports
from api_routes import router as api_router
from routers import audio
from ml_pipeline.predict import MLPredictor
from ingestion.threat_intel import ThreatIntelManager
from database import init_db, get_db
from auth import verify_token, get_current_user
from models import User, DetectionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
ml_predictor = None
threat_intel = None
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting NeuroPhish AI Security System...")
    
    # Initialize database
    await init_db()
    logger.info("✅ Database initialized")
    
    # Initialize ML models
    ml_predictor = MLPredictor()
    await ml_predictor.load_models()
    app.state.ml_predictor = ml_predictor
    logger.info("✅ ML models loaded")
    
    # Initialize threat intelligence
    threat_intel = ThreatIntelManager()
    await threat_intel.initialize()
    app.state.threat_intel = threat_intel
    logger.info("✅ Threat intelligence initialized")
    
    # Start background tasks
    asyncio.create_task(background_model_update(app))
    logger.info("✅ Background tasks started")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Unified Phishing Detection System...")
    if hasattr(app.state, "ml_predictor"):
        await app.state.ml_predictor.cleanup()
    if hasattr(app.state, "threat_intel"):
        await app.state.threat_intel.cleanup()
    logger.info("✅ Cleanup completed")

# ... (skip to exception handlers)



# Create FastAPI application
app = FastAPI(
    title="NeuroPhish API",
    description="AI-powered multi-modal phishing detection with explainable results",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "chrome-extension://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")
app.include_router(audio.router, prefix="/api/v1/audio")
from routers import simulation, vision, ledger
app.include_router(simulation.router, prefix="/api/v1/simulation")
app.include_router(vision.router, prefix="/api/v1/vision")
app.include_router(ledger.router, prefix="/api/v1/ledger")

@app.get("/")
async def root():
    """Root endpoint with system status."""
    return {
        "message": "Unified Phishing Detection System API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "api": "/api/v1"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Check ML models
        ml_status = "healthy" if ml_predictor and ml_predictor.is_ready() else "unhealthy"
        
        # Check threat intelligence
        intel_status = "healthy" if threat_intel and threat_intel.is_ready() else "unhealthy"
        
        # Overall status
        overall_status = "healthy" if ml_status == "healthy" and intel_status == "healthy" else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "ml_models": ml_status,
                "threat_intelligence": intel_status,
                "database": "healthy",  # TODO: Add DB health check
                "cache": "healthy"      # TODO: Add Redis health check
            },
            "metrics": {
                "uptime_seconds": 0,  # TODO: Calculate actual uptime
                "total_predictions": 0,  # TODO: Get from database
                "accuracy": 0.95  # TODO: Get from model metrics
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/metrics")
async def get_metrics():
    """Prometheus-style metrics endpoint."""
    # TODO: Implement proper metrics collection
    return {
        "predictions_total": 0,
        "predictions_accuracy": 0.95,
        "response_time_seconds": 0.15,
        "active_users": 0,
        "threat_feeds_status": 1
    }

from pydantic import BaseModel
class DefenseRequest(BaseModel):
    url: str

@app.post("/api/v1/active-defense")
async def trigger_defense(request: DefenseRequest):
    """Trigger the Poison Pill active defense system against a URL."""
    logger.warning(f"⚔️ ACTIVE DEFENSE REQUESTED FOR: {request.url}")
    
    if not app.state.ml_predictor:
        raise HTTPException(status_code=503, detail="ML System not ready")
        
    result = await app.state.ml_predictor.execute_active_defense(request.url)
    return result

async def background_model_update(app: FastAPI):
    """Background task for periodic model updates."""
    while True:
        try:
            logger.info("🔄 Starting background model update check...")
            
            # Check if model update is needed
            if hasattr(app.state, "ml_predictor") and app.state.ml_predictor.needs_update():
                logger.info("📈 Updating ML models...")
                await app.state.ml_predictor.update_models()
                logger.info("✅ ML models updated successfully")
            
            # Update threat intelligence feeds
            if hasattr(app.state, "threat_intel"):
                logger.info("🔍 Updating threat intelligence feeds...")
                await app.state.threat_intel.update_feeds()
                logger.info("✅ Threat intelligence updated")
            
            # Wait 1 hour before next update
            await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"Background update failed: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes on error

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )