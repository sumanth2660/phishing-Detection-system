"""
API Routes for Unified Phishing Detection System
Comprehensive endpoints for multi-modal phishing detection and analysis.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import asyncio
import uuid

from ml_pipeline.predict import MLPredictor
from ingestion.threat_intel import ThreatIntelManager
from auth import get_current_user, verify_token
from models import User, UserRole, DetectionResult, ThreatIntelligence
from database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

async def get_ml_predictor(request: Request):
    return request.app.state.ml_predictor

async def get_threat_intel(request: Request):
    return request.app.state.threat_intel

async def get_guest_user():
    return User(id=uuid.uuid4(), username="guest", role=UserRole.USER)

# Request/Response Models
class URLPredictionRequest(BaseModel):
    url: HttpUrl
    context: Optional[str] = None
    user_agent: Optional[str] = None

class EmailPredictionRequest(BaseModel):
    subject: str
    body: str
    sender: Optional[EmailStr] = None
    headers: Optional[Dict[str, str]] = None

class SMSPredictionRequest(BaseModel):
    text: str
    sender: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    id: str
    probability: float
    risk_level: str  # low, medium, high, critical
    confidence: float
    reasons: List[Dict[str, Any]]
    domain_details: Optional[Dict[str, Any]] = None
    explain_html: Optional[str] = None
    url: Optional[str] = None
    timestamp: datetime
    processing_time_ms: float
    feature_scores: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class ThreatIntelResponse(BaseModel):
    url: str
    verdict: str  # clean, suspicious, malicious
    sources: List[str]
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    tags: List[str]

class ReportRequest(BaseModel):
    detection_id: str
    include_details: bool = True
    format: str = "pdf"  # pdf, json

# Prediction Endpoints

@router.post("/predict/url", response_model=PredictionResponse)
async def predict_url(
    request: URLPredictionRequest,
    current_user: User = Depends(get_guest_user),
    ml_predictor: MLPredictor = Depends(get_ml_predictor)
):
    """
    Analyze URL for phishing indicators using ML models and heuristics.
    
    - **url**: The URL to analyze
    - **context**: Optional context about where the URL was found
    - **user_agent**: Optional user agent string for context
    """
    if not current_user:
        current_user = User(id=uuid.uuid4(), username="guest", role=UserRole.USER)
    try:
        start_time = datetime.utcnow()
        detection_id = str(uuid.uuid4())
        
        logger.info(f"Analyzing URL: {request.url} for user: {current_user.id}")
        
        # Perform prediction
        result = await ml_predictor.predict_url(
            url=str(request.url),
            context=request.context,
            user_agent=request.user_agent
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Determine risk level
        risk_level = "low"
        if result["probability"] > 0.8:
            risk_level = "critical"
        elif result["probability"] > 0.6:
            risk_level = "high"
        elif result["probability"] > 0.3:
            risk_level = "medium"
        
        response = PredictionResponse(
            id=detection_id,
            probability=result["probability"],
            risk_level=risk_level,
            confidence=result["confidence"],
            reasons=result["reasons"],
            domain_details=result.get("domain_details"),
            explain_html=result.get("explain_html"),
            timestamp=start_time,
            processing_time_ms=processing_time
        )
        
        # Store result in database (background task)
        # TODO: Implement database storage
        
        logger.info(f"URL analysis completed: {detection_id} - Risk: {risk_level}")
        return response
        
    except Exception as e:
        logger.error(f"URL prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/email", response_model=PredictionResponse)
async def predict_email(
    request: EmailPredictionRequest,
    current_user: User = Depends(get_guest_user),
    ml_predictor: MLPredictor = Depends(get_ml_predictor)
):
    """
    Analyze email for phishing indicators using NLP and heuristic analysis.
    
    - **subject**: Email subject line
    - **body**: Email body content (text or HTML)
    - **sender**: Optional sender email address
    - **headers**: Optional email headers for additional analysis
    """
    if not current_user:
        current_user = User(id=uuid.uuid4(), username="guest", role=UserRole.USER)
    try:
        start_time = datetime.utcnow()
        detection_id = str(uuid.uuid4())
        
        logger.info(f"Analyzing email from: {request.sender} for user: {current_user.id}")
        
        # Perform prediction
        result = await ml_predictor.predict_email(
            subject=request.subject,
            body=request.body,
            sender=request.sender,
            headers=request.headers
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Determine risk level
        risk_level = "low"
        if result["probability"] > 0.8:
            risk_level = "critical"
        elif result["probability"] > 0.6:
            risk_level = "high"
        elif result["probability"] > 0.3:
            risk_level = "medium"
        
        response = PredictionResponse(
            id=detection_id,
            probability=result["probability"],
            risk_level=risk_level,
            confidence=result["confidence"],
            reasons=result["reasons"],
            explain_html=result.get("explain_html"),
            timestamp=start_time,
            processing_time_ms=processing_time,
            feature_scores=result.get("feature_scores"),
            metadata=result.get("metadata")
        )
        
        logger.info(f"Email analysis completed: {detection_id} - Risk: {risk_level}")
        return response
        
    except Exception as e:
        logger.error(f"Email prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/sms", response_model=PredictionResponse)
async def predict_sms(
    request: SMSPredictionRequest,
    current_user: User = Depends(get_guest_user),
    ml_predictor: MLPredictor = Depends(get_ml_predictor)
):
    """
    Analyze SMS message for phishing indicators.
    
    - **text**: SMS message content
    - **sender**: Optional sender phone number or short code
    - **metadata**: Optional metadata (carrier, country, etc.)
    """
    if not current_user:
        current_user = User(id=uuid.uuid4(), username="guest", role=UserRole.USER)
    try:
        start_time = datetime.utcnow()
        detection_id = str(uuid.uuid4())
        
        logger.info(f"Analyzing SMS from: {request.sender} for user: {current_user.id}")
        
        # Perform prediction
        result = await ml_predictor.predict_sms(
            text=request.text,
            sender=request.sender,
            metadata=request.metadata
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Determine risk level
        risk_level = "low"
        if result["probability"] > 0.8:
            risk_level = "critical"
        elif result["probability"] > 0.6:
            risk_level = "high"
        elif result["probability"] > 0.3:
            risk_level = "medium"
        
        response = PredictionResponse(
            id=detection_id,
            probability=result["probability"],
            risk_level=risk_level,
            confidence=result["confidence"],
            reasons=result["reasons"],
            explain_html=result.get("explain_html"),
            timestamp=start_time,
            processing_time_ms=processing_time,
            feature_scores=result.get("feature_scores"),
            metadata=result.get("metadata")
        )
        
        logger.info(f"SMS analysis completed: {detection_id} - Risk: {risk_level}")
        return response
        
    except Exception as e:
        logger.error(f"SMS prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/image", response_model=List[PredictionResponse])
async def predict_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_guest_user),
    ml_predictor: MLPredictor = Depends(get_ml_predictor)
):
    """
    Analyze image for phishing indicators by extracting and analyzing URLs.
    
    - **file**: Image file (PNG, JPEG, GIF, etc.)
    """
    if not current_user:
        current_user = User(id=uuid.uuid4(), username="guest", role=UserRole.USER)
    try:
        start_time = datetime.utcnow()
        
        logger.info(f"Analyzing image: {file.filename} for user: {current_user.id}")
        
        # Read file content
        content = await file.read()
        
        # Perform prediction (returns list of results)
        results = await ml_predictor.predict_image(
            image_data=content,
            filename=file.filename
        )
        
        response_list = []
        for result in results:
            detection_id = str(uuid.uuid4())
            
            # Determine risk level
            risk_level = "low"
            if result.get("probability", 0) > 0.8:
                risk_level = "critical"
            elif result.get("probability", 0) > 0.6:
                risk_level = "high"
            elif result.get("probability", 0) > 0.3:
                risk_level = "medium"
            
            response = PredictionResponse(
                id=detection_id,
                probability=result.get("probability", 0),
                risk_level=risk_level,
                confidence=result.get("confidence", 0),
                reasons=result.get("reasons", []),
                domain_details=result.get("domain_details"),
                explain_html=result.get("explain_html"),
                url=result.get("url"),
                timestamp=start_time,
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            # Add URL to response if needed, but PredictionResponse doesn't have it.
            # We might need to add 'url' field to PredictionResponse or include it in explain_html/reasons
            # For now, let's assume the UI will use explain_html which contains the URL highlight
            
            response_list.append(response)
        
        logger.info(f"Image analysis completed: {len(response_list)} URLs found")
        return response_list
        
    except Exception as e:
        logger.error(f"Image prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/audio", response_model=PredictionResponse)
async def predict_audio(
    file: UploadFile = File(...),
    current_user: User = Depends(get_guest_user),
    ml_predictor: MLPredictor = Depends(get_ml_predictor)
):
    """
    Analyze audio file for phishing indicators using speech-to-text and NLP.
    
    - **file**: Audio file (WAV, MP3, M4A, etc.)
    """
    if not current_user:
        current_user = User(id=uuid.uuid4(), username="guest", role=UserRole.USER)
    try:
        start_time = datetime.utcnow()
        detection_id = str(uuid.uuid4())
        
        logger.info(f"Analyzing audio: {file.filename} for user: {current_user.id}")
        
        # Read file content
        content = await file.read()
        
        # Perform prediction
        result = await ml_predictor.predict_audio(
            audio_data=content,
            filename=file.filename
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Determine risk level
        risk_level = "low"
        if result["probability"] > 0.8:
            risk_level = "critical"
        elif result["probability"] > 0.6:
            risk_level = "high"
        elif result["probability"] > 0.3:
            risk_level = "medium"
        
        # Handle explicit errors from predictor
        if "error" in result:
             raise HTTPException(status_code=422, detail=result.get("details", result["error"]))

        response = PredictionResponse(
            id=detection_id,
            probability=result["probability"],
            risk_level=risk_level,
            confidence=result["confidence"],
            reasons=result["reasons"],
            explain_html=result.get("explain_html"),
            timestamp=start_time,
            processing_time_ms=processing_time,
            feature_scores=result.get("feature_scores")
        )
        
        logger.info(f"Audio analysis completed: {detection_id} - Risk: {risk_level}")
        return response
        
    except Exception as e:
        logger.error(f"Audio prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/document", response_model=List[PredictionResponse])
async def predict_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_guest_user),
    ml_predictor: MLPredictor = Depends(get_ml_predictor)
):
    """
    Analyze document (PDF, DOCX, TXT) for phishing indicators by extracting and analyzing URLs.
    
    - **file**: Document file
    """
    if not current_user:
        current_user = User(id=uuid.uuid4(), username="guest", role=UserRole.USER)
    try:
        start_time = datetime.utcnow()
        
        logger.info(f"Analyzing document: {file.filename} for user: {current_user.id}")
        
        # Read file content
        content = await file.read()
        
        # Perform prediction (returns list of results)
        results = await ml_predictor.predict_document(
            file_data=content,
            filename=file.filename
        )
        
        response_list = []
        for result in results:
            detection_id = str(uuid.uuid4())
            
            # Determine risk level
            risk_level = "low"
            if result.get("probability", 0) > 0.8:
                risk_level = "critical"
            elif result.get("probability", 0) > 0.6:
                risk_level = "high"
            elif result.get("probability", 0) > 0.3:
                risk_level = "medium"
            
            response = PredictionResponse(
                id=detection_id,
                probability=result.get("probability", 0),
                risk_level=risk_level,
                confidence=result.get("confidence", 0),
                reasons=result.get("reasons", []),
                domain_details=result.get("domain_details"),
                explain_html=result.get("explain_html"),
                url=result.get("url"),
                timestamp=start_time,
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            response_list.append(response)
        
        logger.info(f"Document analysis completed: {len(response_list)} URLs found")
        return response_list
        
    except Exception as e:
        logger.error(f"Document prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.websocket("/ws/scan")
async def websocket_scan(websocket: WebSocket):
    """WebSocket endpoint for real-time URL scanning from extension."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            url = data.get("url")
            
            if not url:
                await websocket.send_json({"error": "No URL provided"})
                continue
                
            # Get predictor from app state
            ml_predictor = websocket.app.state.ml_predictor
            
            # Perform prediction
            start_time = datetime.utcnow()
            result = await ml_predictor.predict_url(url, context="browser_extension")
            
            # Determine risk level
            risk_level = "low"
            if result.get("probability", 0) > 0.8:
                risk_level = "critical"
            elif result.get("probability", 0) > 0.6:
                risk_level = "high"
            elif result.get("probability", 0) > 0.3:
                risk_level = "medium"
            
            response = {
                "type": "scan_result",
                "url": url,
                "risk_level": risk_level,
                "probability": result.get("probability", 0),
                "reasons": result.get("reasons", []),
                "timestamp": start_time.isoformat()
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
        
    except Exception as e:
        logger.error(f"Document prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Threat Intelligence Endpoints

@router.get("/threatintel/check", response_model=ThreatIntelResponse)
async def check_threat_intel(
    url: str,
    current_user: User = Depends(get_current_user),
    threat_intel: ThreatIntelManager = Depends(get_threat_intel)
):
    """
    Check URL against threat intelligence feeds.
    
    - **url**: URL to check against threat databases
    """
    try:
        logger.info(f"Checking threat intel for: {url}")
        
        result = await threat_intel.check_url(url)
        
        return ThreatIntelResponse(
            url=url,
            verdict=result["verdict"],
            sources=result["sources"],
            first_seen=result.get("first_seen"),
            last_seen=result.get("last_seen"),
            tags=result.get("tags", [])
        )
        
    except Exception as e:
        logger.error(f"Threat intel check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Threat intel check failed: {str(e)}")

# Reporting Endpoints

@router.post("/reports/generate")
async def generate_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Generate forensic report for a detection result.
    
    - **detection_id**: ID of the detection to generate report for
    - **include_details**: Whether to include detailed analysis
    - **format**: Report format (pdf, json)
    """
    try:
        logger.info(f"Generating report for detection: {request.detection_id}")
        
        # Add background task to generate report
        background_tasks.add_task(
            generate_forensic_report,
            request.detection_id,
            request.include_details,
            request.format,
            current_user.id
        )
        
        return {
            "message": "Report generation started",
            "detection_id": request.detection_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.get("/reports/{report_id}")
async def download_report(
    report_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download generated forensic report."""
    try:
        # TODO: Implement report retrieval from storage
        report_path = f"reports/{report_id}.pdf"
        
        return FileResponse(
            path=report_path,
            filename=f"phishing_report_{report_id}.pdf",
            media_type="application/pdf"
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Report not found")
    except Exception as e:
        logger.error(f"Report download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report download failed: {str(e)}")

# Feedback Endpoints

@router.post("/feedback")
async def submit_feedback(
    detection_id: str,
    is_correct: bool,
    feedback_text: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Submit feedback on detection accuracy for model improvement.
    
    - **detection_id**: ID of the detection to provide feedback on
    - **is_correct**: Whether the detection was correct
    - **feedback_text**: Optional additional feedback
    """
    try:
        logger.info(f"Received feedback for detection: {detection_id}")
        
        # TODO: Store feedback in database
        # TODO: Queue for model retraining if needed
        
        return {
            "message": "Feedback received",
            "detection_id": detection_id,
            "status": "recorded"
        }
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

# Utility Functions

async def generate_forensic_report(detection_id: str, include_details: bool, format: str, user_id: str):
    """Background task to generate forensic report."""
    try:
        logger.info(f"Generating forensic report: {detection_id}")
        
        # TODO: Implement report generation logic
        # 1. Retrieve detection data
        # 2. Generate PDF/JSON report
        # 3. Store report file
        # 4. Notify user of completion
        
        logger.info(f"Forensic report generated: {detection_id}")
        
    except Exception as e:
        logger.error(f"Forensic report generation failed: {e}")