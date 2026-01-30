from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import uuid
import logging

from database import get_db
from models import BrandLogo, User
from auth import get_current_user
from ml_pipeline.vision import BrandDetector

router = APIRouter()
logger = logging.getLogger(__name__)
detector = BrandDetector()

@router.post("/brands")
async def add_brand_logo(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a new protected brand logo."""
    content = await file.read()
    
    # Compute features
    phash = detector.compute_phash(content)
    orb_features = detector.compute_orb_features(content)
    
    brand = BrandLogo(
        id=uuid.uuid4(),
        name=name,
        phash=phash,
        orb_features=orb_features
    )
    
    db.add(brand)
    await db.commit()
    await db.refresh(brand)
    
    return {"message": f"Brand {name} protected successfully", "id": brand.id}

@router.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Analyze an image for brand impersonation."""
    content = await file.read()
    
    # Get all known brands
    result = await db.execute(select(BrandLogo))
    brands = result.scalars().all()
    known_brands_data = [
        {"name": b.name, "phash": b.phash, "orb_features": b.orb_features}
        for b in brands
    ]
    
    result = detector.detect_brand(content, known_brands_data)
    
    return result

from sqlalchemy import select

@router.get("/brands")
async def list_brands(
    db: Session = Depends(get_db)
):
    """List all protected brands."""
    result = await db.execute(select(BrandLogo))
    brands = result.scalars().all()
    return [{"id": b.id, "name": b.name, "created_at": b.created_at} for b in brands]
