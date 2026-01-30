from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, EmailStr
import uuid
from datetime import datetime
import logging

from database import get_db
from models import User, Simulation, SimulationTarget
from auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

# Request Models
class CreateSimulationRequest(BaseModel):
    name: str
    template_subject: str
    template_body: str
    target_emails: List[EmailStr]

class SimulationResponse(BaseModel):
    id: str
    name: str
    status: str
    created_at: datetime
    stats: dict

# Endpoints

@router.post("/create", response_model=SimulationResponse)
async def create_simulation(
    request: CreateSimulationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new phishing simulation campaign."""
    sim_id = str(uuid.uuid4())
    
    # Create simulation
    simulation = Simulation(
        id=sim_id,
        user_id=current_user.id,
        name=request.name,
        template_subject=request.template_subject,
        template_body=request.template_body,
        status="active"
    )
    db.add(simulation)
    
    # Add targets
    for email in request.target_emails:
        target = SimulationTarget(
            id=str(uuid.uuid4()),
            simulation_id=sim_id,
            email=email,
            status="pending"
        )
        db.add(target)
    
    db.commit()
    db.refresh(simulation)
    
    return {
        "id": simulation.id,
        "name": simulation.name,
        "status": simulation.status,
        "created_at": simulation.created_at,
        "stats": {"total": len(request.target_emails), "sent": 0, "clicked": 0}
    }

@router.post("/send/{simulation_id}")
async def send_simulation_emails(
    simulation_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Trigger sending emails for a campaign."""
    simulation = db.query(Simulation).filter(Simulation.id == simulation_id).first()
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
        
    targets = db.query(SimulationTarget).filter(
        SimulationTarget.simulation_id == simulation_id,
        SimulationTarget.status == "pending"
    ).all()
    
    # Mock sending in background
    background_tasks.add_task(mock_send_emails, targets, db)
    
    return {"message": f"Sending {len(targets)} emails in background"}

async def mock_send_emails(targets: List[SimulationTarget], db: Session):
    """Mock email sending logic."""
    import asyncio
    for target in targets:
        # Simulate delay
        await asyncio.sleep(0.5)
        
        target.status = "sent"
        target.sent_at = datetime.utcnow()
        db.commit()
        logger.info(f"Simulated email sent to {target.email}")

@router.get("/track/{target_id}")
async def track_click(target_id: str, db: Session = Depends(get_db)):
    """Tracking pixel/link endpoint."""
    target = db.query(SimulationTarget).filter(SimulationTarget.id == target_id).first()
    if target and target.status != "clicked":
        target.status = "clicked"
        target.clicked_at = datetime.utcnow()
        db.commit()
        logger.info(f"Target {target.email} clicked the link!")
        
    return {"message": "Tracking recorded"}

@router.get("/list", response_model=List[SimulationResponse])
async def list_simulations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all simulations for the user."""
    sims = db.query(Simulation).filter(Simulation.user_id == current_user.id).all()
    
    response = []
    for sim in sims:
        # Calculate stats
        total = len(sim.targets)
        sent = sum(1 for t in sim.targets if t.status != "pending")
        clicked = sum(1 for t in sim.targets if t.status == "clicked")
        
        response.append({
            "id": sim.id,
            "name": sim.name,
            "status": sim.status,
            "created_at": sim.created_at,
            "stats": {
                "total": total,
                "sent": sent,
                "clicked": clicked
            }
        })
        
    return response
