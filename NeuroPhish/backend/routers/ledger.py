from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging
from pydantic import BaseModel

from blockchain import Blockchain

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize single blockchain instance
blockchain = Blockchain()

class ThreatInput(BaseModel):
    url: str
    risk_level: str
    source: str
    details: Dict[str, Any]

@router.get("/chain")
async def get_chain():
    """Get the full blockchain."""
    return {
        "chain": blockchain.chain,
        "length": len(blockchain.chain),
        "pending_threats": blockchain.pending_threats
    }

@router.post("/mine")
async def mine_block():
    """Mine a new block with pending threats."""
    if not blockchain.pending_threats:
        return {"message": "No threats to mine"}

    # Run Proof of Work
    last_block = blockchain.get_last_block()
    last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_proof)
    
    # Forge the new Block by adding it to the chain
    previous_hash = blockchain.hash(last_block)
    block = blockchain.create_block(proof, previous_hash)
    
    return {
        "message": "New Block Forged",
        "index": block['index'],
        "transactions": block['threats'],
        "proof": block['proof'],
        "previous_hash": block['previous_hash'],
    }

@router.post("/add_threat")
async def add_threat(threat: ThreatInput):
    """Add a new verified threat to the transaction pool."""
    index = blockchain.add_threat(threat.dict())
    return {"message": f"Threat added to pending block {index}"}

@router.get("/validate")
async def validate_chain():
    """Validate the integrity of the blockchain."""
    is_valid = blockchain.is_chain_valid(blockchain.chain)
    if is_valid:
        return {"message": "Blockchain is valid ✅", "valid": True}
    else:
        return {"message": "Blockchain is invalid! ❌", "valid": False}
