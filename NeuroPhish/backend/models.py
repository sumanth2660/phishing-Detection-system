"""
Database models for Unified Phishing Detection System
SQLAlchemy models for all system entities and relationships.
"""

from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Float, JSON, ForeignKey, Enum, Uuid
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

Base = declarative_base()

# Enums
class RiskLevel(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ContentType(enum.Enum):
    URL = "url"
    EMAIL = "email"
    SMS = "sms"
    IMAGE = "image"
    AUDIO = "audio"

class ThreatVerdict(enum.Enum):
    CLEAN = "clean"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    UNKNOWN = "unknown"

class UserRole(enum.Enum):
    USER = "user"
    ANALYST = "analyst"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

# User Management
class User(Base):
    __tablename__ = "users"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    detections = relationship("DetectionResult", back_populates="user")
    feedback = relationship("UserFeedback", back_populates="user")
    simulation_results = relationship("SimulationResult", back_populates="user")

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid, ForeignKey("users.id"), nullable=False)
    token = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

# Detection Results
class DetectionResult(Base):
    __tablename__ = "detection_results"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid, ForeignKey("users.id"), nullable=False)
    content_type = Column(Enum(ContentType), nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)  # SHA-256 hash
    
    # Input data
    url = Column(Text)
    email_subject = Column(Text)
    email_body = Column(Text)
    sms_text = Column(Text)
    file_name = Column(String(255))
    file_size = Column(Integer)
    
    # Analysis results
    probability = Column(Float, nullable=False)
    risk_level = Column(Enum(RiskLevel), nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Model outputs
    heuristic_score = Column(Float)
    ml_score = Column(Float)
    ensemble_score = Column(Float)
    
    # Explainability data
    feature_contributions = Column(JSON)  # SHAP values and feature importance
    attention_weights = Column(JSON)      # Transformer attention weights
    explanation_html = Column(Text)       # HTML with highlighted tokens
    
    # Metadata
    processing_time_ms = Column(Float)
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="detections")
    feedback = relationship("UserFeedback", back_populates="detection")

# URL-specific data
class URLAnalysis(Base):
    __tablename__ = "url_analyses"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    detection_id = Column(Uuid, ForeignKey("detection_results.id"), nullable=False)
    
    url = Column(Text, nullable=False)
    domain = Column(String(255), nullable=False, index=True)
    tld = Column(String(50))
    path = Column(Text)
    query = Column(Text)
    
    # Technical details
    http_status = Column(Integer)
    ssl_issuer = Column(String(255))
    cert_age_days = Column(Integer)
    redirect_chain_count = Column(Integer)
    
    # Content analysis
    html_raw = Column(Text)
    extracted_text = Column(Text)
    title = Column(Text)
    meta_description = Column(Text)
    
    # Features
    url_length = Column(Integer)
    subdomain_count = Column(Integer)
    entropy = Column(Float)
    contains_ip = Column(Boolean)
    suspicious_tld = Column(Boolean)
    typosquatting_score = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

# Email-specific data
class EmailAnalysis(Base):
    __tablename__ = "email_analyses"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    detection_id = Column(Uuid, ForeignKey("detection_results.id"), nullable=False)
    
    # Email headers
    headers = Column(JSON)
    from_address = Column(String(255), index=True)
    to_address = Column(String(255))
    reply_to = Column(String(255))
    subject = Column(Text)
    
    # Content
    body_text = Column(Text)
    body_html = Column(Text)
    attachments_meta = Column(JSON)
    
    # Authentication
    spf_pass = Column(Boolean)
    dkim_valid = Column(Boolean)
    dmarc_pass = Column(Boolean)
    
    # Features
    urgency_score = Column(Float)
    link_count = Column(Integer)
    external_links = Column(Integer)
    html_to_text_ratio = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

# SMS-specific data
class SMSAnalysis(Base):
    __tablename__ = "sms_analyses"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    detection_id = Column(Uuid, ForeignKey("detection_results.id"), nullable=False)
    
    phone_from = Column(String(50))
    text = Column(Text, nullable=False)
    
    # Metadata
    carrier = Column(String(100))
    country = Column(String(2))  # ISO country code
    
    # Features
    text_length = Column(Integer)
    url_count = Column(Integer)
    urgency_score = Column(Float)
    sender_reputation = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

# Threat Intelligence
class ThreatIntelligence(Base):
    __tablename__ = "threat_intelligence"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    
    # Threat data
    url = Column(Text, nullable=False, index=True)
    domain = Column(String(255), index=True)
    ip_address = Column(String(45))  # IPv6 support
    
    # Verdict and sources
    verdict = Column(Enum(ThreatVerdict), nullable=False)
    source = Column(String(100), nullable=False)  # PhishTank, VirusTotal, etc.
    confidence = Column(Float)
    
    # Temporal data
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    
    # Metadata
    tags = Column(JSON)  # List of threat tags
    raw_data = Column(JSON)  # Original API response
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class BrandLogo(Base):
    __tablename__ = "brand_logos"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    logo_image_path = Column(String(500))
    phash = Column(String(64), nullable=False, index=True) # Perceptual hash
    orb_features = Column(JSON) # Serialized ORB keypoints/descriptors
    
    created_at = Column(DateTime, default=datetime.utcnow)

class Simulation(Base):
    __tablename__ = "simulations"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(Uuid, ForeignKey("users.id"))
    name = Column(String)
    template_subject = Column(String)
    template_body = Column(String)
    status = Column(String, default="draft")  # draft, active, completed
    created_at = Column(DateTime, default=datetime.utcnow)
    
    targets = relationship("SimulationTarget", back_populates="simulation")

class SimulationTarget(Base):
    __tablename__ = "simulation_targets"
    
    id = Column(String, primary_key=True, index=True)
    simulation_id = Column(String, ForeignKey("simulations.id"))
    email = Column(String)
    status = Column(String, default="pending")  # pending, sent, opened, clicked
    sent_at = Column(DateTime, nullable=True)
    opened_at = Column(DateTime, nullable=True)
    clicked_at = Column(DateTime, nullable=True)
    
    simulation = relationship("Simulation", back_populates="targets")

# User Feedback
class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid, ForeignKey("users.id"), nullable=False)
    detection_id = Column(Uuid, ForeignKey("detection_results.id"), nullable=False)
    
    is_correct = Column(Boolean, nullable=False)
    feedback_text = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="feedback")
    detection = relationship("DetectionResult", back_populates="feedback")

# Model Management
class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    
    name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # transformer, xgboost, cnn, etc.
    content_type = Column(Enum(ContentType), nullable=False)
    
    # Model metadata
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    checksum = Column(String(64))
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=False, nullable=False)
    is_production = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    deployed_at = Column(DateTime)

# Simulation and Training
class SimulationCampaign(Base):
    __tablename__ = "simulation_campaigns"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Campaign settings
    content_type = Column(Enum(ContentType), nullable=False)
    template_data = Column(JSON)  # Phishing template configuration
    target_users = Column(JSON)   # List of target user IDs
    
    # Schedule
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    results = relationship("SimulationResult", back_populates="campaign")

class SimulationResult(Base):
    __tablename__ = "simulation_results"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    campaign_id = Column(Uuid, ForeignKey("simulation_campaigns.id"), nullable=False)
    user_id = Column(Uuid, ForeignKey("users.id"), nullable=False)
    
    # Interaction data
    email_opened = Column(Boolean, default=False)
    link_clicked = Column(Boolean, default=False)
    credentials_entered = Column(Boolean, default=False)
    reported_phishing = Column(Boolean, default=False)
    
    # Timing
    sent_at = Column(DateTime, nullable=False)
    opened_at = Column(DateTime)
    clicked_at = Column(DateTime)
    reported_at = Column(DateTime)
    
    # Relationships
    campaign = relationship("SimulationCampaign", back_populates="results")
    user = relationship("User", back_populates="simulation_results")

# System Configuration
class SystemConfig(Base):
    __tablename__ = "system_config"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    description = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Audit Logging
class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    
    user_id = Column(Uuid, ForeignKey("users.id"))
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50))
    resource_id = Column(String(100))
    
    # Request details
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    # Data
    old_values = Column(JSON)
    new_values = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)