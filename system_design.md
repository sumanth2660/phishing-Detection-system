Unified Phishing Detection System - System Architecture Design
Implementation Approach
We will implement a modular, scalable phishing detection system with the following key components:

Data Ingestion & Labeling Module - Collect and process multi-modal data (URLs, emails, SMS, images, audio)
ML/NLP Pipeline - Train and deploy ensemble models with explainability features
Backend API Service - FastAPI-based REST API for predictions and management
Frontend Dashboard - React TypeScript interface with dark/galaxy theme
Browser Extension - Real-time Chrome extension with client-side inference
Reporting & Analytics - PDF forensic reports and performance monitoring
Simulation & Training - Phishing awareness testing platform
The system prioritizes accuracy (>95%), interpretability (SHAP + attention), reproducibility (MLOps), and clean UX with real-time performance.

Main User-UI Interaction Patterns
Security Analyst Workflow

Upload samples → View prediction results → Analyze explainability → Generate forensic reports → Review false positives
End User Protection

Browse web → Extension detects threats → Warning modal → User decision → Feedback collection
Administrator Management

Configure threat feeds → Schedule retraining → Monitor system performance → Manage simulation campaigns
Simulation Participant

Receive test phishing → Take action → View educational feedback → Track improvement
System Architecture
@startuml
!define RECTANGLE class

package "Frontend Layer" {
    [React Dashboard] as dashboard
    [Chrome Extension] as extension
}

package "API Gateway" {
    [FastAPI Backend] as api
    [Authentication] as auth
    [Rate Limiting] as rate
}

package "ML Pipeline" {
    [Data Ingestion] as ingestion
    [Preprocessing] as preprocess
    [Model Training] as training
    [Prediction Service] as prediction
    [Explainability] as explain
}

package "Data Layer" {
    [PostgreSQL] as db
    [Redis Cache] as cache
    [File Storage] as storage
    [Model Registry] as models
}

package "External Services" {
    [PhishTank API] as phishtank
    [VirusTotal API] as virustotal
    [URLhaus API] as urlhaus
    [Threat Intel] as threatintel
}

package "Infrastructure" {
    [Docker Containers] as docker
    [CI/CD Pipeline] as cicd
    [Monitoring] as monitor
}

dashboard --> api : HTTPS/REST
extension --> api : HTTPS/REST
api --> auth : JWT Validation
api --> rate : Request Throttling
api --> prediction : ML Inference
api --> db : Data Persistence
api --> cache : Session/Results
prediction --> models : Load Models
prediction --> explain : Generate Explanations
ingestion --> threatintel : Fetch Feeds
ingestion --> db : Store Data
preprocess --> storage : Read/Write Files
training --> models : Save Artifacts
threatintel --> phishtank
threatintel --> virustotal
threatintel --> urlhaus
docker --> monitor : Metrics
cicd --> docker : Deploy
@enduml
UI Navigation Flow
@startuml
state "Dashboard" as Dashboard {
    [*] --> Dashboard
}
state "URL-Scanner" as URLScanner
state "Email-Inspector" as EmailInspector
state "SMS-Analyzer" as SMSAnalyzer
state "Image-OCR" as ImageOCR
state "Audio-STT" as AudioSTT
state "Reports" as Reports
state "Simulation" as Simulation
state "Settings" as Settings

Dashboard --> URLScanner : scan URL
Dashboard --> EmailInspector : analyze email
Dashboard --> SMSAnalyzer : check SMS
Dashboard --> ImageOCR : process image
Dashboard --> AudioSTT : transcribe audio
Dashboard --> Reports : view reports
Dashboard --> Simulation : run simulation
Dashboard --> Settings : configure

URLScanner --> Dashboard : back to home
URLScanner --> Reports : generate report
EmailInspector --> Dashboard : back to home
EmailInspector --> Reports : generate report
SMSAnalyzer --> Dashboard : back to home
ImageOCR --> Dashboard : back to home
AudioSTT --> Dashboard : back to home
Reports --> Dashboard : back to home
Simulation --> Dashboard : back to home
Settings --> Dashboard : back to home
@enduml
Class Diagram
@startuml
interface IDataCollector {
    +collect_data(source: str): List[Sample]
    +validate_data(data: Dict): bool
}

interface IPredictionService {
    +predict_url(url: str): PredictionResult
    +predict_email(email: EmailSample): PredictionResult
    +predict_sms(sms: SMSSample): PredictionResult
    +predict_image(image: ImageSample): PredictionResult
    +predict_audio(audio: AudioSample): PredictionResult
}

interface IExplainabilityService {
    +explain_prediction(sample: Sample, prediction: PredictionResult): ExplanationResult
    +generate_shap_values(features: Dict): Dict
    +highlight_tokens(text: str, attention_weights: List[float]): str
}

class DataIngestionService {
    +threat_intel_collector: ThreatIntelCollector
    +email_collector: EmailCollector
    +url_collector: URLCollector
    +collect_threat_feeds(): List[ThreatIntel]
    +process_uploaded_samples(files: List[File]): List[Sample]
}

class MLPipeline {
    +preprocessor: DataPreprocessor
    +text_model: TransformerModel
    +url_model: XGBoostModel
    +ensemble_model: EnsembleModel
    +train_models(dataset: Dataset): ModelArtifacts
    +predict(sample: Sample): PredictionResult
}

class PredictionAPI {
    +ml_pipeline: MLPipeline
    +explainer: ExplainabilityService
    +cache: RedisCache
    +predict_endpoint(request: PredictionRequest): PredictionResponse
    +explain_endpoint(sample_id: str): ExplanationResponse
}

class ReportGenerator {
    +template_engine: Jinja2
    +pdf_generator: WeasyPrint
    +generate_forensic_report(detection: Detection): bytes
    +create_executive_summary(stats: SystemStats): bytes
}

class Sample {
    +id: str
    +content: str
    +metadata: Dict
    +label: Optional[str]
    +timestamp: datetime
    +source: str
}

class PredictionResult {
    +probability: float
    +confidence: float
    +reasons: List[FeatureContribution]
    +model_version: str
    +processing_time: float
}

class ExplanationResult {
    +shap_values: Dict
    +attention_weights: List[float]
    +highlighted_text: str
    +feature_contributions: List[FeatureContribution]
    +natural_language_explanation: str
}

class FeatureContribution {
    +feature_name: str
    +contribution_score: float
    +description: str
    +category: str
}

IDataCollector ..> DataIngestionService
IPredictionService ..> MLPipeline
IExplainabilityService ..> ExplanationResult
DataIngestionService --> Sample
MLPipeline --> PredictionResult
PredictionAPI --> MLPipeline
PredictionAPI --> IExplainabilityService
ReportGenerator --> PredictionResult
Sample --> PredictionResult
PredictionResult --> ExplanationResult
ExplanationResult --> FeatureContribution
@enduml
Sequence Diagram
@startuml
actor User
participant "React Dashboard" as UI
participant "FastAPI Backend" as API
participant "ML Pipeline" as ML
participant "Explainability Service" as Explain
participant "PostgreSQL" as DB
participant "Redis Cache" as Cache
participant "Report Generator" as Report

User -> UI: Upload Email Sample
UI -> API: POST /predict/email
    note right
        Input: {
            "content": "email_body_text",
            "headers": {...},
            "attachments": [...],
            "metadata": {...}
        }
    end note

API -> Cache: Check cached result
Cache --> API: Cache miss

API -> ML: predict_email(sample)
ML -> ML: preprocess_email(sample)
ML -> ML: extract_features(sample)
ML -> ML: ensemble_predict(features)
ML --> API: PredictionResult
    note right
        Output: {
            "probability": 0.92,
            "confidence": 0.88,
            "reasons": [
                {"feature": "suspicious_sender", "score": 0.7},
                {"feature": "urgency_keywords", "score": 0.6}
            ],
            "model_version": "v1.2.3"
        }
    end note

API -> Explain: explain_prediction(sample, result)
Explain -> Explain: generate_shap_values(features)
Explain -> Explain: highlight_suspicious_tokens(text)
Explain --> API: ExplanationResult
    note right
        Output: {
            "highlighted_text": "<span class='suspicious'>urgent</span> action required",
            "shap_values": {...},
            "natural_explanation": "This email is likely phishing because..."
        }
    end note

API -> DB: INSERT detection_log
API -> Cache: Store result (TTL: 1h)

API --> UI: PredictionResponse
UI -> User: Display results with explanations

User -> UI: Generate Forensic Report
UI -> API: POST /report/generate
API -> Report: create_forensic_report(detection_id)
Report -> DB: Fetch detection details
Report -> Report: render_pdf_template(data)
Report --> API: PDF bytes
API --> UI: PDF download link
UI -> User: Download PDF report
@enduml
Database ER Diagram
@startuml
entity "users" as users {
    * id : uuid <<PK>>
    --
    * email : varchar(255)
    * password_hash : varchar(255)
    * role : enum('admin', 'analyst', 'user')
    * created_at : timestamp
    * last_login : timestamp
    is_active : boolean
}

entity "samples" as samples {
    * id : uuid <<PK>>
    --
    * content_type : enum('url', 'email', 'sms', 'image', 'audio')
    * raw_content : text
    * metadata : jsonb
    * file_path : varchar(500)
    * label : enum('phishing', 'legitimate', 'unknown')
    * source : varchar(100)
    * created_at : timestamp
    * updated_at : timestamp
    user_id : uuid <<FK>>
}

entity "predictions" as predictions {
    * id : uuid <<PK>>
    --
    * sample_id : uuid <<FK>>
    * probability : decimal(5,4)
    * confidence : decimal(5,4)
    * model_version : varchar(50)
    * processing_time_ms : integer
    * features : jsonb
    * created_at : timestamp
    user_id : uuid <<FK>>
}

entity "explanations" as explanations {
    * id : uuid <<PK>>
    --
    * prediction_id : uuid <<FK>>
    * shap_values : jsonb
    * attention_weights : jsonb
    * highlighted_content : text
    * feature_contributions : jsonb
    * natural_explanation : text
    * created_at : timestamp
}

entity "threat_intelligence" as threat_intel {
    * id : uuid <<PK>>
    --
    * url : varchar(2048)
    * domain : varchar(255)
    * ip_address : inet
    * verdict : enum('malicious', 'suspicious', 'clean')
    * source : varchar(100)
    * first_seen : timestamp
    * last_seen : timestamp
    * tags : text[]
    * confidence_score : decimal(3,2)
    * created_at : timestamp
}

entity "user_feedback" as feedback {
    * id : uuid <<PK>>
    --
    * prediction_id : uuid <<FK>>
    * user_id : uuid <<FK>>
    * feedback_type : enum('false_positive', 'false_negative', 'correct')
    * comments : text
    * created_at : timestamp
}

entity "simulation_campaigns" as campaigns {
    * id : uuid <<PK>>
    --
    * name : varchar(255)
    * description : text
    * template_type : enum('email', 'sms', 'url')
    * template_content : jsonb
    * target_users : uuid[]
    * start_date : timestamp
    * end_date : timestamp
    * status : enum('draft', 'active', 'completed')
    * created_by : uuid <<FK>>
    * created_at : timestamp
}

entity "simulation_results" as sim_results {
    * id : uuid <<PK>>
    --
    * campaign_id : uuid <<FK>>
    * user_id : uuid <<FK>>
    * action_taken : enum('clicked', 'reported', 'ignored', 'submitted_data')
    * timestamp : timestamp
    * ip_address : inet
    * user_agent : text
}

entity "model_artifacts" as models {
    * id : uuid <<PK>>
    --
    * model_name : varchar(100)
    * model_type : enum('text_transformer', 'url_xgboost', 'ensemble')
    * version : varchar(20)
    * file_path : varchar(500)
    * metrics : jsonb
    * training_data_hash : varchar(64)
    * created_at : timestamp
    * is_active : boolean
}

entity "audit_logs" as audit {
    * id : uuid <<PK>>
    --
    * user_id : uuid <<FK>>
    * action : varchar(100)
    * resource_type : varchar(50)
    * resource_id : uuid
    * details : jsonb
    * ip_address : inet
    * user_agent : text
    * timestamp : timestamp
}

users ||--o{ samples : "samples.user_id -> users.id"
users ||--o{ predictions : "predictions.user_id -> users.id"
users ||--o{ feedback : "feedback.user_id -> users.id"
users ||--o{ campaigns : "campaigns.created_by -> users.id"
users ||--o{ sim_results : "sim_results.user_id -> users.id"
users ||--o{ audit : "audit.user_id -> users.id"

samples ||--o{ predictions : "predictions.sample_id -> samples.id"
predictions ||--|| explanations : "explanations.prediction_id -> predictions.id"
predictions ||--o{ feedback : "feedback.prediction_id -> predictions.id"

campaigns ||--o{ sim_results : "sim_results.campaign_id -> campaigns.id"
@enduml
Anything UNCLEAR
The following aspects require clarification or assumptions:

Real-time Performance Requirements: Specific latency targets for browser extension vs. dashboard predictions
Data Retention Policies: How long to store samples, predictions, and user data for compliance
Third-party API Rate Limits: Specific quotas and costs for PhishTank, VirusTotal, URLhaus APIs
Model Retraining Frequency: Automated vs. manual triggers for model updates
Multi-tenant Architecture: Whether to support multiple organizations with data isolation
Compliance Requirements: Specific GDPR, HIPAA, or industry regulations to implement
Scalability Targets: Expected concurrent users and prediction volume for infrastructure sizing
Offline Capabilities: Level of functionality required when external APIs are unavailable
Editor
