Unified Phishing Detection System
A comprehensive AI-powered phishing detection system with multi-modal threat analysis, real-time protection, and explainable AI capabilities.

🌟 Features
Multi-Modal Detection: URLs, emails, SMS, images, and audio analysis
Real-Time Protection: Browser extension with instant threat warnings
Explainable AI: SHAP-based explanations and token-level highlighting
Threat Intelligence: Integration with PhishTank, VirusTotal, URLhaus
Simulation Mode: Phishing awareness training campaigns
Forensic Reports: Detailed PDF reports for security incidents
🏗️ Architecture
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Database      │
│   React/TS      │────│   FastAPI       │────│   PostgreSQL    │
│   Dark Theme    │    │   ML Pipeline   │    │   + Redis       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │──────────────│ Browser Ext     │──────────────│
                        │ Chrome MV3      │
                        │ ONNX/TF.js      │
                        └─────────────────┘
🚀 Quick Start
Prerequisites
Python 3.9+
Node.js 18+
PostgreSQL 13+
Redis 6+
Docker & Docker Compose
Installation
Clone the repository
git clone <repository-url>
cd phish-guard
Backend Setup
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
Frontend Setup
cd frontend
npm install
npm run dev
Database Setup
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Run migrations
cd backend
alembic upgrade head
Browser Extension
cd extension
# Load unpacked extension in Chrome Developer Mode
# Point to the extension directory
🔧 Configuration
Environment Variables
# Backend
DATABASE_URL=postgresql://user:pass@localhost/phishguard
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key
PHISHTANK_API_KEY=your-api-key
VIRUSTOTAL_API_KEY=your-api-key

# Frontend
REACT_APP_API_URL=http://localhost:8000
📊 API Documentation
Once running, visit:

API Docs: http://localhost:8000/docs
Frontend: http://localhost:3000
Key Endpoints
# URL Analysis
POST /api/v1/predict/url
curl -X POST "http://localhost:8000/api/v1/predict/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Email Analysis  
POST /api/v1/predict/email
curl -X POST "http://localhost:8000/api/v1/predict/email" \
  -H "Content-Type: application/json" \
  -d '{"subject": "Urgent Action Required", "body": "Click here..."}'

# Generate Report
POST /api/v1/reports/generate
🧪 Testing
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests
cd frontend
npm test

# Integration tests
pytest tests/integration/ -v
🐳 Docker Deployment
# Build and run all services
docker-compose up --build

# Production deployment
docker-compose -f docker-compose.prod.yml up -d
🔒 Security Features
JWT-based authentication
Role-based access control (RBAC)
End-to-end encryption
Privacy-preserving local processing
GDPR compliance
Audit logging
📈 Performance
Response Time: <200ms for API calls
Accuracy: >95% precision/recall
Throughput: 10,000+ requests/minute
Uptime: 99.9% availability target
🤖 ML Pipeline
Models
Text Analysis: DistilBERT fine-tuned transformer
URL Analysis: CNN + XGBoost ensemble
Image Analysis: OCR + logo detection
Audio Analysis: Whisper STT + NLP pipeline
Training
cd backend/ml_pipeline
python train.py --model text --epochs 5
python train.py --model url --estimators 200
python export_models.py --format onnx
🎯 Browser Extension
Installation
Open Chrome Extensions (chrome://extensions/)
Enable Developer Mode
Click “Load unpacked”
Select the extension/ directory
Features
Real-time URL scanning
Privacy-preserving analysis
Customizable warning thresholds
Offline capability with ONNX models
📋 Development
Project Structure
phish-guard/
├── backend/           # FastAPI application
├── frontend/          # React TypeScript app
├── extension/         # Chrome extension
├── data/             # Training datasets
├── notebooks/        # Jupyter analysis
├── tests/           # Test suites
└── docker-compose.yml
Contributing
Fork the repository
Create feature branch (git checkout -b feature/amazing-feature)
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Open Pull Request
📊 Monitoring & Analytics
Real-time detection metrics
Model performance tracking
User behavior analytics
Threat intelligence feeds
Compliance reporting
🔄 CI/CD Pipeline
GitHub Actions workflow:

Automated testing
Security scanning
Docker image building
Deployment to staging/production
📚 Documentation
API Reference
Architecture Guide
Deployment Guide
User Manual
🆘 Support
Documentation: docs/
Issues: GitHub Issues
Email: support@phishguard.com
📄 License
This project is licensed under the MIT License - see LICENSE file.

🙏 Acknowledgments
Hugging Face Transformers
SHAP for explainable AI
PhishTank, VirusTotal for threat intelligence
Open source security community
Editor
