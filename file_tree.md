Unified Phishing Detection System - Project Structure
Recommended File and Folder Structure
phish-guard/
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── Makefile
│
├── docs/
│   ├── PRD_Unified_Phishing_Detection_System.md
│   ├── system_design.md
│   ├── api_documentation.md
│   ├── deployment_guide.md
│   └── user_manual.md
│
├── backend/
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── database.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── prediction.py
│   │   │   ├── auth.py
│   │   │   ├── samples.py
│   │   │   ├── reports.py
│   │   │   └── simulation.py
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── rate_limit.py
│   │   │   └── cors.py
│   │   └── dependencies.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── user.py
│   │   │   ├── sample.py
│   │   │   ├── prediction.py
│   │   │   ├── threat_intel.py
│   │   │   └── simulation.py
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── prediction.py
│   │       ├── sample.py
│   │       ├── user.py
│   │       └── report.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── prediction_service.py
│   │   ├── report_service.py
│   │   └── cache_service.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── collectors/
│   │   │   ├── __init__.py
│   │   │   ├── phishtank.py
│   │   │   ├── virustotal.py
│   │   │   ├── urlhaus.py
│   │   │   └── base_collector.py
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── email_processor.py
│   │   │   ├── url_processor.py
│   │   │   ├── sms_processor.py
│   │   │   ├── image_processor.py
│   │   │   └── audio_processor.py
│   │   └── threat_intel.py
│   ├── ml_pipeline/
│   │   ├── __init__.py
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── text_preprocessor.py
│   │   │   ├── url_preprocessor.py
│   │   │   ├── image_preprocessor.py
│   │   │   └── feature_extractor.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── text_model.py
│   │   │   ├── url_model.py
│   │   │   ├── ensemble_model.py
│   │   │   └── heuristics.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py
│   │   │   ├── evaluator.py
│   │   │   └── model_registry.py
│   │   ├── prediction/
│   │   │   ├── __init__.py
│   │   │   ├── predictor.py
│   │   │   └── batch_predictor.py
│   │   └── explainability/
│   │       ├── __init__.py
│   │       ├── shap_explainer.py
│   │       ├── attention_explainer.py
│   │       └── explanation_generator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── validators.py
│   │   ├── crypto.py
│   │   └── file_handler.py
│   └── Dockerfile
│
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── vite.config.ts
│   ├── index.html
│   ├── public/
│   │   ├── favicon.ico
│   │   ├── logo.png
│   │   └── manifest.json
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── ui/
│   │   │   │   ├── Button.tsx
│   │   │   │   ├── Input.tsx
│   │   │   │   ├── Card.tsx
│   │   │   │   ├── Modal.tsx
│   │   │   │   └── LoadingSpinner.tsx
│   │   │   ├── layout/
│   │   │   │   ├── Header.tsx
│   │   │   │   ├── Sidebar.tsx
│   │   │   │   └── Layout.tsx
│   │   │   ├── prediction/
│   │   │   │   ├── PredictionForm.tsx
│   │   │   │   ├── ResultDisplay.tsx
│   │   │   │   ├── ExplanationView.tsx
│   │   │   │   └── ConfidenceMeter.tsx
│   │   │   ├── dashboard/
│   │   │   │   ├── StatsCard.tsx
│   │   │   │   ├── RecentAlerts.tsx
│   │   │   │   └── ChartsSection.tsx
│   │   │   └── simulation/
│   │   │       ├── CampaignList.tsx
│   │   │       ├── CampaignForm.tsx
│   │   │       └── ResultsTable.tsx
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── URLScanner.tsx
│   │   │   ├── EmailInspector.tsx
│   │   │   ├── SMSAnalyzer.tsx
│   │   │   ├── ImageOCR.tsx
│   │   │   ├── AudioSTT.tsx
│   │   │   ├── Reports.tsx
│   │   │   ├── Simulation.tsx
│   │   │   └── Settings.tsx
│   │   ├── hooks/
│   │   │   ├── useAuth.ts
│   │   │   ├── usePrediction.ts
│   │   │   └── useWebSocket.ts
│   │   ├── services/
│   │   │   ├── api.ts
│   │   │   ├── auth.ts
│   │   │   └── websocket.ts
│   │   ├── store/
│   │   │   ├── index.ts
│   │   │   ├── authSlice.ts
│   │   │   └── predictionSlice.ts
│   │   ├── types/
│   │   │   ├── api.ts
│   │   │   ├── auth.ts
│   │   │   └── prediction.ts
│   │   ├── utils/
│   │   │   ├── constants.ts
│   │   │   ├── formatters.ts
│   │   │   └── validators.ts
│   │   └── styles/
│   │       ├── globals.css
│   │       └── components.css
│   └── Dockerfile
│
├── extension/
│   ├── manifest.json
│   ├── popup/
│   │   ├── popup.html
│   │   ├── popup.js
│   │   └── popup.css
│   ├── content/
│   │   ├── content_script.js
│   │   └── content_styles.css
│   ├── background/
│   │   └── background.js
│   ├── models/
│   │   ├── url_model.onnx
│   │   └── model_loader.js
│   ├── utils/
│   │   ├── api_client.js
│   │   ├── dom_analyzer.js
│   │   └── heuristics.js
│   ├── assets/
│   │   ├── icons/
│   │   │   ├── icon16.png
│   │   │   ├── icon48.png
│   │   │   └── icon128.png
│   │   └── warning_modal.html
│   └── build/
│       └── extension.zip
│
├── data/
│   ├── raw/
│   │   ├── phishtank/
│   │   ├── virustotal/
│   │   ├── urlhaus/
│   │   └── uploads/
│   ├── processed/
│   │   ├── v1/
│   │   ├── v2/
│   │   └── current/
│   ├── labeled/
│   │   ├── emails/
│   │   ├── urls/
│   │   ├── sms/
│   │   ├── images/
│   │   └── audio/
│   └── models/
│       ├── text_transformer/
│       ├── url_xgboost/
│       ├── ensemble/
│       └── artifacts/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   ├── evaluation_analysis.ipynb
│   └── feature_analysis.ipynb
│
├── reports/
│   ├── templates/
│   │   ├── forensic_report.html
│   │   └── executive_summary.html
│   ├── generated/
│   └── static/
│       ├── css/
│       └── images/
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_api/
│   │   ├── test_ml_pipeline/
│   │   ├── test_ingestion/
│   │   └── test_services/
│   ├── integration/
│   │   ├── test_api_endpoints.py
│   │   ├── test_ml_workflow.py
│   │   └── test_database.py
│   └── e2e/
│       ├── test_user_workflows.py
│       └── test_extension.py
│
├── scripts/
│   ├── setup_dev.sh
│   ├── deploy.sh
│   ├── backup_data.sh
│   ├── train_models.py
│   └── migrate_db.py
│
├── infrastructure/
│   ├── docker/
│   │   ├── backend.Dockerfile
│   │   ├── frontend.Dockerfile
│   │   └── nginx.Dockerfile
│   ├── k8s/
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ingress.yaml
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
│
└── .github/
    └── workflows/
        ├── ci.yml
        ├── cd.yml
        ├── security_scan.yml
        └── model_training.yml


Key Directory Explanations

backend/: FastAPI application with modular structure for API routes, ML pipeline, data ingestion, and services
frontend/: React TypeScript application with Tailwind CSS and dark/galaxy theme components
extension/: Chrome extension with manifest v3, content scripts, and ONNX model for client-side inference
data/: Organized data storage with versioning for raw, processed, labeled data and trained models
notebooks/: Jupyter notebooks for data analysis, model development, and evaluation
reports/: PDF report generation templates and static assets
tests/: Comprehensive testing structure with unit, integration, and end-to-end tests
infrastructure/: Docker, Kubernetes, and Terraform configurations for deployment
.github/workflows/: CI/CD pipelines for automated testing, building, and deployment
This structure supports modular development, clear separation of concerns, scalability, and maintainability while following Python and JavaScript best practices.
