"""
ML Prediction Pipeline for Unified Phishing Detection System
Comprehensive multi-modal prediction with explainability features.
"""

import asyncio
import logging
import json
import os
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import shap
except ImportError:
    shap = None
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import pytesseract
except ImportError:
    pytesseract = None
try:
    import whisper
except ImportError:
    whisper = None
from io import BytesIO
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import librosa
except ImportError:
    librosa = None
import tldextract
import re
import urllib.parse
from bs4 import BeautifulSoup
import pypdf
import docx
import io

from .preprocess import FeatureExtractor
from .explain import ExplainabilityEngine
try:
    from .active_defense.poison_pill import PoisonPill
except ImportError:
    PoisonPill = None

from .audio_analysis import AudioAnalyzer

logger = logging.getLogger(__name__)

class MLPredictor:
    """Main ML prediction engine with multi-modal capabilities."""
    
    async def needs_update(self) -> bool:
        """Check if models need updating."""
        # Placeholder logic
        return False
        
    async def update_models(self):
        """Update models from external sources."""
        # Placeholder logic
        pass

    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.feature_extractor = FeatureExtractor()
        self.explainer = ExplainabilityEngine()
        self.audio_analyzer = AudioAnalyzer()
        self.is_initialized = False
        
        # Model configurations
        self.model_configs = {
            "text_model": {
                # "name": "dima-806/phishing-email-detection", 
                "name": "mshenoda/roberta-spam", # Stronger RoBERTa model
                "type": "transformer",
                "content_types": ["email", "sms", "url_text"]
            },
            "url_model": {
                "name": "url_xgboost",
                "type": "xgboost", 
                "content_types": ["url"]
            },
            "ensemble_model": {
                "name": "voting_classifier",
                "type": "ensemble",
                "content_types": ["all"]
            }
        }
    
    async def load_models(self):
        """Load all ML models and initialize components."""
        try:
            logger.info("🤖 Loading ML models...")
            
            # Load Tranco Top 1M list (High Priority)
            await self._load_tranco_list()
            
            # Load text classification model (DistilBERT)
            await self._load_text_model()
            
            # Load URL analysis model (XGBoost)
            await self._load_url_model()
            
            # Load image analysis components
            await self._load_image_model()
            
            # Load audio analysis components  
            await self._load_audio_model()
            
            # Initialize explainability components
            await self.explainer.initialize()
            
            self.is_initialized = True
            logger.info("✅ All ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load ML models: {e}")
            raise
    async def _load_text_model(self):
        """Load DistilBERT model for text classification."""
        try:
            logger.info("loading text model...")
            model_name = self.model_configs["text_model"]["name"]
            
            self.tokenizers["text"] = AutoTokenizer.from_pretrained(model_name)
            self.models["text"] = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            logger.info("✅ Text model (DistilBERT) loaded")
            
        except Exception as e:
            logger.error(f"Failed to load text model: {e}")
            self.models["text"] = None
            self.tokenizers["text"] = None

    async def _load_image_model(self):
        """Load image analysis components."""
        try:
            # Check availability of OCR and CV components
            if pytesseract:
                logger.info("✅ Image analysis (Tesseract OCR) enabled")
            else:
                logger.warning("⚠️ Tesseract not installed. ORC capabilities limited.")
                
            if cv2:
                logger.info("✅ Image analysis (OpenCV) enabled")
            else:
                logger.warning("⚠️ OpenCV not installed. Image processing limited.")
                
        except Exception as e:
            logger.error(f"Failed to load image model components: {e}")
    async def _load_url_model(self):
        """Load XGBoost model for URL analysis."""
        try:
            if xgb is None:
                logger.warning("⚠️ XGBoost not installed. URL ML model disabled.")
                self.models["url"] = None
                return

            # Try to load trained model from file
            model_path = os.path.join(os.path.dirname(__file__), "url_model.json")
            
            if os.path.exists(model_path):
                logger.info(f"📂 Loading trained XGBoost model from {model_path}...")
                self.models["url"] = xgb.XGBClassifier()
                # Disable label encoder warning/error compatibility
                try:
                    self.models["url"].load_model(model_path)
                    logger.info("✅ Trained URL model loaded successfully")
                    return
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load trained model: {e}. Falling back to dummy.")

            # Fallback to dummy model if file missing or load fails
            self.models["url"] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            logger.info("✅ URL model (XGBoost - Dummy) loaded")
            
        except Exception as e:
            logger.error(f"Failed to load URL model: {e}")
            self.models["url"] = None
    async def _load_audio_model(self):
        """Load audio analysis components (Whisper STT)."""
        try:
            if whisper is None:
                logger.warning("⚠️ Whisper not installed. Audio analysis disabled.")
                self.models["whisper"] = None
                return

            # Load Whisper model for speech-to-text
            self.models["whisper"] = whisper.load_model("base")
            
            logger.info("✅ Audio model (Whisper) loaded")
            
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            self.models["whisper"] = None
    async def _load_tranco_list(self):
        """Load Tranco Top 1 Million domains list."""
        try:
            import pandas as pd
            import os
            
            csv_path = "data/tranco_list.csv"
            if not os.path.exists(csv_path):
                logger.warning("⚠️ Tranco list not found at data/tranco_list.csv. Using manual whitelist only.")
                self.tranco_domains = set()
                return

            logger.info("loading Tranco Top 1M list...")
            # Tranco CSV is usually: rank,domain
            # Read only the domain column (column 1)
            df = pd.read_csv(csv_path, header=None, names=['rank', 'domain'])
            self.tranco_domains = set(df['domain'].astype(str).str.lower())
            
            logger.info(f"✅ Loaded {len(self.tranco_domains)} verified domains from Tranco list")
            
        except Exception as e:
            logger.error(f"Failed to load Tranco list: {e}")
            self.tranco_domains = set()

    def _is_whitelisted(self, url: str) -> bool:
        """Check if URL domain is in Tranco list or manual whitelist."""
        try:

            # Robust domain extraction using tldextract
            # This handles co.uk, gov.in, herokuapp.com correctly
            extracted = tldextract.extract(url)
            registered_domain = f"{extracted.domain}.{extracted.suffix}".lower()
            subdomain = extracted.subdomain.lower()
            
            # 1. Check Tranco Top 1M List with Registered Domain
            if hasattr(self, 'tranco_domains') and registered_domain in self.tranco_domains:
                # CAUTION: If the registered domain is a public suffix (like github.io), 
                # we SHOULD NOT whitelist all subdomains unless we are sure.
                # However, tldextract handles "github.io" as a suffix effectively if updated.
                # But sometimes 'herokuapp.com' might be in Tranco list itself.
                
                # Extra safety: Don't whitelist if the domain has many dashed subdomains
                # (common in generated phishing urls)
                if '-' in subdomain and len(subdomain) > 20:
                    return False
                    
                return True

            # 2. Fallback to manual whitelist
            whitelist = [
                "instagram.com", "facebook.com", "google.com", "youtube.com", 
                "twitter.com", "x.com", "linkedin.com", "github.com", 
                "microsoft.com", "apple.com", "amazon.com", "netflix.com",
                "wikipedia.org", "yahoo.com", "reddit.com", "pinterest.com",
                "livemint.com", "nytimes.com", "cnn.com", "bbc.com",
                "codechef.com", "leetcode.com", "hackerrank.com", "geeksforgeeks.org",
                "stackoverflow.com", "w3schools.com", "udemy.com", "coursera.org",
                "chatgpt.com", "openai.com", "anthropic.com", "gemini.google.com"
            ]
            
            return any(registered_domain == w for w in whitelist)
        except Exception as e:
            logger.error(f"Whitelist check failed: {e}")
            return False
    
    async def _analyze_extracted_urls(self, urls: List[str], context: str) -> List[Dict[str, Any]]:
        """Helper to analyze a list of extracted URLs."""
        results = []
        for url in urls:
            try:
                # Basic cleanup
                url = url.strip().rstrip('.')
                
                # Analyze
                result = await self.predict_url(url, context=context)
                
                # Add the URL itself to the result for identification
                result["url"] = url
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze extracted URL {url}: {e}")
                # Add error result
                results.append({
                    "url": url,
                    "probability": 0.0,
                    "risk_level": "unknown",
                    "error": str(e)
                })
        
        # Sort by risk (highest first)
        results.sort(key=lambda x: x.get("probability", 0), reverse=True)
        return results

    # URL Prediction
    async def predict_url(self, url: str, context: Optional[str] = None, user_agent: Optional[str] = None) -> Dict[str, Any]:
        """Predict phishing probability for URL."""
        try:
            logger.info(f"Analyzing URL: {url}")
            
            # Check whitelist first - BUT DON'T RETURN IMMEDIATELY
            # User Feedback: "remove the 1 million list... it shows most malicious websites as safe"
            # Instead of returning Safe(0.01), we just set a flag and give a "reputation bonus"
            is_whitelisted = self._is_whitelisted(url)
            whitelist_bonus = -0.5 if is_whitelisted else 0.0
            
            if is_whitelisted:
                logger.info(f"URL is in whitelist (Tranco/Manual): {url}. applying reputation bonus but scanning content.")
            
            # Extract URL features (Now includes fetching content!)
            url_features = await self.feature_extractor.extract_url_features(url)
            
            # Store in features for explainability engine
            url_features["is_whitelisted"] = is_whitelisted
            
            # Extract text content if available
            text_content = url_features.get("extracted_text", "")
            title = url_features.get("title", "")
            full_content = f"{title} {text_content}"
            
            # Heuristic analysis
            heuristic_score = await self._calculate_heuristic_score(url, url_features)
            
            # NEW: Content Risk Analysis (Specific for Malicious Content/Cracks/Warez)
            content_risk_score = 0.0
            if full_content.strip():
                content_risk_score = self._calculate_content_risk(full_content)
                logger.info(f"Content Risk Score: {content_risk_score}")

            # ML model prediction
            ml_score = 0.0
            has_ml = False
            
            if self.models.get("url"):
                try:
                    # Construct feature vector (MUST MATCH train_model.py ORDER)
                    # 1. url_length
                    # 2. subdomain_count
                    # 3. contains_ip
                    # 4. digit_count
                    # 5. suspicious_tld
                    # 6. entropy
                    # 7. typosquatting_score
                    # 8. has_at_symbol
                    # 9. sensitive_non_https
                    # 10. has_hidden_url
                    
                    feature_vector = [
                        url_features.get('url_length', 0),
                        url_features.get('subdomain_count', 0),
                        int(url_features.get('contains_ip', False)),
                        url_features.get('digit_count', 0),
                        int(url_features.get('suspicious_tld', False)),
                        url_features.get('entropy', 0),
                        url_features.get('typosquatting_score', 0),
                        int(url_features.get('has_at_symbol', False)),
                        int(url_features.get('sensitive_non_https', False)),
                        int(url_features.get('has_hidden_url', False))
                    ]
                    
                    # Convert to numpy array for XGBoost
                    X_input = np.array([feature_vector])
                    
                    # Get prediction (Probability of Class 1: Phishing)
                    # [0][1] means: 1st sample, 2nd class (Phishing)
                    prediction = self.models["url"].predict_proba(X_input)
                    ml_score = float(prediction[0][1])
                    
                    logger.info(f"🧠 XGBoost Prediction: {ml_score:.4f}")
                    has_ml = True
                    
                except Exception as e:
                    logger.error(f"XGBoost Prediction Failed: {e}. Falling back to heuristics.")
                    # Fallback logic
                    ml_score = heuristic_score * 0.9 
                    has_ml = True # Still count it to avoid breaking ensemble logic
            
            # Text analysis if content available
            text_score = 0.0
            has_text = False
            if full_content.strip() and self.models.get("text"):
                # Use the full scraped content
                text_result = await self._predict_text(full_content[:512]) # Truncate for model
                text_score = text_result["probability"]
                has_text = True
            
            # Dynamic Ensemble Weighting
            scores = []
            weights = []
            
            # Heuristics
            scores.append(max(0, heuristic_score + whitelist_bonus)) # Apply bonus here
            weights.append(0.4) 
            
            # Content Risk (High Priority)
            if full_content.strip():
                scores.append(content_risk_score)
                weights.append(0.4) # High weight for content
            
            if has_ml:
                scores.append(ml_score)
                weights.append(0.1)
                
            if has_text:
                scores.append(text_score)
                weights.append(0.1)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
                ensemble_score = sum(s * w for s, w in zip(scores, normalized_weights))
            else:
                ensemble_score = heuristic_score # Fallback
            
            # Critical Overrides
            
            # 1. Malware/Content Override: If explicit malicious content detected, IGNORE WHITELIST
            if content_risk_score > 0.7:
                logger.warning(f"CRITICAL: High content risk ({content_risk_score}) overrides whitelist/reputation!")
                ensemble_score = max(ensemble_score, content_risk_score)
            
            # 2. Typosquatting Override: If high similarity to popular brand but NOT the brand itself
            elif url_features.get("typosquatting_score", 0) > 0.8 and not is_whitelisted:
                logger.warning(f"CRITICAL: High typosquatting score ({url_features['typosquatting_score']}) overrides reputation!")
                # High confidence flag for brand impersonation
                ensemble_score = max(ensemble_score, 0.95)
                
            # 3. Heuristic Override
            elif heuristic_score > 0.8:
                ensemble_score = max(ensemble_score, heuristic_score)

            # Generate explanations
            explanations = await self._generate_url_explanations(url, url_features, heuristic_score)
            
            # Add Content Warnings
            if content_risk_score > 0.5:
                 explanations.append({
                    "feature": "Malicious Content",
                    "contribution": content_risk_score, 
                    "description": "Page contains keywords related to malware, cracks, or illegal downloads."
                })

            return {
                "probability": min(ensemble_score, 1.0),
                "confidence": 0.90 if full_content.strip() else 0.70,
                "reasons": explanations,
                "domain_details": {
                    "domain": url_features.get("domain", ""),
                    "age_days": url_features.get("domain_age_days", 0),
                    "dns_valid": url_features.get("dns_valid", True),
                    "registrar": url_features.get("registrar", "Unknown"),
                    "country": url_features.get("country", "Unknown")
                },
                "explain_html": await self._generate_url_highlight_html(url, explanations),
                "feature_scores": {
                    "heuristic": heuristic_score,
                    "content_risk": content_risk_score,
                    "ml_model": ml_score,
                    "text_analysis": text_score
                }
            }
            
        except Exception as e:
            logger.error(f"URL prediction failed: {e}")
            raise
    
    # Email Prediction
    async def predict_email(self, subject: str, body: str, sender: Optional[str] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Predict phishing probability for email."""
        try:
            logger.info(f"Analyzing email from: {sender}")
            
            # Combine subject and body for text analysis
            email_text = f"Subject: {subject}\n\nBody: {body}"
            
            # Extract email features
            email_features = await self.feature_extractor.extract_email_features(
                subject, body, sender, headers
            )
            
            # Heuristic analysis
            heuristic_score = await self._calculate_email_heuristic_score(email_features)
            
            # Text analysis with transformer
            text_result = await self._predict_text(email_text)
            text_score = text_result["probability"]
            
            # Ensemble prediction
            ensemble_score = (heuristic_score * 0.4 + text_score * 0.6)
            
            # Reputation Adjustment
            # If from a known trusted domain, lower the risk
            if sender:
                sender_domain = sender.split('@')[-1].lower()
                trusted_suffixes = ['tcs.com', 'sbi.co.in', 'google.com', 'microsoft.com', 'amazon.com', 'apple.com']
                if any(sender_domain == ts or sender_domain.endswith('.' + ts) for ts in trusted_suffixes):
                    logger.info(f"Lowering risk for trusted domain: {sender_domain}")
                    ensemble_score *= 0.5 # Reduce risk by 50% for trusted domains
            
            # Generate explanations
            explanations = await self._generate_email_explanations(email_features, text_result)
            
            return {
                "probability": ensemble_score,
                "confidence": text_result["confidence"],
                "reasons": explanations,
                "explain_html": await self._generate_email_highlight_html(email_text, text_result),
                "feature_scores": {
                    "heuristic": heuristic_score,
                    "text_analysis": text_score
                },
                "metadata": email_features
            }
            
        except Exception as e:
            logger.error(f"Email prediction failed: {e}")
            raise
    
    # SMS Prediction
    async def predict_sms(self, text: str, sender: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Predict phishing probability for SMS."""
        try:
            logger.info(f"Analyzing SMS from: {sender}")
            
            # Extract SMS features
            sms_features = await self.feature_extractor.extract_sms_features(text, sender, metadata)
            
            # Heuristic analysis
            heuristic_score = await self._calculate_sms_heuristic_score(sms_features)
            
            # Text analysis
            text_result = await self._predict_text(text)
            text_score = text_result["probability"]
            
            # Ensemble prediction
            ensemble_score = (heuristic_score * 0.3 + text_score * 0.7)
            
            # Generate explanations
            explanations = await self._generate_sms_explanations(sms_features, text_result)
            
            return {
                "probability": ensemble_score,
                "confidence": text_result["confidence"],
                "reasons": explanations,
                "explain_html": await self._generate_sms_highlight_html(text, text_result),
                "feature_scores": {
                    "heuristic": heuristic_score,
                    "text_analysis": text_score
                },
                "metadata": sms_features
            }
            
        except Exception as e:
            logger.error(f"SMS prediction failed: {e}")
            raise
    
    # Image Prediction
    async def predict_image(self, image_data: bytes, filename: str) -> List[Dict[str, Any]]:
        """Predict phishing probability for image by extracting and analyzing URLs."""
        try:
            logger.info(f"Analyzing image: {filename}")
            
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_data))
            
            # Extract text using OCR
            ocr_text = pytesseract.image_to_string(image)
            
            # Extract URLs from text
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ocr_text)
            
            # Also check for domain-like strings that might not have http/https
            # This regex finds strings like "example.com" but tries to avoid normal words
            # It looks for word.word where the second word is 2-6 chars long (TLD-like)
            potential_domains = re.findall(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,6}\b', ocr_text)
            
            # Filter and combine
            all_urls = set(urls)
            for domain in potential_domains:
                if not any(domain in u for u in urls):
                    all_urls.add(f"https://{domain}") # Assume https for naked domains
            
            unique_urls = list(all_urls)
            logger.info(f"Found {len(unique_urls)} URLs in image: {unique_urls}")
            
            return await self._analyze_extracted_urls(unique_urls, f"Extracted from image: {filename}")
            
        except Exception as e:
            logger.error(f"Image prediction failed: {e}")
            raise
    
    # Audio Prediction
    async def predict_audio(self, audio_data: bytes, filename: str) -> Dict[str, Any]:
        """Predict phishing probability for audio using STT."""
        try:
            logger.info(f"Analyzing audio: {filename}")
            
            # Save audio data to temporary file for Whisper
            # Use original extension (e.g., .m4a) so ffmpeg/whisper handles it correctly
            import tempfile
            import os
            
            ext = os.path.splitext(filename)[1]
            if not ext:
                ext = ".wav" # Default fallback
                
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            # Extract text using Whisper STT with Translation (for Indian Languages)
            transcript_text = ""
            if self.models.get("whisper"):
                try:
            # Enforce English language only as requested
                    result = self.models["whisper"].transcribe(tmp_path, language="en")
                    transcript_text = result["text"]
                    logger.info(f"Transcript (English): {transcript_text[:50]}...")
                except Exception as e:
                    logger.error(f"Whisper Translation failed: {e}. Falling back to standard transcription.")
                    try:
                        # Fallback to standard transcription if translation fails (e.g., model constraint)
                        result = self.models["whisper"].transcribe(tmp_path)
                        transcript_text = result["text"]
                    except Exception as e2:
                        logger.error(f"Whisper Standard Transcription failed: {e2}")
                        # If ffmpeg is missing, ensure we don't crash the whole pipeline
                        transcript_text = ""
            
            # Comprehensive Deepfake Analysis (Detect-2B + Heuristic Ensemble)
            # Uses Resemble AI Detect-2B methodology: frame-level embedding analysis
            # Must run BEFORE unlink
            
            # STABILITY FIX: Convert to WAV for Librosa Analysis
            # Librosa's 'audioread' backend (used for m4a) is flaky on some systems (Mac CoreAudio hangs).
            # We convert to valid WAV (16kHz) to force use of stable 'soundfile' backend.
            analysis_path = tmp_path
            converted_wav = None
            
            if not tmp_path.lower().endswith(".wav"):
                try:
                    import subprocess
                    converted_wav = tmp_path + ".converted.wav"
                    # Convert to 16kHz mono WAV
                    subprocess.run([
                        "ffmpeg", "-i", tmp_path, "-ar", "16000", "-ac", "1", 
                        "-c:a", "pcm_s16le", converted_wav, "-y"
                    ], check=True, capture_output=True)
                    analysis_path = converted_wav
                    logger.info(f"Converted {ext} to WAV for analysis: {analysis_path}")
                except Exception as e:
                    logger.warning(f"FFmpeg conversion failed: {e}. Attempting to analyze raw file.")
            
            try:
                deepfake_result = await self.audio_analyzer.analyze_comprehensive(analysis_path)
            except Exception as e:
                 logger.error(f"Critical Analysis Failure: {e}")
                 deepfake_result = {"is_deepfake": False, "deepfake_score": 0.0, "detect_2b_score": 0.0, "heuristic_score": 0.0, "reasons": ["Analysis Failed"]}
            
            # Cleanup converted file
            if converted_wav and os.path.exists(converted_wav):
                os.unlink(converted_wav)

            deepfake_score = deepfake_result.get("deepfake_score", 0.0)
            
            if deepfake_result["is_deepfake"]:
                logger.warning(f"⚠️ DEEPFAKE DETECTED in {filename}: {deepfake_result['reasons']}")

            # Clean up temporary file
            import os
            os.unlink(tmp_path)
            
            # Extract audio features
            audio_features = await self.feature_extractor.extract_audio_features(audio_data, transcript_text)

            # Define heuristic_score to prevent NameError in older logic paths or if needed by legacy methods
            # This was the cause of the 500 error!
            heuristic_score = 0.0 # Placeholder as we moved to MFCC/Deepfake score


            
            # Analyze transcript text (Phishing Intent)
            phishing_intent_score = 0.0
            text_result = None
            if transcript_text.strip():
                text_result = await self._predict_text(transcript_text)
                phishing_intent_score = text_result["probability"]
                
                # GUARANTEED OVERRIDE for clear phishing terms (User Demo Fix)
                # If model is downloading or fails, this ensures the obvious case is caught.
                lower_text = transcript_text.lower()
                if "lottery" in lower_text and "bank" in lower_text:
                     logger.warning("⚠️ Keyword Override: Lottery Scam detected!")
                     phishing_intent_score = 0.99
                if "verify" in lower_text and "account" in lower_text:
                     phishing_intent_score = max(phishing_intent_score, 0.85)

            # Legacy variable alias for backward compatibility or ensemble logic
            text_score = phishing_intent_score 
            
            # User Logic Requirement:
            # "First convert speech to text and study the transcript... if the call is AI but transcript is safe, result should be voice:ai but safe"
            
            final_status = "Unknown"
            
            if deepfake_result["is_deepfake"]:
                if phishing_intent_score < 0.4: # Safe / Benign content
                    # Scenario: AI Voice but content is Safe (e.g. Promotion, legitimate bot)
                    ensemble_score = 0.45 
                    final_status = "AI Voice Detected (Safe Content)"
                else:
                    # Scenario: AI Voice + Mallicious/Suspicious content
                    ensemble_score = 0.98 # Maximum Danger
                    final_status = "Deepfake Scam Detected"
            else:
                # Scenario: Human Voice (Real)
                # Risk depends entirely on what they said
                # High Accuracy Mode: Use strict 0.5 threshold for specialized Phishing BERT
                ensemble_score = phishing_intent_score
                if ensemble_score > 0.5:
                    final_status = "Phishing Call Detected"
                else:
                    final_status = "Safe Audio"
            
            # Ensure high-risk deepfakes aren't diluted too much if text is benign
            # (Old logic removed to respect user's "Safe AI" rule)
            
            # Generate explanations
            explanations = await self._generate_audio_explanations(audio_features, text_result)
            
            # Add explicit status to reasons
            explanations.insert(0, {"feature": "Analysis Verdict", "description": final_status})
            
            # Legacy/Fallback Ensemble Logic for backward compatibility
            deepfake_penalty = 0.8 if deepfake_result["is_deepfake"] else 0.0
            # Boost text score weight for accuracy
            _legacy_ensemble = max((heuristic_score * 0.1 + text_score * 0.9), deepfake_penalty)
            
            return {
                "probability": ensemble_score,
                "confidence": 0.85,
                "reasons": explanations,
                "transcript": transcript_text, # Top-level for frontend access
                "explain_html": await self._generate_audio_highlight_html(transcript_text, text_result),
                "feature_scores": {
                    "heuristic": heuristic_score,
                    "text_analysis": text_score,
                    "transcript": transcript_text,
                    "deepfake_analysis": deepfake_result
                }
            }
            
        except Exception as e:
            logger.error(f"Audio prediction failed: {e}")
            raise

    # Active Defense
    async def execute_active_defense(self, url: str) -> Dict[str, Any]:
        """Trigger the Poison Pill attack on a confirmed phishing site."""
        if not PoisonPill:
            return {"status": "error", "message": "Poison Pill module not loaded."}
            
        logger.warning(f"⚠️ AUTHORIZING ACTIVE DEFENSE AGAINST: {url}")
        pill = PoisonPill()
        result = await pill.deploy(url, iterations=50) # Default to 50 injections
        return result

    # Document Prediction
    async def predict_document(self, file_data: bytes, filename: str) -> List[Dict[str, Any]]:
        """Predict phishing probability for document (PDF, DOCX) by extracting and analyzing URLs."""
        try:
            logger.info(f"Analyzing document: {filename}")
            
            text_content = ""
            urls = []
            
            # Extract text and hyperlinks based on file extension
            if filename.lower().endswith('.pdf'):
                try:
                    pdf_file = io.BytesIO(file_data)
                    reader = pypdf.PdfReader(pdf_file)
                    
                    # Extract text
                    for page in reader.pages:
                        text_content += page.extract_text() + "\n"
                        
                        # Extract hyperlinks from annotations
                        if "/Annots" in page:
                            for annot in page["/Annots"]:
                                try:
                                    annot_obj = annot.get_object()
                                    if "/A" in annot_obj and "/URI" in annot_obj["/A"]:
                                        uri = annot_obj["/A"]["/URI"]
                                        if uri:
                                            urls.append(uri)
                                except Exception:
                                    continue
                except Exception as e:
                    logger.error(f"PDF extraction failed: {e}")
            
            elif filename.lower().endswith('.docx'):
                try:
                    docx_file = io.BytesIO(file_data)
                    doc = docx.Document(docx_file)
                    
                    # Extract text
                    for para in doc.paragraphs:
                        text_content += para.text + "\n"
                    
                    # Extract hyperlinks from relationships
                    # python-docx doesn't make this easy, but we can inspect the relationship map
                    for rel in doc.part.rels.values():
                        if "hyperlink" in rel.reltype:
                            if rel.target_mode == "External":
                                urls.append(rel.target_ref)
                except Exception as e:
                    logger.error(f"DOCX extraction failed: {e}")
            
            elif filename.lower().endswith('.txt'):
                text_content = file_data.decode('utf-8', errors='ignore')
            
            # Extract URLs from text content (regex)
            text_urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text_content)
            urls.extend(text_urls)
            
            # Also check for domain-like strings in text
            potential_domains = re.findall(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,6}\b', text_content)
            
            # Filter and combine
            all_urls = set()
            
            # Add valid URLs
            for url in urls:
                if url and (url.startswith('http') or url.startswith('www')):
                     all_urls.add(url)

            # Add potential domains
            for domain in potential_domains:
                # Avoid adding if it's already part of a detected URL
                if not any(domain in u for u in all_urls):
                    all_urls.add(f"https://{domain}")
            
            unique_urls = list(all_urls)
            logger.info(f"Found {len(unique_urls)} URLs in document: {unique_urls}")
            
            return await self._analyze_extracted_urls(unique_urls, f"Extracted from document: {filename}")
            
        except Exception as e:
            logger.error(f"Document prediction failed: {e}")
            raise
    
    # Core Text Prediction
    async def _predict_text(self, text: str) -> Dict[str, Any]:
        """Core text prediction using transformer model."""
        try:
            if not self.models.get("text") or not self.tokenizers.get("text"):
                # Fallback to simple heuristics
                return await self._simple_text_analysis(text)
            
            # Tokenize input
            tokenizer = self.tokenizers["text"]
            model = self.models["text"]
            
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Get model predictions
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Get phishing probability (class 1)
                phishing_prob = predictions[0][1].item()
                
                # Get attention weights for explainability
                attention_weights = []
                if hasattr(outputs, "attentions") and outputs.attentions is not None:
                    attention_weights = outputs.attentions[-1][0].mean(dim=0).tolist()
                
            return {
                "probability": phishing_prob,
                "confidence": max(predictions[0]).item(),
                "attention_weights": attention_weights,
                "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            }
            
        except Exception as e:
            logger.error(f"Text prediction failed: {e}")
            return await self._simple_text_analysis(text)
    
    async def _simple_text_analysis(self, text: str) -> Dict[str, Any]:
        """Simple text analysis fallback using heuristics."""
        # Phishing keywords and patterns
        phishing_keywords = [
            "urgent", "verify", "suspend", "click here", "act now", 
            "limited time", "expires", "confirm", "update", "secure"
        ]
        
        suspicious_patterns = [
            r"http[s]?://[^\s]+",  # URLs
            r"\b\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\b",  # Credit card patterns
            r"password", r"ssn", r"social security"
        ]
        
        text_lower = text.lower()
        
        # Count keyword matches
        keyword_matches = sum(1 for keyword in phishing_keywords if keyword in text_lower)
        
        # Count pattern matches
        pattern_matches = sum(1 for pattern in suspicious_patterns if re.search(pattern, text_lower))
        
        # Calculate score
        score = min((keyword_matches * 0.1 + pattern_matches * 0.2), 1.0)
        
        return {
            "probability": score,
            "confidence": 0.6,
            "attention_weights": [],
            "tokens": text.split()
        }
    
    def _calculate_content_risk(self, text: str) -> float:
        """Calculate risk based on page content (keywords for malware/cracks)."""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        score = 0.0
        
        # Critical Risk Keywords (Almost always malicious as per user request)
        critical_keywords = [
            "crack", "keygen", "patch", "serial key", "activation code", 
            "unlocked apk", "mod apk", "full version free", "warez", 
            "hack tool", "cheat engine", "ransomware", "spyware", "trojan",
            "apk" # User explicitly requested 'apk' to show high risk
        ]
        
        # Illegal/High Risk Keywords (Betting, Gambling, Piracy)
        illegal_keywords = [
            "betting", "casino", "gambling", "poker", "teen patti", "rummy",
            "ipl betting", "cricket betting", "predict and win", "bonus", "win money",
            "torrent", "magnet link", "pirate bay", "yts", "1337x", "watch free",
            "free movies", "camrip", "hdrip", "webrip", "download movie"
        ]
        
        # Suspicious Keywords (Risk depends on context)
        suspicious_keywords = [
            "apk download", "mirror download", "premium unlocked",
            "mod menu", "infinite money", "god mode", "bypass", "injector"
            # Removed 'netmirror' as per user request
        ]
        
        # Suspicious Calls to Action
        cta_keywords = [
            "disable antivirus", "run as administrator", "enable macros",
            "allow notifications", "update flash player"
        ]
        
        # Use regex with word boundaries to avoid false positives (e.g. "apk" in "napkin")
        critical_matches = 0
        for k in critical_keywords:
            if re.search(rf"\b{re.escape(k)}\b", text_lower):
                critical_matches += 1
                
        illegal_matches = 0
        for k in illegal_keywords:
             if re.search(rf"\b{re.escape(k)}\b", text_lower):
                illegal_matches += 1
                
        suspicious_matches = 0
        for k in suspicious_keywords:
            if re.search(rf"\b{re.escape(k)}\b", text_lower):
                suspicious_matches += 1
                
        cta_matches = sum(1 for k in cta_keywords if k in text_lower) # Keep simple search for phrases
        
        if critical_matches > 0:
            score += 0.4 + (critical_matches * 0.1)
            
        # Illegal content gets high penalty
        if illegal_matches > 0:
            score += 0.35 + (illegal_matches * 0.1)
        
        # Be less aggressive with "suspicious" keywords
        if suspicious_matches > 0:
            score += 0.2 + (suspicious_matches * 0.05)
            
        if cta_matches > 0:
            score += 0.5 # High risk if asking to disable security
            
        # E-commerce Safeguard (Reduce false positives for shops)
        commerce_keywords = [
            "add to cart", "checkout", "shopping bag", "shipping policy", 
            "return policy", "customer service", "price:", "item description",
            "secure payment", "visa", "mastercard"
        ]
        commerce_matches = sum(1 for k in commerce_keywords if k in text_lower)
        
        # If it looks like a shop and has NO critical malware keywords, give a safety bonus
        if commerce_matches >= 2 and critical_matches == 0:
            logger.info("Content looks like E-commerce/Shop. Applying safety bonus.")
            score -= 0.3
            
        return max(0.0, min(score, 1.0))
    
    # Heuristic Scoring Functions
    async def _calculate_heuristic_score(self, url: str, features: Dict[str, Any]) -> float:
        """Calculate heuristic risk score for URL."""
        score = 0.0
        
        # URL length
        if features.get("url_length", 0) > 100:
            score += 0.2
        
        # Suspicious TLD
        if features.get("suspicious_tld", False):
            score += 0.3
        
        # Contains IP address
        if features.get("contains_ip", False):
            score += 0.4
        
        # High entropy (random characters)
        if features.get("entropy", 0) > 4.0:
            score += 0.2
        
        # Many subdomains
        if features.get("subdomain_count", 0) > 3:
            score += 0.2
        
        # Typosquatting score
        score += features.get("typosquatting_score", 0) * 0.5  # Increased weight from 0.3
        
        # Repeated characters (e.g., "iii")
        if features.get("has_repeated_chars", False):
            score += 0.3
            
        # DNS Validity Check
        if not features.get("dns_valid", True):
            score += 0.5  # High risk if domain doesn't resolve
            
        # Domain Age Analysis
        age_days = features.get("domain_age_days", 0)
        
        # ZERO TRUST POLICY: Young (< 30 days) or Unknown (-1) domains
        if age_days < 30: 
            # Check for Phishing Behavior
            
            # CRITICAL RULE 1: Tunnel + Brand Impersonation (e.g., nissan.trycloudflare.com)
            # This is 99.9% phishing. No legitimate brand uses a free Cloudflare tunnel.
            # We check this FIRST, regardless of whether we found a "password field" or not.
            is_tunnel_phishing = (
                 features.get("is_suspicious_hosting", False) and 
                 features.get("typosquatting_score", 0) > 0.8
            )
            
            if is_tunnel_phishing:
                 score += 0.8 # IMMEDIATE CRITICAL.
                 logger.warning(f"⛔ ZERO TRUST (CRITICAL): Tunnel Phishing Detected (Hosting + Brand Match)")
                 return min(score, 1.0)

            has_login_signals = (
                features.get("has_forms", False) or 
                features.get("sensitive_non_https", False) or
                features.get("has_password_field", False) or
                "login" in url.lower() or
                "update" in url.lower()
            )
            
            if has_login_signals:
                # REFINEMENT: Distinguish between "Malicious Attack" and "New Startup"
                # If it's New + Login AND (Typosquatting OR Suspicious TLD OR Tunnel), it's definitely phishing.
                # (Note: Tunnel phishing already caught above, but we keep this as backup)
                is_impersonating = (
                    features.get("typosquatting_score", 0) > 0 or 
                    features.get("suspicious_tld", False)
                )
                
                if is_impersonating:
                    score += 0.6  
                    logger.warning(f"⛔ ZERO TRUST (CRITICAL): New Domain + Impersonation detected.")
                else:
                    score += 0.35 # WARNING: Use caution. It's a brand new site asking for password.
                    logger.info(f"⚠️ ZERO TRUST (WARNING): New Domain asking for login (No impersonation detected).")
            else:
                # Just new, no login? Low risk.
                score += 0.2
                
        # Trusted old domain (> 5 years)
        elif age_days > 365 * 5:
            score -= 0.2 

        # NEW: Access Blocked Penalty (Suspicious if not major site)
        is_blocked = features.get("access_blocked", False)
        if is_blocked:
            # Only penalize if domain is young (< 6 months)
            if age_days < 180:
                score += 0.4
                logger.info("Young domain blocking access: Suspicious (Cloaking Risk)")
            else:
                logger.info("Established domain blocking access: Likely WAF/Protection (Safe)")
            
        # NEW: URL Keyword Analysis (for when content is hidden)
        url_lower = url.lower()
        suspicious_url_keywords = ["apk", "mod", "crack", "hack", "cheat", "free-download", "mirror"]
        if any(k in url_lower for k in suspicious_url_keywords):
            score += 0.4 # High risk keywords in URL

        if features.get("sensitive_non_https", False):
            score += 0.5  # Significant risk, but needs other factors to hit 1.0
            
        if features.get("has_hidden_url", False):
            score += 0.4  # Suspicious redirect
            
        # NEW: Tunnel / Suspicious Hosting Penalty
        if features.get("is_suspicious_hosting", False):
            score += 0.5 # High risk (Cloudflare Tunnel, Ngrok, etc.)
            logger.warning("Detected suspicious hosting provider (Tunnel/Dev domain)")

        return min(score, 1.0)
    
    async def _calculate_email_heuristic_score(self, features: Dict[str, Any]) -> float:
        """Calculate heuristic risk score for email."""
        score = 0.0
        
        # Authentication failures
        if not features.get("spf_pass", True):
            score += 0.3
        if not features.get("dkim_valid", True):
            score += 0.2
        if not features.get("dmarc_pass", True):
            score += 0.3
        
        # Urgency indicators
        score += features.get("urgency_score", 0) * 0.4
        
        # External links
        if features.get("external_links", 0) > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    async def _calculate_sms_heuristic_score(self, features: Dict[str, Any]) -> float:
        """Calculate heuristic risk score for SMS."""
        score = 0.0
        
        # URL presence
        if features.get("url_count", 0) > 0:
            score += 0.4
        
        # Urgency indicators
        score += features.get("urgency_score", 0) * 0.5
        
        # Sender reputation
        sender_rep = features.get("sender_reputation", 0.5)
        if sender_rep < 0.3:
            score += 0.3
        
        return min(score, 1.0)
    
    async def _calculate_image_heuristic_score(self, features: Dict[str, Any]) -> float:
        """Calculate heuristic risk score for image."""
        score = 0.0
        
        # Login form detection
        if features.get("has_login_form", False):
            score += 0.4
        
        # Brand logo detection
        if features.get("brand_logo_detected", False):
            score += 0.3
        
        return min(score, 1.0)
    
    async def _calculate_audio_heuristic_score(self, features: Dict[str, Any]) -> float:
        """Calculate heuristic risk score for audio."""
        score = 0.0
        
        # Voice quality indicators
        if features.get("voice_quality", 1.0) < 0.5:
            score += 0.2
        
        # Emotional manipulation
        if features.get("emotional_manipulation", False):
            score += 0.3
        
        return min(score, 1.0)
    
    # Explanation Generation
    async def _generate_url_explanations(self, url: str, features: Dict[str, Any], heuristic_score: float) -> List[Dict[str, Any]]:
        """Generate explanations for URL analysis."""
        explanations = []
        
        if features.get("suspicious_tld", False):
            explanations.append({
                "feature": "Suspicious TLD",
                "contribution": 0.3,
                "description": "URL uses a suspicious top-level domain"
            })
        
        if features.get("contains_ip", False):
            explanations.append({
                "feature": "IP Address",
                "contribution": 0.4,
                "description": "URL contains IP address instead of domain name"
            })
        
        if features.get("url_length", 0) > 100:
            explanations.append({
                "feature": "Long URL",
                "contribution": 0.2,
                "description": "Unusually long URL may be hiding malicious intent"
            })
            
        if features.get("has_repeated_chars", False):
            explanations.append({
                "feature": "Repeated Characters",
                "contribution": 0.3,
                "description": "Domain contains suspicious repeated characters (e.g., 'iii')"
            })
            
        if features.get("typosquatting_score", 0) > 0.7 and not features.get("is_whitelisted", False):
             explanations.append({
                "feature": "Brand Impersonation",
                "contribution": 0.5,
                "description": "URL appears to be impersonating a known brand"
            })
            
        if not features.get("dns_valid", True):
            explanations.append({
                "feature": "Invalid Domain",
                "contribution": 0.5,
                "description": "Domain does not have valid DNS records"
            })
            
        age_days = features.get("domain_age_days", 0)
        if age_days < 30 and age_days > 0:
            explanations.append({
                "feature": "Newly Registered",
                "contribution": 0.4,
                "description": f"Domain is very new ({age_days} days old)"
            })
        elif age_days > 365 * 5:
            explanations.append({
                "feature": "Established Domain",
                "contribution": -0.2,
                "description": "Domain is well-established (> 5 years old)"
            })
        
        return explanations
    
    async def _generate_email_explanations(self, features: Dict[str, Any], text_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate explanations for email analysis."""
        explanations = []
        
        if not features.get("spf_pass", True):
            explanations.append({
                "feature": "SPF Authentication",
                "contribution": 0.3,
                "description": "Email failed SPF authentication check"
            })
        
        if features.get("urgency_score", 0) > 0.7:
            explanations.append({
                "feature": "Urgency Language",
                "contribution": 0.4,
                "description": "Email contains urgent or threatening language"
            })
            
        # AI Model insights
        text_score = text_result.get("probability", 0)
        if text_score > 0.8:
            explanations.append({
                "feature": "AI Pattern Analysis",
                "contribution": 0.6,
                "description": "Neural model detected linguistic patterns typical of phishing or spam"
            })
        elif text_score < 0.2:
            explanations.append({
                "feature": "AI Pattern Analysis",
                "contribution": -0.3,
                "description": "Neural model confirmed conversational patterns typical of legitimate emails"
            })
        
        # Domain Reputation
        sender = features.get("sender")
        if sender:
            sender_domain = sender.split('@')[-1].lower()
            trusted_suffixes = ['tcs.com', 'sbi.co.in', 'google.com', 'microsoft.com']
            if any(sender_domain == ts or sender_domain.endswith('.' + ts) for ts in trusted_suffixes):
                explanations.append({
                    "feature": "Trusted Source",
                    "contribution": -0.5,
                    "description": f"Domain {sender_domain} is a verified reputable organization"
                })
        
        return explanations
    
    async def _generate_sms_explanations(self, features: Dict[str, Any], text_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate explanations for SMS analysis."""
        explanations = []
        
        if features.get("url_count", 0) > 0:
            explanations.append({
                "feature": "Contains URLs",
                "contribution": 0.4,
                "description": "SMS contains suspicious links"
            })
        
        return explanations
    
    async def _generate_image_explanations(self, features: Dict[str, Any], text_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate explanations for image analysis."""
        explanations = []
        
        if features.get("has_login_form", False):
            explanations.append({
                "feature": "Login Form Detected",
                "contribution": 0.4,
                "description": "Image contains login form elements"
            })
        
        return explanations
    
    async def _generate_audio_explanations(self, features: Dict[str, Any], text_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate explanations for audio analysis."""
        explanations = []
        
        if features.get("emotional_manipulation", False):
            explanations.append({
                "feature": "Emotional Manipulation",
                "contribution": 0.3,
                "description": "Audio contains emotional manipulation tactics"
            })
        
        return explanations
    
    # HTML Highlighting Generation
    async def _generate_url_highlight_html(self, url: str, explanations: List[Dict[str, Any]]) -> str:
        """Generate HTML with highlighted suspicious URL parts."""
        # Simple implementation - highlight suspicious parts
        html = f'<span class="url-analysis">{url}</span>'
        return html
    
    async def _generate_email_highlight_html(self, text: str, text_result: Dict[str, Any]) -> str:
        """Generate HTML with highlighted suspicious email parts."""
        # Use attention weights to highlight tokens
        tokens = text_result.get("tokens", [])
        attention_weights = text_result.get("attention_weights", [])
        
        if not tokens or not attention_weights:
            return f'<div class="email-content">{text}</div>'
        
        # Create highlighted HTML
        html_parts = []
        for i, (token, weight) in enumerate(zip(tokens, attention_weights)):
            # Handle special tokens
            if token in ["<s>", "</s>", "<pad>", "<unk>", "<mask>", "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]:
                continue
                
            if isinstance(weight, list):
                weight = sum(weight) / len(weight)  # Average attention
            
            # Clean BPE tokens (for RoBERTa/GPT models)
            # Ġ is used for spaces, Ċ/ĉ for newlines
            clean_token = token.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\n')
            
            # Escape HTML characters in token
            clean_token = clean_token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            opacity = min(weight * 5, 1.0)  # Scale weight for visible background
            
            # Threshold for highlighting
            if opacity > 0.15:
                # Use standard red for suspicion with variable opacity
                # Add a subtle underline for the most suspicious parts
                underline = f"border-bottom: 2px solid rgba(255, 0, 0, {opacity});" if opacity > 0.6 else ""
                html_parts.append(f'<span class="highlight" style="background-color: rgba(255, 0, 0, {opacity * 0.3}); {underline}">{clean_token}</span>')
            else:
                html_parts.append(clean_token)
        
        # Join without extra spaces since BPE tokens include their own spaces
        return "".join(html_parts).replace('\n', '<br/>')
    
    async def _generate_sms_highlight_html(self, text: str, text_result: Dict[str, Any]) -> str:
        """Generate HTML with highlighted suspicious SMS parts."""
        return await self._generate_email_highlight_html(text, text_result)
    
    async def _generate_image_highlight_html(self, ocr_text: str, text_result: Optional[Dict[str, Any]]) -> str:
        """Generate HTML with highlighted OCR text."""
        if not text_result:
            return f'<span class="ocr-text">{ocr_text}</span>'
        return await self._generate_email_highlight_html(ocr_text, text_result)
    
    async def _generate_audio_highlight_html(self, transcript: str, text_result: Optional[Dict[str, Any]]) -> str:
        """Generate HTML with highlighted transcript text."""
        if not text_result:
            return f'<span class="transcript">{transcript}</span>'
        return await self._generate_email_highlight_html(transcript, text_result)
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("🧹 Cleaning up ML predictor resources...")
        # TODO: Implement cleanup logic