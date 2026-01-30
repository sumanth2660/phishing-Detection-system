"""
Explainability Engine for Unified Phishing Detection System
SHAP-based model explanations and interpretability features.
"""

try:
    import shap
except ImportError:
    shap = None
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """Comprehensive explainability engine using SHAP and custom methods."""
    
    def __init__(self):
        self.explainers = {}
        self.feature_names = {}
        self.is_initialized = False
        
        # Configuration for different model types
        self.explainer_configs = {
            "transformer": {
                "method": "attention",
                "max_tokens": 512
            },
            "xgboost": {
                "method": "shap_tree",
                "max_features": 100
            },
            "ensemble": {
                "method": "shap_linear",
                "combine_explanations": True
            }
        }
    
    async def initialize(self):
        """Initialize explainability components."""
        try:
            logger.info("🔍 Initializing explainability engine...")
            
            # Initialize SHAP
            if shap:
                shap.initjs()
            else:
                logger.warning("⚠️ SHAP not installed. Explainability features limited.")
            
            # Set up feature importance tracking
            self.feature_importance_history = []
            
            self.is_initialized = True
            logger.info("✅ Explainability engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize explainability engine: {e}")
            raise
    
    def _calculate_simple_importance(self, features: Dict[str, Any], prediction: float) -> Dict[str, float]:
        """Calculate simple feature importance based on correlation with prediction."""
        importance = {}
        
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, (int, float, bool)):
                # Simple importance: feature value * prediction (normalized)
                raw_importance = abs(float(feature_value)) * prediction
                
                # Normalize by feature type
                if isinstance(feature_value, bool):
                    importance[feature_name] = raw_importance if feature_value else 0
                else:
                    # Scale numeric features
                    importance[feature_name] = min(raw_importance, 1.0)
        
        return importance
    
    async def _generate_visualizations(self, explanations: Dict[str, Any], content_type: str) -> Dict[str, str]:
        """Generate visualization plots for explanations."""
        try:
            visualizations = {}
            
            # Feature importance bar chart
            if "feature_importance" in explanations:
                importance_data = explanations["feature_importance"]
                if importance_data:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Sort features by importance
                    sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:10]
                    features, values = zip(*sorted_features)
                    
                    bars = ax.barh(features, values, color='skyblue')
                    ax.set_xlabel('Feature Importance')
                    ax.set_title(f'Top 10 Feature Importance - {content_type.title()}')
                    ax.grid(axis='x', alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{value:.3f}', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    
                    # Convert to base64 string
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    visualizations["feature_importance"] = f"data:image/png;base64,{img_str}"
                    
                    plt.close()
            
            # Token attribution visualization for transformer models
            if "token_attributions" in explanations:
                token_data = explanations["token_attributions"]
                if token_data and len(token_data) > 0:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    
                    tokens = [item["token"] for item in token_data[:20]]  # Limit to 20 tokens
                    attributions = [item["attribution"] for item in token_data[:20]]
                    
                    colors = ['red' if attr > 0.3 else 'orange' if attr > 0.1 else 'lightgray' 
                             for attr in attributions]
                    
                    bars = ax.bar(range(len(tokens)), attributions, color=colors)
                    ax.set_xticks(range(len(tokens)))
                    ax.set_xticklabels(tokens, rotation=45, ha='right')
                    ax.set_ylabel('Attribution Score')
                    ax.set_title('Token-Level Attention Weights')
                    ax.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    visualizations["token_attributions"] = f"data:image/png;base64,{img_str}"
                    
                    plt.close()
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {}
    
    async def _generate_natural_explanation(self, explanations: Dict[str, Any], prediction: float, content_type: str) -> str:
        """Generate natural language explanation of the prediction."""
        try:
            risk_level = "low"
            if prediction > 0.8:
                risk_level = "critical"
            elif prediction > 0.6:
                risk_level = "high"
            elif prediction > 0.3:
                risk_level = "medium"
            
            explanation_parts = [
                f"This {content_type} has been classified as {risk_level} risk with a confidence score of {prediction:.1%}."
            ]
            
            # Add specific explanations based on triggered rules
            if "triggered_rules" in explanations:
                rules = explanations["triggered_rules"]
                if rules:
                    explanation_parts.append("The main reasons for this classification are:")
                    for rule in rules[:3]:  # Top 3 rules
                        contribution = rule.get("contribution", 0)
                        description = rule.get("description", "")
                        explanation_parts.append(f"• {description} (contributes {contribution:.1%} to the risk score)")
            
            # Add feature importance insights
            if "feature_importance" in explanations:
                importance = explanations["feature_importance"]
                if importance:
                    top_feature = max(importance.items(), key=lambda x: x[1])
                    explanation_parts.append(f"The most significant indicator is '{top_feature[0]}' with an importance score of {top_feature[1]:.3f}.")
            
            # Add recommendations based on risk level
            if risk_level in ["high", "critical"]:
                explanation_parts.append("⚠️ Recommendation: Exercise extreme caution. Do not click links, download attachments, or provide personal information.")
            elif risk_level == "medium":
                explanation_parts.append("⚠️ Recommendation: Be cautious and verify the source before taking any action.")
            else:
                explanation_parts.append("✅ This appears to be legitimate, but always remain vigilant.")
            
            return " ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Natural explanation generation failed: {e}")
            return f"Unable to generate detailed explanation. Risk score: {prediction:.1%}"