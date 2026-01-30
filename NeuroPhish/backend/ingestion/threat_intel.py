"""
Threat Intelligence Manager for Unified Phishing Detection System
Integration with external threat feeds and intelligence sources.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib
import json
import os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ThreatIntelManager:
    """Manages threat intelligence feeds and URL reputation checks."""
    
    def __init__(self):
        self.feeds = {}
        self.cache = {}
        self.is_initialized = False
        
        # API configurations
        self.api_configs = {
            "phishtank": {
                "url": "http://data.phishtank.com/data/online-valid.json",
                "api_key": os.getenv("PHISHTANK_API_KEY"),
                "rate_limit": 1000,  # requests per hour
                "cache_ttl": 3600    # 1 hour
            },
            "virustotal": {
                "url": "https://www.virustotal.com/vtapi/v2/url/report",
                "api_key": os.getenv("VIRUSTOTAL_API_KEY"),
                "rate_limit": 4,     # requests per minute for free tier
                "cache_ttl": 1800    # 30 minutes
            },
            "urlhaus": {
                "url": "https://urlhaus-api.abuse.ch/v1/url/",
                "api_key": None,     # No API key required
                "rate_limit": 100,   # requests per minute
                "cache_ttl": 3600    # 1 hour
            }
        }
        
        # Request tracking for rate limiting
        self.request_history = {
            "phishtank": [],
            "virustotal": [],
            "urlhaus": []
        }
    
    async def initialize(self):
        """Initialize threat intelligence feeds."""
        try:
            logger.info("🔍 Initializing threat intelligence manager...")
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "PhishGuard/1.0"}
            )
            
            # Load initial threat feeds
            await self._load_initial_feeds()
            
            self.is_initialized = True
            logger.info("✅ Threat intelligence manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize threat intelligence: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if threat intelligence is ready."""
        return self.is_initialized
    
    async def check_url(self, url: str) -> Dict[str, Any]:
        """Check URL against all available threat intelligence sources."""
        try:
            logger.info(f"Checking threat intelligence for: {url}")
            
            # Check cache first
            cache_key = hashlib.sha256(url.encode()).hexdigest()
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if datetime.utcnow() - cached_result["timestamp"] < timedelta(minutes=30):
                    logger.info("Returning cached threat intelligence result")
                    return cached_result["data"]
            
            # Collect results from all sources
            results = {}
            
            # Check PhishTank
            phishtank_result = await self._check_phishtank(url)
            if phishtank_result:
                results["phishtank"] = phishtank_result
            
            # Check VirusTotal
            virustotal_result = await self._check_virustotal(url)
            if virustotal_result:
                results["virustotal"] = virustotal_result
            
            # Check URLhaus
            urlhaus_result = await self._check_urlhaus(url)
            if urlhaus_result:
                results["urlhaus"] = urlhaus_result
            
            # Aggregate results
            aggregated_result = self._aggregate_results(url, results)
            
            # Cache result
            self.cache[cache_key] = {
                "data": aggregated_result,
                "timestamp": datetime.utcnow()
            }
            
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Threat intelligence check failed: {e}")
            return {
                "url": url,
                "verdict": "unknown",
                "sources": [],
                "error": str(e)
            }
    
    async def _check_phishtank(self, url: str) -> Optional[Dict[str, Any]]:
        """Check URL against PhishTank database."""
        try:
            if not self._can_make_request("phishtank"):
                logger.warning("PhishTank rate limit reached")
                return None
            
            # PhishTank uses POST requests for URL checks
            data = {
                "url": url,
                "format": "json"
            }
            
            if self.api_configs["phishtank"]["api_key"]:
                data["app_key"] = self.api_configs["phishtank"]["api_key"]
            
            async with self.session.post(
                "http://checkurl.phishtank.com/checkurl/",
                data=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    self._track_request("phishtank")
                    
                    return {
                        "source": "phishtank",
                        "verdict": "malicious" if result.get("results", {}).get("in_database") else "clean",
                        "confidence": 0.9 if result.get("results", {}).get("valid") else 0.1,
                        "details": result.get("results", {})
                    }
                else:
                    logger.warning(f"PhishTank API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"PhishTank check failed: {e}")
            return None
    
    async def _check_virustotal(self, url: str) -> Optional[Dict[str, Any]]:
        """Check URL against VirusTotal."""
        try:
            if not self.api_configs["virustotal"]["api_key"]:
                logger.warning("VirusTotal API key not configured")
                return None
            
            if not self._can_make_request("virustotal"):
                logger.warning("VirusTotal rate limit reached")
                return None
            
            params = {
                "apikey": self.api_configs["virustotal"]["api_key"],
                "resource": url
            }
            
            async with self.session.get(
                self.api_configs["virustotal"]["url"],
                params=params
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    self._track_request("virustotal")
                    
                    # Parse VirusTotal response
                    positives = result.get("positives", 0)
                    total = result.get("total", 0)
                    
                    if total > 0:
                        malicious_ratio = positives / total
                        verdict = "malicious" if malicious_ratio > 0.1 else "suspicious" if malicious_ratio > 0 else "clean"
                        confidence = min(malicious_ratio * 2, 1.0) if malicious_ratio > 0 else 0.8
                    else:
                        verdict = "unknown"
                        confidence = 0.0
                    
                    return {
                        "source": "virustotal",
                        "verdict": verdict,
                        "confidence": confidence,
                        "details": {
                            "positives": positives,
                            "total": total,
                            "scan_date": result.get("scan_date")
                        }
                    }
                else:
                    logger.warning(f"VirusTotal API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"VirusTotal check failed: {e}")
            return None
    
    async def _check_urlhaus(self, url: str) -> Optional[Dict[str, Any]]:
        """Check URL against URLhaus database."""
        try:
            if not self._can_make_request("urlhaus"):
                logger.warning("URLhaus rate limit reached")
                return None
            
            data = {"url": url}
            
            async with self.session.post(
                self.api_configs["urlhaus"]["url"],
                data=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    self._track_request("urlhaus")
                    
                    query_status = result.get("query_status")
                    
                    if query_status == "ok":
                        # URL found in database
                        url_status = result.get("url_status", "")
                        threat_type = result.get("threat", "")
                        
                        verdict = "malicious" if url_status == "online" else "suspicious"
                        confidence = 0.9 if url_status == "online" else 0.6
                        
                        return {
                            "source": "urlhaus",
                            "verdict": verdict,
                            "confidence": confidence,
                            "details": {
                                "threat_type": threat_type,
                                "url_status": url_status,
                                "first_seen": result.get("date_added"),
                                "tags": result.get("tags", [])
                            }
                        }
                    else:
                        # URL not found - likely clean
                        return {
                            "source": "urlhaus",
                            "verdict": "clean",
                            "confidence": 0.7,
                            "details": {"query_status": query_status}
                        }
                else:
                    logger.warning(f"URLhaus API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"URLhaus check failed: {e}")
            return None
    
    def _can_make_request(self, source: str) -> bool:
        """Check if we can make a request without exceeding rate limits."""
        now = datetime.utcnow()
        config = self.api_configs[source]
        history = self.request_history[source]
        
        # Clean old requests from history
        if source == "virustotal":
            # 4 requests per minute
            cutoff = now - timedelta(minutes=1)
            self.request_history[source] = [req_time for req_time in history if req_time > cutoff]
            return len(self.request_history[source]) < config["rate_limit"]
        else:
            # Other sources have hourly limits
            cutoff = now - timedelta(hours=1)
            self.request_history[source] = [req_time for req_time in history if req_time > cutoff]
            return len(self.request_history[source]) < config["rate_limit"]
    
    def _track_request(self, source: str):
        """Track a request for rate limiting purposes."""
        self.request_history[source].append(datetime.utcnow())
    
    def _aggregate_results(self, url: str, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple threat intelligence sources."""
        if not results:
            return {
                "url": url,
                "verdict": "unknown",
                "sources": [],
                "confidence": 0.0,
                "tags": []
            }
        
        # Collect verdicts and confidences
        verdicts = []
        confidences = []
        sources = []
        tags = []
        
        for source, result in results.items():
            verdict = result.get("verdict", "unknown")
            confidence = result.get("confidence", 0.0)
            
            verdicts.append(verdict)
            confidences.append(confidence)
            sources.append(source)
            
            # Collect tags
            details = result.get("details", {})
            if "tags" in details:
                tags.extend(details["tags"])
        
        # Determine overall verdict
        malicious_count = verdicts.count("malicious")
        suspicious_count = verdicts.count("suspicious")
        clean_count = verdicts.count("clean")
        
        if malicious_count > 0:
            overall_verdict = "malicious"
            overall_confidence = max(confidences)
        elif suspicious_count > 0:
            overall_verdict = "suspicious"
            overall_confidence = sum(confidences) / len(confidences)
        elif clean_count > 0:
            overall_verdict = "clean"
            overall_confidence = sum(confidences) / len(confidences)
        else:
            overall_verdict = "unknown"
            overall_confidence = 0.0
        
        return {
            "url": url,
            "verdict": overall_verdict,
            "confidence": overall_confidence,
            "sources": sources,
            "source_results": results,
            "tags": list(set(tags)),
            "first_seen": None,  # TODO: Extract from source results
            "last_seen": datetime.utcnow()
        }
    
    async def _load_initial_feeds(self):
        """Load initial threat intelligence feeds."""
        try:
            logger.info("Loading initial threat intelligence feeds...")
            
            # Load PhishTank feed (if available)
            if self.api_configs["phishtank"]["api_key"]:
                await self._load_phishtank_feed()
            
            logger.info("Initial threat feeds loaded")
            
        except Exception as e:
            logger.warning(f"Failed to load initial feeds: {e}")
    
    async def _load_phishtank_feed(self):
        """Load PhishTank bulk feed."""
        try:
            async with self.session.get(self.api_configs["phishtank"]["url"]) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Process PhishTank entries
                    phish_urls = {}
                    for entry in data:
                        url = entry.get("url", "")
                        if url:
                            phish_urls[url] = {
                                "phish_id": entry.get("phish_id"),
                                "submission_time": entry.get("submission_time"),
                                "verified": entry.get("verified") == "yes",
                                "online": entry.get("online") == "yes"
                            }
                    
                    self.feeds["phishtank"] = phish_urls
                    logger.info(f"Loaded {len(phish_urls)} PhishTank entries")
                    
        except Exception as e:
            logger.error(f"Failed to load PhishTank feed: {e}")
    
    async def update_feeds(self):
        """Update threat intelligence feeds."""
        try:
            logger.info("Updating threat intelligence feeds...")
            await self._load_initial_feeds()
            logger.info("Threat intelligence feeds updated")
            
        except Exception as e:
            logger.error(f"Failed to update feeds: {e}")
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'session') and self.session:
                await self.session.close()
            logger.info("✅ Threat intelligence manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")