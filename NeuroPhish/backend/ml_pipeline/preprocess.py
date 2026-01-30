"""
Feature extraction and preprocessing for Unified Phishing Detection System.
Comprehensive feature engineering for multi-modal phishing detection.
"""

import re
import math
import hashlib
import tldextract
import urllib.parse
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np
from bs4 import BeautifulSoup
import email
from email.mime.text import MIMEText
import dns.resolver
import whois
import logging
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Comprehensive feature extraction for all content types."""
    
    def __init__(self):
        # Suspicious TLD list
        self.suspicious_tlds = {
            'tk', 'ml', 'ga', 'cf', 'top', 'click', 'download', 'stream',
            'science', 'racing', 'review', 'country', 'kim', 'cricket'
        }
        
        # Popular brands for typosquatting detection
        self.popular_brands = {
            'google', 'facebook', 'amazon', 'microsoft', 'apple', 'netflix',
            'paypal', 'ebay', 'instagram', 'twitter', 'linkedin', 'youtube',
            'gmail', 'yahoo', 'outlook', 'dropbox', 'adobe', 'spotify',
            'whatsapp', 'telegram', 'tiktok', 'roblox', 'twitch',
            'sbi', 'onlinesbi', 'hdfc', 'icici', 'axisbank', 'kotak', 'pnb',
            'paytm', 'phonepe', 'gpay', 'bhim', 'upi', 'nissan', 'toyota',
            'ford', 'honda', 'chevrolet', 'tesla', 'bmw', 'mercedes', 'audi',
            'tcs', 'tata', 'infosys', 'wipro', 'hcl', 'accenture'
        }
        
        # Phishing keywords
        self.urgency_keywords = {
            'urgent', 'immediate', 'expires', 'suspended', 'verify', 'confirm',
            'update', 'secure', 'act now', 'limited time', 'click here',
            'congratulations', 'winner', 'prize', 'free', 'bonus'
        }
        
        # Suspicious patterns
        self.suspicious_patterns = {
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'ssn': r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }

        # Tunneling and dynamic DNS services often used for phishing
        self.suspicious_hosting = {
            'trycloudflare.com', 'ngrok.io', 'localtunnel.me', 'herokuapp.com', 
            'firebasestorage.googleapis.com', 'ipfs.io', 'workers.dev', 'pages.dev',
            'vercel.app', 'netlify.app', 'github.io', 'gitlab.io', 'surge.sh'
        }
    
    async def extract_url_features(self, url: str) -> Dict[str, Any]:
        """Extract comprehensive features from URL."""
        try:
            features = {}
            
            # Basic URL parsing
            parsed = urllib.parse.urlparse(url)
            extracted = tldextract.extract(url)
            
            # Basic features
            features['url'] = url
            features['url_length'] = len(url)
            
            # Smart Long URL Analysis
            # Don't just blindly flag length. Check if it's hiding a URL inside.
            # Example: http://safe.com/redirect?url=http://evil.com
            potential_hidden_url = re.search(r'(http[s]?://.+)', url[10:]) # Look for http after the start
            features['has_hidden_url'] = bool(potential_hidden_url)
            features['hidden_url_target'] = potential_hidden_url.group(1) if potential_hidden_url else None
            features['domain'] = extracted.domain
            features['subdomain'] = extracted.subdomain
            features['tld'] = extracted.suffix
            features['path'] = parsed.path
            features['query'] = parsed.query
            features['scheme'] = parsed.scheme
            
            # URL structure features
            features['subdomain_count'] = len(extracted.subdomain.split('.')) if extracted.subdomain else 0
            features['path_length'] = len(parsed.path)
            features['query_length'] = len(parsed.query)
            features['fragment_length'] = len(parsed.fragment) if parsed.fragment else 0
            
            # Character analysis
            features['digit_count'] = sum(c.isdigit() for c in url)
            features['letter_count'] = sum(c.isalpha() for c in url)
            features['special_char_count'] = len(url) - features['digit_count'] - features['letter_count']
            
            # Entropy calculation
            features['entropy'] = self._calculate_entropy(url)
            features['domain_entropy'] = self._calculate_entropy(extracted.domain) if extracted.domain else 0
            
            # Suspicious indicators
            features['contains_ip'] = self._contains_ip_address(url)
            features['suspicious_tld'] = extracted.suffix.lower() in self.suspicious_tlds
            features['has_at_symbol'] = '@' in url
            features['has_dash_in_domain'] = '-' in extracted.domain if extracted.domain else False
            features['has_repeated_chars'] = self._has_repeated_characters(extracted.domain) if extracted.domain else False
            
            # Typosquatting detection (Check BOTH Domain and Subdomain)
            # Hackers use subdomains on tunnels e.g., 'nissan.trycloudflare.com'
            full_check_string = f"{extracted.subdomain}.{extracted.domain}" if extracted.subdomain else extracted.domain
            features['typosquatting_score'] = self._calculate_typosquatting_score(full_check_string)
            
            # Hosting / Tunnel Analysis (NEW)
            fqdn = extracted.fqdn.lower()
            features['is_suspicious_hosting'] = any(h in fqdn for h in self.suspicious_hosting)
            
            # URL shortener detection
            features['is_shortened'] = self._is_url_shortener(extracted.domain)
            
            # HTTPS usage
            features['uses_https'] = parsed.scheme == 'https'
            
            # Extended HTTPS check: Flag if it's a login page but NOT https
            is_sensitive_page = self._detect_login_form(features.get('extracted_text', '')) or \
                                any(x in url.lower() for x in ['login', 'signin', 'bank', 'secure', 'account', 'update'])
            
            features['sensitive_non_https'] = is_sensitive_page and not features['uses_https']
            
            # Port analysis
            features['uses_non_standard_port'] = parsed.port not in [None, 80, 443]
            features['port'] = parsed.port
            
            # Path analysis
            features['path_depth'] = len([p for p in parsed.path.split('/') if p])
            features['has_suspicious_path'] = self._has_suspicious_path(parsed.path)
            
            # Query parameter analysis
            query_params = urllib.parse.parse_qs(parsed.query)
            features['query_param_count'] = len(query_params)
            features['has_redirect_param'] = any(param in query_params for param in ['redirect', 'url', 'goto', 'next'])
            
            # Domain age and reputation
            whois_info = await self._get_whois_info(extracted.fqdn)
            features['domain_age_days'] = whois_info.get('age_days', 0)
            features['registrar'] = whois_info.get('registrar', 'Unknown')
            features['country'] = whois_info.get('country', 'Unknown')
            features['domain_reputation'] = 0.5  # TODO: Implement reputation check
            
            # DNS Analysis
            # Use full FQDN for DNS check, not just the domain part
            dns_valid = await self._check_dns_records(extracted.fqdn)
            features['dns_valid'] = dns_valid
            
            # Try to fetch content for analysis
            try:
                content_features = await self._extract_url_content_features(url)
                features.update(content_features)
            except Exception as e:
                logger.warning(f"Could not fetch URL content: {e}")
                features.update({
                    'html_raw': '',
                    'extracted_text': '',
                    'title': '',
                    'has_forms': False,
                    'external_links': 0,
                    'iframe_count': 0
                })
            
            return features
            
        except Exception as e:
            logger.error(f"URL feature extraction failed: {e}")
            return {'url': url, 'error': str(e)}
    
    async def extract_email_features(self, subject: str, body: str, sender: Optional[str] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract comprehensive features from email."""
        try:
            features = {}
            
            # Basic features
            features['subject'] = subject
            features['body'] = body
            features['sender'] = sender
            features['subject_length'] = len(subject)
            features['body_length'] = len(body)
            
            # Text analysis
            combined_text = f"{subject} {body}"
            features['total_length'] = len(combined_text)
            features['word_count'] = len(combined_text.split())
            
            # Urgency analysis
            features['urgency_score'] = self._calculate_urgency_score(combined_text)
            
            # HTML analysis if body contains HTML
            if '<html>' in body.lower() or '<body>' in body.lower():
                html_features = self._extract_html_features(body)
                features.update(html_features)
                features['is_html'] = True
            else:
                features['is_html'] = False
                features['html_to_text_ratio'] = 0
            
            # Link analysis
            link_features = self._extract_link_features(body)
            features.update(link_features)
            
            # Sender analysis
            if sender:
                sender_features = self._extract_sender_features(sender)
                features.update(sender_features)
            
            # Header analysis
            if headers:
                header_features = self._extract_header_features(headers)
                features.update(header_features)
            else:
                # Default values
                features.update({
                    'spf_pass': True,
                    'dkim_valid': True,
                    'dmarc_pass': True,
                    'received_count': 0
                })
            
            # Pattern matching
            pattern_features = self._extract_pattern_features(combined_text)
            features.update(pattern_features)
            
            # Language analysis
            features['language'] = 'en'  # TODO: Implement language detection
            features['has_mixed_languages'] = False
            
            # Attachment analysis (placeholder)
            features['attachment_count'] = 0
            features['has_executable_attachments'] = False
            
            return features
            
        except Exception as e:
            logger.error(f"Email feature extraction failed: {e}")
            return {'subject': subject, 'error': str(e)}
    
    async def extract_sms_features(self, text: str, sender: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract comprehensive features from SMS."""
        try:
            features = {}
            
            # Basic features
            features['text'] = text
            features['sender'] = sender
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            features['char_count'] = len(text)
            
            # Character analysis
            features['digit_count'] = sum(c.isdigit() for c in text)
            features['upper_count'] = sum(c.isupper() for c in text)
            features['special_char_count'] = sum(not c.isalnum() and not c.isspace() for c in text)
            
            # URL analysis
            urls = re.findall(self.suspicious_patterns['url'], text)
            features['url_count'] = len(urls)
            features['urls'] = urls
            features['has_shortened_url'] = any(self._is_url_shortener_simple(url) for url in urls)
            
            # Urgency analysis
            features['urgency_score'] = self._calculate_urgency_score(text)
            
            # Pattern matching
            pattern_features = self._extract_pattern_features(text)
            features.update(pattern_features)
            
            # Sender analysis
            if sender:
                features['sender_is_shortcode'] = sender.isdigit() and len(sender) <= 6
                features['sender_is_alpha'] = sender.isalpha()
                features['sender_length'] = len(sender)
            else:
                features['sender_is_shortcode'] = False
                features['sender_is_alpha'] = False
                features['sender_length'] = 0
            
            # Metadata analysis
            if metadata:
                features['carrier'] = metadata.get('carrier', '')
                features['country'] = metadata.get('country', '')
                features['timestamp'] = metadata.get('timestamp')
            
            # Reputation (placeholder)
            features['sender_reputation'] = 0.5  # TODO: Implement sender reputation
            
            return features
            
        except Exception as e:
            logger.error(f"SMS feature extraction failed: {e}")
            return {'text': text, 'error': str(e)}
    
    async def extract_image_features(self, image: Any, ocr_text: str) -> Dict[str, Any]:
        """Extract comprehensive features from image."""
        try:
            features = {}
            
            # Basic image properties
            features['width'] = image.width
            features['height'] = image.height
            features['aspect_ratio'] = image.width / image.height if image.height > 0 else 0
            features['format'] = image.format
            features['mode'] = image.mode
            
            # OCR text analysis
            features['ocr_text'] = ocr_text
            features['ocr_text_length'] = len(ocr_text)
            features['ocr_word_count'] = len(ocr_text.split()) if ocr_text else 0
            
            # Convert to OpenCV format for analysis
            cv_image = None
            if cv2 is not None:
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Color analysis
                color_features = self._extract_color_features(cv_image)
                features.update(color_features)
                
                # Form detection (simple heuristic)
                features['has_input_fields'] = self._detect_input_fields(cv_image)
            else:
                features['has_input_fields'] = False
            
            # Form detection (text based)
            features['has_login_form'] = self._detect_login_form(ocr_text)
            
            # Brand logo detection (placeholder)
            features['brand_logo_detected'] = False  # TODO: Implement logo detection
            features['detected_brands'] = []
            
            # Text analysis from OCR
            if ocr_text:
                text_features = self._extract_pattern_features(ocr_text)
                features.update({f'ocr_{k}': v for k, v in text_features.items()})
                
                features['ocr_urgency_score'] = self._calculate_urgency_score(ocr_text)
            
            return features
            
        except Exception as e:
            logger.error(f"Image feature extraction failed: {e}")
            return {'error': str(e)}
    
    async def extract_audio_features(self, audio_data: bytes, transcript: str) -> Dict[str, Any]:
        """Extract comprehensive features from audio."""
        try:
            features = {}
            
            # Basic audio properties
            features['audio_size'] = len(audio_data)
            features['transcript'] = transcript
            features['transcript_length'] = len(transcript)
            features['transcript_word_count'] = len(transcript.split()) if transcript else 0
            
            # Audio analysis (placeholder - requires librosa)
            features['duration'] = 0  
            features['sample_rate'] = 0  
            features['voice_quality'] = 1.0  
            
            # TODO: Implement real audio features here with proper try-except wrapping
            # For now, just ensuring it doesn't crash on m4a
            
            # Transcript analysis
            if transcript:
                text_features = self._extract_pattern_features(transcript)
                features.update({f'transcript_{k}': v for k, v in text_features.items()})
                
                features['transcript_urgency_score'] = self._calculate_urgency_score(transcript)
                
                # Emotional manipulation detection
                features['emotional_manipulation'] = self._detect_emotional_manipulation(transcript)
            
            # Speech patterns (placeholder)
            features['speaking_rate'] = 0  # TODO: Calculate words per minute
            features['pause_frequency'] = 0  # TODO: Analyze pauses
            
            return features
            
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            return {'transcript': transcript, 'error': str(e)}
    
    # Helper methods
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0
        text_length = len(text)
        for count in char_counts.values():
            probability = count / text_length
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _contains_ip_address(self, url: str) -> bool:
        """Check if URL contains IP address instead of domain."""
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        return bool(re.search(ip_pattern, url))
    
    def _calculate_typosquatting_score(self, domain_text: str) -> float:
        """Calculate typosquatting similarity score against popular brands."""
        if not domain_text:
            return 0
        
        domain_lower = domain_text.lower()
        max_similarity = 0
        
        for brand in self.popular_brands:
            # Method 1: Direct containment (e.g. "nissan-update" contains "nissan")
            if brand in domain_lower:
                # If the brand is clearly visible in the text, it's a strong match (1.0)
                # unless it IS the brand domain itself (which we can't easily assert here without allowlist, 
                # but for 'trycloudflare', it's definitely phishing).
                return 1.0 
                
            # Method 2: Levenshtein (for "g00gle")
            # Only run strict Levenshtein on parts to avoid noise
            similarity = self._levenshtein_similarity(domain_lower, brand)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_similarity(s2, s1)
        
        if len(s2) == 0:
            return 0
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        distance = previous_row[-1]
        max_len = max(len(s1), len(s2))
        return 1 - (distance / max_len) if max_len > 0 else 0
    
    def _is_url_shortener(self, domain: str) -> bool:
        """Check if domain is a known URL shortener."""
        shorteners = {
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
            'short.link', 'tiny.cc', 'is.gd', 'buff.ly'
        }
        return domain.lower() in shorteners
    
    def _is_url_shortener_simple(self, url: str) -> bool:
        """Simple check for URL shorteners."""
        shortener_patterns = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly']
        return any(pattern in url.lower() for pattern in shortener_patterns)
    
    def _has_suspicious_path(self, path: str) -> bool:
        """Check if URL path contains suspicious elements."""
        suspicious_elements = ['admin', 'login', 'signin', 'verify', 'secure', 'update']
        return any(element in path.lower() for element in suspicious_elements)

    def _has_repeated_characters(self, text: str) -> bool:
        """Check for suspicious repeated characters (e.g., 'iii', 'ooo')."""
        if not text:
            return False
        # Check for 3 or more identical consecutive characters
        return bool(re.search(r'(.)\1\1', text))
    
    async def _extract_url_content_features(self, url: str) -> Dict[str, Any]:
        """Extract features from URL content by fetching the page."""
        try:
            import aiohttp
            import asyncio
            
            # Timeout for fetching
            timeout = aiohttp.ClientTimeout(total=8) # Increased timeout
            
            # Mimic a real Chrome browser on macOS to bypass 403 WAFs
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1'
            }
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, ssl=False) as response:
                    # If 403/401, it might be protected content or anti-bot
                    if response.status in [403, 401, 503]:
                         logger.warning(f"Access blocked to {url} (Status {response.status}). WAF/Cloudflare detected.")
                         # Return specific flag for suspicious blocking
                         return {
                            'html_raw': '', 'extracted_text': '', 'title': '',
                            'has_forms': False, 'external_links': 0, 'script_count': 0,
                            'access_blocked': True, # New flag
                            'status_code': response.status
                        }
                    
                    # Read content
                    html = await response.text(errors='ignore')
                    
                    # Limit processing size
                    if len(html) > 1_000_000:
                        html = html[:1_000_000]
                    
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove scripts and styles for text extraction
                    for script in soup(["script", "style"]):
                        script.decompose()
                        
                    clean_text = soup.get_text(separator=' ', strip=True)
                    title = soup.title.string if soup.title else ""
                    
                    # Log snippet
                    logger.info(f"Fetched content from {url}: Title='{title}', Text Length={len(clean_text)}")
                    
                    return {
                        'html_raw': html[:10000],  # Store truncated raw HTML
                        'extracted_text': clean_text,
                        'title': title,
                        'has_forms': len(soup.find_all('form')) > 0,
                        'external_links': len([a for a in soup.find_all('a', href=True) if 'http' in a['href']]),
                        'iframe_count': len(soup.find_all('iframe')),
                        'script_count': len(soup.find_all('script')),
                        'meta_refresh': bool(soup.find('meta', attrs={'http-equiv': re.compile(r'refresh', re.I)}))
                    }
                    
        except Exception as e:
            logger.warning(f"Could not fetch URL content: {e}")
            return {
                'html_raw': '',
                'extracted_text': '',
                'title': '',
                'has_forms': False,
                'external_links': 0,
                'iframe_count': 0,
                'script_count': 0,
                'meta_refresh': False
            }
    
    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency score based on keywords and patterns."""
        text_lower = text.lower()
        
        # Count urgency keywords
        urgency_count = sum(1 for keyword in self.urgency_keywords if keyword in text_lower)
        
        # Additional urgency indicators
        urgency_patterns = [
            r'\b(expires?|expir(ing|ed))\b',
            r'\b(suspend(ed)?|suspension)\b',
            r'\b(immediate(ly)?|urgent(ly)?)\b',
            r'\b(act now|click (here|now))\b',
            r'\b(limited time|hurry)\b'
        ]
        
        pattern_count = sum(1 for pattern in urgency_patterns if re.search(pattern, text_lower))
        
        # Normalize score
        total_words = len(text.split())
        if total_words == 0:
            return 0
        
        score = (urgency_count + pattern_count * 2) / total_words
        return min(score, 1.0)
    
    def _extract_html_features(self, html_content: str) -> Dict[str, Any]:
        """Extract features from HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            
            return {
                'html_to_text_ratio': len(html_content) / len(text) if text else 0,
                'form_count': len(soup.find_all('form')),
                'input_count': len(soup.find_all('input')),
                'link_count': len(soup.find_all('a')),
                'iframe_count': len(soup.find_all('iframe')),
                'script_count': len(soup.find_all('script')),
                'has_password_field': bool(soup.find('input', {'type': 'password'})),
                'external_links': len([a for a in soup.find_all('a', href=True) if 'http' in a['href']])
            }
        except Exception:
            return {
                'html_to_text_ratio': 0,
                'form_count': 0,
                'input_count': 0,
                'link_count': 0,
                'iframe_count': 0,
                'script_count': 0,
                'has_password_field': False,
                'external_links': 0
            }
    
    def _extract_link_features(self, text: str) -> Dict[str, Any]:
        """Extract link-related features from text."""
        urls = re.findall(self.suspicious_patterns['url'], text)
        
        return {
            'link_count': len(urls),
            'external_links': len([url for url in urls if 'http' in url]),
            'has_shortened_links': any(self._is_url_shortener_simple(url) for url in urls),
            'urls': urls
        }
    
    def _extract_sender_features(self, sender: str) -> Dict[str, Any]:
        """Extract features from email sender."""
        return {
            'sender_domain': sender.split('@')[1] if '@' in sender else '',
            'sender_local_length': len(sender.split('@')[0]) if '@' in sender else 0,
            'sender_has_numbers': any(c.isdigit() for c in sender),
            'sender_has_special_chars': any(c in '+-_.' for c in sender)
        }
    
    def _extract_header_features(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Extract features from email headers."""
        # Placeholder - implement actual header analysis
        return {
            'spf_pass': True,
            'dkim_valid': True,
            'dmarc_pass': True,
            'received_count': headers.get('received', '').count('Received:') if 'received' in headers else 0,
            'reply_to_mismatch': False  # TODO: Check if reply-to differs from sender
        }
    
    def _extract_pattern_features(self, text: str) -> Dict[str, Any]:
        """Extract pattern-based features from text."""
        features = {}
        
        for pattern_name, pattern in self.suspicious_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            features[f'has_{pattern_name}'] = len(matches) > 0
            features[f'{pattern_name}_count'] = len(matches)
        
        return features
    
    def _extract_color_features(self, cv_image) -> Dict[str, Any]:
        """Extract color-based features from image."""
        # Calculate color histogram
        hist_b = cv2.calcHist([cv_image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([cv_image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([cv_image], [2], None, [256], [0, 256])
        
        return {
            'dominant_color_blue': np.argmax(hist_b),
            'dominant_color_green': np.argmax(hist_g),
            'dominant_color_red': np.argmax(hist_r),
            'color_variance': np.var(cv_image),
            'brightness': np.mean(cv_image)
        }
    
    def _detect_login_form(self, ocr_text: str) -> bool:
        """Detect login form elements in OCR text."""
        login_indicators = [
            'username', 'password', 'login', 'sign in', 'email',
            'user id', 'account', 'submit', 'enter'
        ]
        text_lower = ocr_text.lower()
        return any(indicator in text_lower for indicator in login_indicators)
    
    def _detect_input_fields(self, cv_image) -> bool:
        """Detect input field-like rectangles in image."""
        # Simple edge detection to find rectangular shapes
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count rectangular contours that might be input fields
        rectangular_contours = 0
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Rectangle
                rectangular_contours += 1
        
        return rectangular_contours > 2  # Threshold for input fields
    
    def _detect_emotional_manipulation(self, text: str) -> bool:
        """Detect emotional manipulation in text."""
        emotional_keywords = [
            'fear', 'scared', 'worried', 'panic', 'emergency',
            'threat', 'danger', 'risk', 'lose', 'miss out',
            'regret', 'sorry', 'apologize', 'mistake'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in emotional_keywords)

    async def _get_whois_info(self, domain: str) -> Dict[str, Any]:
        """Get comprehensive WHOIS information."""
        if not domain:
            return {}
        try:
            # Run whois in thread pool to avoid blocking
            import asyncio
            import whois
            
            def get_whois():
                try:
                    return whois.whois(domain)
                except Exception as e:
                    logger.warning(f"WHOIS inner exception for {domain}: {e}")
                    return None
            
            w = await asyncio.to_thread(get_whois)
            
            if not w:
                logger.warning(f"WHOIS returned None for {domain}")
                return {}
            
            logger.info(f"WHOIS success for {domain}: {w}")
            
            info = {}
            
            # Creation date / Age
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            
            if isinstance(creation_date, datetime):
                # Handle timezone aware datetimes to prevent crash
                now = datetime.now()
                if creation_date.tzinfo:
                    now = now.astimezone(creation_date.tzinfo)
                    
                info['creation_date'] = creation_date
                info['age_days'] = max(0, (now - creation_date).days)
            else:
                info['age_days'] = -1 # Use -1 to indicate Unknown, not New (0)
                
            # Registrar
            info['registrar'] = w.registrar if w.registrar else "Unknown"
            
            # Country
            info['country'] = w.country if w.country else "Unknown"
            
            return info
            
        except Exception as e:
            logger.warning(f"WHOIS lookup failed for {domain}: {e}")
            return {}

    async def _check_dns_records(self, domain: str) -> bool:
        """Check if domain has valid DNS records."""
        if not domain:
            return False
        try:
            import asyncio
            import dns.resolver
            
            def check_dns():
                try:
                    dns.resolver.resolve(domain, 'A')
                    return True
                except Exception:
                    return False
            
            return await asyncio.to_thread(check_dns)
            
        except Exception as e:
            logger.warning(f"DNS lookup failed for {domain}: {e}")
            return False