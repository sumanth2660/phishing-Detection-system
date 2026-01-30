import cv2
import numpy as np
import imagehash
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class BrandDetector:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def compute_phash(self, image_bytes: bytes) -> str:
        """Compute perceptual hash of an image."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return str(imagehash.phash(image))
        except Exception as e:
            logger.error(f"Error computing pHash: {e}")
            return ""

    def compute_orb_features(self, image_bytes: bytes):
        """Compute ORB keypoints and descriptors."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            kp, des = self.orb.detectAndCompute(img, None)
            
            # Serialize for storage
            if des is not None:
                return des.tolist()
            return []
        except Exception as e:
            logger.error(f"Error computing ORB features: {e}")
            return None

    def compare_hashes(self, hash1: str, hash2: str, threshold: int = 10) -> bool:
        """Compare two pHashes. Returns True if similar."""
        try:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            return (h1 - h2) <= threshold
        except Exception:
            return False

    def detect_brand(self, image_bytes: bytes, known_brands: list) -> dict:
        """
        Compare image against known brands using ORB feature matching.
        Rough pHash filter is skipped because it fails on full screenshots vs logos.
        """
        input_descriptors = self.compute_descriptors(image_bytes)
        if input_descriptors is None:
             return {"detected": False, "brand": None, "method": "orb"}
             
        best_brand = None
        max_good_matches = 0
        MIN_MATCH_COUNT = 8  # Threshold for valid detection
        
        for brand in known_brands:
            brand_descriptors_list = brand.get('orb_features')
            if not brand_descriptors_list:
                continue
                
            # Convert back to numpy array
            try:
                brand_des = np.array(brand_descriptors_list, dtype=np.uint8)
                
                # KNN Matcher
                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                matches = bf.knnMatch(brand_des, input_descriptors, k=2)
                
                # Lowe's Ratio Test (Relaxed to 0.85 for better recall)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.85 * n.distance:
                        good_matches.append(m)
                
                logger.info(f"🔍 Brand scan: {brand['name']} - {len(good_matches)} good matches found")

                if len(good_matches) > max_good_matches:
                    max_good_matches = len(good_matches)
                    best_brand = brand['name']
                    
            except Exception as e:
                logger.error(f"Error matching brand {brand['name']}: {e}")
                continue
        
        # Lower threshold to 4 (works better for logos in screenshots)
        if max_good_matches >= 4:
            logger.info(f"✅ Brand detected: {best_brand} with {max_good_matches} matches")
            return {
                "detected": True,
                "brand": best_brand,
                "confidence": min(max_good_matches / 20.0, 1.0),
                "method": "orb_feature_matching"
            }
            
        return {"detected": False, "brand": None, "method": "orb"}

    def compute_descriptors(self, image_bytes: bytes):
        """Helper to get just descriptors for matching."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None: return None
            _, des = self.orb.detectAndCompute(img, None)
            return des
        except Exception:
            return None
