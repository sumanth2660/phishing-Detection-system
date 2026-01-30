
import asyncio
import aiohttp
import random
import string
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class PoisonPill:
    """
    Active Defense System: Floods phishing sites with fake credentials.
    """
    
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1"
        ]
        
        self.first_names = ["John", "Jane", "Michael", "Emily", "David", "Sarah", "James", "Emma", "Robert", "Olivia", "William", "Ava"]
        self.last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        self.domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]

    def _generate_fake_data(self, field_type):
        """Generate realistic fake data based on input field name/type."""
        field_type = field_type.lower()
        
        fn = random.choice(self.first_names)
        ln = random.choice(self.last_names)
        
        if "email" in field_type or "user" in field_type:
            sep = random.choice([".", "_", "", "-"])
            num = random.randint(1, 9999) if random.random() > 0.5 else ""
            return f"{fn.lower()}{sep}{ln.lower()}{num}@{random.choice(self.domains)}"
            
        if "pass" in field_type:
            # Generate a "realistic" weak/medium password
            base = random.choice(["Pass", "Secret", "Love", "Dragon", "Football", "Master", "Admin"])
            num = random.randint(1, 999)
            sym = random.choice(["!", "@", "#", "$"])
            return f"{base}{num}{sym}"
            
        if "phone" in field_type:
            return f"+1{random.randint(200, 999)}{random.randint(100, 999)}{random.randint(1000, 9999)}"
            
        if "card" in field_type or "cvv" in field_type:
            return str(random.randint(1000000000000000, 9999999999999999))
            
        return "12345" # Fallback

    async def deploy(self, target_url, iterations=50):
        """
        Executes the Poison Pill attack.
        1. Scans URL for forms.
        2. Generates fake data.
        3. Floods the form endpoint.
        """
        logger.info(f"💊 ACTIVATING POISON PILL against: {target_url}")
        
        results = {"sent": 0, "failed": 0, "target_endpoint": None}
        
        async with aiohttp.ClientSession() as session:
            # 1. Reconnaissance: Get the form details
            try:
                async with session.get(target_url, headers={"User-Agent": random.choice(self.user_agents)}, ssl=False) as resp:
                    html = await resp.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    forms = soup.find_all('form')
                    if not forms:
                        logger.warning("No forms found on target page!")
                        return {"status": "failed", "reason": "No forms found"}
                    
                    # Target the first visible form (usually the login form)
                    target_form = forms[0]
                    action = target_form.get('action')
                    if not action:
                        action = target_url # Post to self if no action
                    elif not action.startswith('http'):
                        action = urljoin(target_url, action)
                        
                    results["target_endpoint"] = action
                    
                    # Extract inputs
                    inputs = target_form.find_all('input')
                    input_names = [i.get('name') for i in inputs if i.get('name') and i.get('type') != 'hidden']
                    
                    if not input_names:
                        logger.warning("No named inputs found!")
                        return {"status": "failed", "reason": "No named inputs"}
                        
                    logger.info(f"Target Acquired: POST {action} | Fields: {input_names}")

            except Exception as e:
                logger.error(f"Reconnaissance failed: {e}")
                return {"status": "failed", "reason": str(e)}

            # 2. Attack Phase (Stealth Mode)
            # We add delays and better headers to bypass basic WAFs (like non-Enterprise Cloudflare)
            logger.info(f"🚀 Launching {iterations} stealth requests...")
            
            # Common referers to look legitimate
            referers = ["https://www.google.com/", "https://www.bing.com/", "https://duckduckgo.com/"]
            
            async def send_bogus_request():
                payload = {}
                for name in input_names:
                    # CRITICAL: ALWAYS user random fake data. NEVER use real user data.
                    payload[name] = self._generate_fake_data(name)
                
                # Realistic Headers to fool WAFs
                headers = {
                    "User-Agent": random.choice(self.user_agents),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Origin": "{uri.scheme}://{uri.netloc}".format(uri=aiohttp.helpers.URL(target_url)),
                    "Referer": random.choice(referers)
                }
                
                try:
                    # Random delay between 0.5s and 2.0s to mimic human behavior
                    await asyncio.sleep(random.uniform(0.5, 2.0))
                    
                    async with session.post(
                        action, 
                        data=payload, 
                        headers=headers,
                        timeout=10
                    ) as attack_resp:
                        return attack_resp.status == 200 or attack_resp.status == 302
                except:
                    return False

            # Run in smaller parallel batches to reduce "Bot" signature
            batch_size = 5 
            for i in range(0, iterations, batch_size):
                tasks = [send_bogus_request() for _ in range(batch_size)]
                batch_results = await asyncio.gather(*tasks)
                
                results["sent"] += batch_results.count(True)
                results["failed"] += batch_results.count(False)
                
                # Cooldown between batches
                await asyncio.sleep(1.0)
                
        if results["sent"] == 0:
            results["reason"] = f"WAF/Firewall Blocked Requests (Failed {results['failed']}/{iterations})"
            
        logger.info(f"✅ Poison Pill Complete. Injected {results['sent']} fake records.")
        return results

if __name__ == "__main__":
    # Test Run
    pill = PoisonPill()
    # Disclaimer: Only run on authorized targets or localhost!
    # asyncio.run(pill.deploy("http://localhost:8000/fake-login"))
