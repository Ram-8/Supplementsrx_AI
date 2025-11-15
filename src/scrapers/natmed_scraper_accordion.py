
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import time
import sys
import re

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import RAW_DATA_DIR, LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "natmed_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NatMedAccordionScraper:
    SUPPLEMENT_URLS = {
        "Chromium": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Chromium",
        "Chromium Picolinate": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Chromium%20Picolinate",
        "Berberine": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Berberine",
        "Alpha-Lipoic Acid": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Alpha-Lipoic%20Acid",
        "Cinnamon": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Cinnamon",
        "Gymnema Sylvestre": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Gymnema",   # monograph title is typically "Gymnema"
        "Fenugreek": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Fenugreek",
        "Bitter Melon": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Bitter%20Melon",
        "Ginseng": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Ginseng",
        "Resveratrol": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Resveratrol",
        "N-Acetyl Cysteine": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/N-Acetyl%20Cysteine",

        "Magnesium": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Magnesium",
        "Vitamin D": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Vitamin%20D",
        "Zinc": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Zinc",
        "Coenzyme Q-10": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Coenzyme%20Q-10",
        "Fish Oil": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Fish%20Oil",
        "Turmeric (Curcumin)": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Turmeric",
        "Green Tea": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Green%20Tea",
        "Aloe Vera": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Aloe%20Vera",
        "L-Carnitine": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/L-Carnitine",
        "Vitamin B12": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Vitamin%20B12",
        "Vitamin E": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Vitamin%20E",
        "Biotin": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Biotin",
        "Niacin": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Niacin",
        "Inositol": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Inositol",
        "Milk Thistle (Silymarin)": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Milk%20Thistle",
        "Banaba (Lagerstroemia speciosa)": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Banaba",
        "Vanadium": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Vanadium",
        "Pycnogenol": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Pycnogenol",
        "Taurine": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Taurine",
        "L-Arginine": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/L-Arginine",

        "Ginger": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Ginger",
        "Mulberry (Morus alba)": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Mulberry",
        "Psyllium": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Psyllium",
        "Glucomannan": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Glucomannan",
        "Amla (Indian Gooseberry)": "https://naturalmedicines.therapeuticresearch.com/Data/ProMonographs/Indian%20Gooseberry",
}

    
    def __init__(self, username: str, password: str, headless: bool = True):
        self.username = username
        self.password = password
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.page = None
        self.is_logged_in = False
        
    def start_browser(self):
        logger.info("Starting browser...")
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page()
        self.page.set_default_timeout(30000)
    
    def login(self) -> bool:
        if self.is_logged_in:
            return True
        
        try:
            logger.info("Logging in...")
            self.page.goto("https://naturalmedicines.therapeuticresearch.com")
            time.sleep(2)
            
            self.page.click("text=Login")
            time.sleep(2)
            
            self.page.fill("input[name='username']", self.username)
            time.sleep(1)
            
            for selector in ["button[type='submit']", "input[type='submit']"]:
                try:
                    if self.page.locator(selector).first.is_visible(timeout=2000):
                        self.page.locator(selector).first.click()
                        break
                except:
                    continue
            
            time.sleep(3)
            self.page.fill("input[type='password']", self.password)
            time.sleep(1)
            
            for selector in ["button[type='submit']", "input[type='submit']"]:
                try:
                    if self.page.locator(selector).first.is_visible(timeout=2000):
                        self.page.locator(selector).first.click()
                        break
                except:
                    continue
            
            time.sleep(5)
            
            if "login" not in self.page.url.lower():
                self.is_logged_in = True
                logger.info("✓ Logged in")
                return True
            
            return False
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean whitespace"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_accordion_content(self, soup: BeautifulSoup, section_name: str, button_id: str) -> str:
        """
        Extract content from Bootstrap accordion panel
        
        Structure:
        <div class="accordion-panel-heading">
            <h2>
                <button id="dosing">Section Name</button>
            </h2>
        </div>
        <div class="accordion-panel-collapse">  ← CONTENT HERE
            ...
        </div>
        """
        try:
            # Find button with this ID
            button = soup.find('button', id=button_id)
            
            if not button:
                # Try finding by text
                all_buttons = soup.find_all('button', class_='accordion-toggle')
                for btn in all_buttons:
                    btn_text = btn.get_text(strip=True).lower()
                    if section_name.lower() in btn_text:
                        button = btn
                        break
            
            if not button:
                logger.warning(f"    Could not find button for '{section_name}'")
                return ""
            
            logger.info(f"    Found button: '{button.get_text(strip=True)}'")
            
            # Navigate up to the accordion-panel-heading div
            heading_div = button.find_parent('div', class_='accordion-panel-heading')
            
            if not heading_div:
                logger.warning(f"    Could not find accordion-panel-heading")
                return ""
            
            # The content is in the next sibling: accordion-panel-collapse
            content_div = heading_div.find_next_sibling('div', class_='accordion-panel-collapse')
            
            if not content_div:
                logger.warning(f"    Could not find accordion-panel-collapse")
                return ""
            
            # Extract text from this div
            text = content_div.get_text(separator=' ', strip=True)
            cleaned = self._clean_text(text)
            
            logger.info(f"    Extracted: {len(cleaned)} chars")
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"    Error extracting '{section_name}': {e}")
            return ""
    
    def scrape_supplement(self, supplement_name: str) -> Optional[Dict]:
        """Scrape supplement from accordion panels"""
        
        url = self.SUPPLEMENT_URLS.get(supplement_name)
        if not url:
            logger.warning(f"No URL for: {supplement_name}")
            return None
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping: {supplement_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Navigate
            self.page.goto(url, timeout=30000)
            time.sleep(8)  # Wait for JavaScript to load accordions
            self.page.wait_for_load_state("networkidle")
            
            logger.info(f"  URL: {self.page.url}")
            
            # Get HTML
            html = self.page.content()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract sections from accordions
            sections = {}
            
            section_definitions = [
                ("Overview", "overview"),
                ("Warnings", "warnings"),
                ("Safety", "safety"),
                ("Effectiveness", "effectiveness"),
                ("Dosing & Administration", "dosing"),
                ("Interactions with Drugs", "interactions"),
                ("Mechanism of Action", "mechanism"),
            ]
            
            for section_name, button_id in section_definitions:
                logger.info(f"\n  📄 {section_name}")
                content = self._extract_accordion_content(soup, section_name, button_id)
                sections[button_id] = content
            
            # Build structured data
            supplement_data = {
                "supplement_name": supplement_name,
                "scientific_name": "",  # Could extract from Overview
                "overview": sections.get("overview", ""),
                "warnings": sections.get("warnings", ""),
                "effectiveness_text": sections.get("effectiveness", ""),
                "safety_text": sections.get("safety", ""),
                "dosing_text": sections.get("dosing", ""),
                "interactions_text": sections.get("interactions", ""),
                "mechanism_text": sections.get("mechanism", ""),
                "metadata": {
                    "scrape_timestamp": datetime.now().isoformat(),
                    "source_url": url,
                    "data_version": "3.0",
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            # Quality check
            total_chars = sum(len(sections.get(k, "")) for k in sections)
            logger.info(f"\n  ✓ Total content: {total_chars} chars")
            
            if total_chars < 1000:
                logger.warning("  ⚠️ Low content extracted")
            else:
                logger.info("  ✅ Good content extraction!")
            
            return supplement_data
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return None
    
    def save_to_json(self, data: Dict):
        """Save to JSON"""
        filename = f"{data['supplement_name'].lower().replace(' ', '_')}.json"
        filepath = RAW_DATA_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved: {filename}")
    
    def scrape_multiple_supplements(self, supplement_list: List[str]):
        """Batch scrape"""
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH SCRAPING: {len(supplement_list)} supplements")
        logger.info(f"{'='*60}\n")
        
        self.start_browser()
        
        if not self.login():
            logger.error("❌ Login failed")
            return
        
        successful = 0
        failed = []
        
        for i, supplement in enumerate(supplement_list, 1):
            logger.info(f"\n[{i}/{len(supplement_list)}] {supplement}")
            
            try:
                data = self.scrape_supplement(supplement)
                if data:
                    self.save_to_json(data)
                    successful += 1
                else:
                    failed.append(supplement)
                
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Failed {supplement}: {e}")
                failed.append(supplement)
        
        logger.info("\n" + "="*60)
        logger.info(f"✅ COMPLETE!")
        logger.info(f"   Success: {successful}/{len(supplement_list)}")
        if failed:
            logger.info(f"   Failed: {', '.join(failed)}")
        logger.info("="*60)
    
    def close(self):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    username = os.getenv("NATMED_USERNAME")
    password = os.getenv("NATMED_PASSWORD")
    
    if not username or not password:
        print("❌ Set credentials in .env")
        exit(1)
    
    supplements = [
        "Chromium",
        "Chromium Picolinate",
        "Berberine",
        "Alpha-Lipoic Acid",
        "Cinnamon",
        "Gymnema Sylvestre",
        "Fenugreek",
        "Bitter Melon",
        "Ginseng",
        "Resveratrol",
        "N-Acetyl Cysteine",
        "Magnesium",
        "Vitamin D",
        "Zinc",
        "Coenzyme Q-10",
        "Fish Oil",
        "Turmeric (Curcumin)",
        "Green Tea",
        "Aloe Vera",
        "L-Carnitine",
        "Vitamin B12",
        "Vitamin E",
        "Biotin",
        "Niacin",
        "Inositol",
        "Milk Thistle (Silymarin)",
        "Banaba (Lagerstroemia speciosa)",
        "Vanadium",
        "Pycnogenol",
        "Taurine",
        "L-Arginine",
        "Ginger",
        "Mulberry (Morus alba)",
        "Psyllium",
        "Glucomannan",
        "Amla (Indian Gooseberry)",
    ]
    
    print("🚀 NatMed Accordion Scraper")
    print("="*60)
    
    scraper = NatMedAccordionScraper(username, password, headless=False)
    
    try:
        scraper.scrape_multiple_supplements(supplements)
    finally:
        scraper.close()
    
    print("\n✅ Check your data:")
    print("  cat data/raw/ashwagandha.json | head -100")