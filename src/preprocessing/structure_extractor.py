"""
Structure Extractor - Parses raw text into structured fields for Neo4j

This script takes your scraped JSON files and:
1. KEEPS all original text (for RAG/embeddings)
2. EXTRACTS structured data (for Knowledge Graph)

Input:  data/raw/*.json (text-heavy)
Output: data/processed/*.json (text + structured fields)
"""

import json
import re
from pathlib import Path
from typing import Dict, List
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import RAW_DATA_DIR


class StructureExtractor:
    """Extract structured fields from raw text data"""
    
    def __init__(self):
        print("📊 Structure Extractor initialized")
    
    # ================================================================
    # DRUG INTERACTIONS EXTRACTION
    # ================================================================
    
    def extract_drug_interactions(self, interactions_text: str) -> List[Dict]:
        """
        Extract drug interactions from continuous text
        
        Pattern in NatMed:
        "DRUG NAME Interaction Rating ... Severity ... 
         NEXT DRUG NAME Interaction Rating ..."
        """
        interactions = []
        
        if not interactions_text or len(interactions_text) < 50:
            return interactions
        
        print("      🔍 Parsing drug interactions...")
        
        # Remove "Expand All | Collapse All" noise
        text = re.sub(r'Expand All \| Collapse All', '', interactions_text)
        
        # Strategy: Find patterns like "DRUG NAME Interaction Rating"
        # Drug names are in CAPS, followed by "Interaction Rating"
        
        # Split by "Interaction Rating" to get sections
        sections = re.split(r'(?=\b[A-Z/\s]{10,}?\s+Interaction Rating)', text)
        
        for section in sections:
            section = section.strip()
            
            if len(section) < 50:
                continue
            
            # Extract drug name (CAPS at the start, before "Interaction Rating")
            drug_match = re.match(r'^([A-Z/\s\-]{10,}?)(?=\s+Interaction Rating)', section)
            
            if drug_match:
                drug_name = drug_match.group(1).strip()
                
                # Clean up drug name
                drug_name = re.sub(r'\s+', ' ', drug_name)
                
                # Get the rest of the section text
                section_text = section[len(drug_name):].strip()
                
                # Extract this interaction
                self._save_interaction(drug_name, [section_text], interactions)
        
        # Fallback: If no matches, try a simpler approach
        if not interactions:
            # Find all-caps phrases that look like drug names
            drug_pattern = r'\b([A-Z][A-Z/\-\s]{10,}?)\s+Interaction Rating'
            drug_matches = list(re.finditer(drug_pattern, text))
            
            for i, match in enumerate(drug_matches):
                drug_name = match.group(1).strip()
                start_pos = match.end()
                
                # Get text until next drug or end
                if i < len(drug_matches) - 1:
                    end_pos = drug_matches[i + 1].start()
                    section_text = text[start_pos:end_pos]
                else:
                    section_text = text[start_pos:]
                
                # Limit length
                section_text = section_text[:1500]
                
                self._save_interaction(drug_name, [section_text], interactions)
        
        print(f"      ✓ Found {len(interactions)} drug interactions")
        return interactions
    
    def _save_interaction(self, drug_name: str, text_lines: List[str], 
                         interactions: List[Dict]):
        """Helper to save a drug interaction"""
        
        full_text = ' '.join(text_lines)
        
        # Extract severity (HIGH, MODERATE, LOW)
        severity = "Unknown"
        severity_match = re.search(r'Severity\s+(HIGH|MODERATE|LOW|MINOR)', 
                                  full_text, re.IGNORECASE)
        if severity_match:
            severity = severity_match.group(1).capitalize()
        
        # Extract interaction rating
        rating = "See description"
        rating_match = re.search(r'Interaction Rating\s+([^\.]+?)(?:Severity|$)', 
                                full_text, re.IGNORECASE)
        if rating_match:
            rating = rating_match.group(1).strip()
        
        # Clean description (remove rating metadata)
        description = re.sub(r'Interaction Rating.*?Level of Evidence[^\n]*', 
                           '', full_text, flags=re.IGNORECASE | re.DOTALL)
        description = description.strip()[:600]  # Limit length
        
        if description and len(description) > 20:
            interactions.append({
                "drug_name": drug_name,
                "drug_class": drug_name,
                "severity": severity,
                "interaction_type": rating,
                "description": description
            })
    
    # ================================================================
    # CONDITIONS EXTRACTION
    # ================================================================
    
    def extract_conditions(self, effectiveness_text: str) -> List[str]:
        """
        Extract health conditions from effectiveness text
        
        How it works:
        1. Look for "Possibly Effective", "Likely Effective" sections
        2. Extract condition names that follow these ratings
        3. Look for common health terms
        
        Example input:
        "Possibly Effective
         Anxiety. Small clinical studies suggest..."
        
        Example output:
        ["Anxiety", "Stress", "Sleep"]
        """
        conditions = []
        
        if not effectiveness_text:
            return conditions
        
        print("      🔍 Extracting conditions...")
        
        # Strategy 1: Find text after effectiveness ratings
        effectiveness_levels = [
            "Likely Effective",
            "Possibly Effective", 
            "Insufficient Evidence",
            "Possibly Ineffective"
        ]
        
        for level in effectiveness_levels:
            # Find sections starting with this rating
            pattern = rf'{level}\s+([A-Z][a-z]+(?:\s+[a-z]+){{0,4}})\.'
            matches = re.findall(pattern, effectiveness_text)
            
            for match in matches:
                condition = match.strip()
                if 5 < len(condition) < 50 and condition not in conditions:
                    conditions.append(condition)
        
        # Strategy 2: Look for common supplement uses
        common_conditions = [
            "stress", "anxiety", "insomnia", "sleep", "depression",
            "cognitive function", "memory", "inflammation", "pain",
            "diabetes", "blood pressure", "cholesterol", "heart health",
            "immune support", "thyroid", "testosterone", "fertility",
            "digestion", "bone health", "muscle", "energy", "fatigue"
        ]
        
        text_lower = effectiveness_text.lower()
        for condition in common_conditions:
            if condition in text_lower:
                condition_title = condition.title()
                if condition_title not in conditions:
                    conditions.append(condition_title)
        
        print(f"      ✓ Found {len(conditions)} conditions")
        return conditions[:15]  # Limit to top 15
    
    # ================================================================
    # DOSAGE EXTRACTION
    # ================================================================
    
    def extract_dosages(self, dosing_text: str) -> List[Dict]:
        """
        Extract dosage guidelines
        
        How it works:
        1. Find dosage amounts (e.g., "300 mg", "1-2 grams")
        2. Extract frequency (daily, twice daily, etc.)
        3. Get surrounding context
        
        Example input:
        "Adult Oral: Ashwagandha has most often been used in doses 
         of up to 1000 mg daily for up to 12 weeks."
        
        Example output:
        [{
          "dosage": "1000 mg",
          "frequency": "Daily",
          "notes": "Most often used for up to 12 weeks"
        }]
        """
        dosages = []
        
        if not dosing_text:
            return dosages
        
        print("      🔍 Extracting dosages...")
        
        # Find dosage patterns: "300 mg", "1-2 grams", "500-1000 mg"
        dosage_pattern = r'(\d+(?:-\d+)?)\s*(mg|g|mcg|IU|grams?|milligrams?)'
        dosage_matches = re.findall(dosage_pattern, dosing_text, re.IGNORECASE)
        
        if dosage_matches:
            # Take first dosage found (usually the main recommendation)
            first_dosage = f"{dosage_matches[0][0]} {dosage_matches[0][1]}"
            
            # Find the sentence containing this dosage
            sentences = re.split(r'[\.!?]', dosing_text)
            dosage_context = ""
            
            for sentence in sentences:
                if first_dosage in sentence or dosage_matches[0][0] in sentence:
                    dosage_context = sentence.strip()
                    break
            
            # Determine frequency
            frequency = "As directed"
            if "daily" in dosing_text.lower():
                frequency = "Daily"
            elif "twice daily" in dosing_text.lower():
                frequency = "Twice daily"
            elif "three times" in dosing_text.lower():
                frequency = "Three times daily"
            
            dosages.append({
                "condition": "General use",
                "dosage": first_dosage,
                "frequency": frequency,
                "duration": "See notes",
                "form": "Oral",
                "notes": dosage_context[:500]
            })
        
        print(f"      ✓ Found {len(dosages)} dosage guidelines")
        return dosages
    
    # ================================================================
    # SAFETY EXTRACTION
    # ================================================================
    
    def extract_safety_ratings(self, safety_text: str) -> Dict:
        """
        Extract safety ratings and warnings
        
        How it works:
        1. Look for safety rating phrases (Likely Safe, Possibly Safe, etc.)
        2. Find pregnancy/breastfeeding sections
        3. Extract warning sentences
        
        Example input:
        "Possibly Safe when used orally and appropriately, short-term.
         PREGNANCY: Likely Unsafe when used orally."
        
        Example output:
        {
          "general_safety": "Possibly Safe",
          "pregnancy_safety": "Likely Unsafe",
          "warnings": ["Discontinue 2 weeks before surgery"]
        }
        """
        safety = {
            "general_safety": "Unknown",
            "pregnancy_safety": "Unknown",
            "breastfeeding_safety": "Unknown",
            "children_safety": "Unknown",
            "warnings": []
        }
        
        if not safety_text:
            return safety
        
        print("      🔍 Extracting safety ratings...")
        
        text_lower = safety_text.lower()
        
        # General safety rating
        if "likely safe" in text_lower:
            safety["general_safety"] = "Likely Safe"
        elif "possibly safe" in text_lower:
            safety["general_safety"] = "Possibly Safe"
        elif "possibly unsafe" in text_lower:
            safety["general_safety"] = "Possibly Unsafe"
        elif "likely unsafe" in text_lower:
            safety["general_safety"] = "Likely Unsafe"
        
        # Pregnancy safety
        pregnancy_section = re.search(r'PREGNANCY:([^\.]+)', safety_text, re.IGNORECASE)
        if pregnancy_section:
            preg_text = pregnancy_section.group(1).lower()
            if "likely unsafe" in preg_text:
                safety["pregnancy_safety"] = "Likely Unsafe"
            elif "possibly unsafe" in preg_text:
                safety["pregnancy_safety"] = "Possibly Unsafe"
            elif "likely safe" in preg_text:
                safety["pregnancy_safety"] = "Likely Safe"
            elif "possibly safe" in preg_text:
                safety["pregnancy_safety"] = "Possibly Safe"
            else:
                safety["pregnancy_safety"] = "Insufficient Evidence"
        
        # Breastfeeding safety
        lactation_section = re.search(r'LACTATION:([^\.]+)', safety_text, re.IGNORECASE)
        if lactation_section:
            lact_text = lactation_section.group(1).lower()
            if "unsafe" in lact_text:
                safety["breastfeeding_safety"] = "Possibly Unsafe"
            elif "safe" in lact_text:
                safety["breastfeeding_safety"] = "Possibly Safe"
            else:
                safety["breastfeeding_safety"] = "Insufficient Evidence"
        
        # Extract warnings (sentences with warning keywords)
        warning_keywords = ["avoid", "do not", "discontinue", "caution", "contraindicated"]
        sentences = re.split(r'[\.!?]', safety_text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in warning_keywords:
                if keyword in sentence_lower and len(sentence.strip()) > 20:
                    warning = sentence.strip()
                    if warning not in safety["warnings"]:
                        safety["warnings"].append(warning)
                    break
        
        # Limit warnings
        safety["warnings"] = safety["warnings"][:10]
        
        print(f"      ✓ Safety: {safety['general_safety']}")
        print(f"      ✓ Warnings: {len(safety['warnings'])}")
        
        return safety
    
    # ================================================================
    # MAIN PROCESSING
    # ================================================================
    
    def process_supplement(self, raw_json_path: Path) -> Dict:
        """
        Process a single supplement JSON file
        
        Input:  data/raw/ashwagandha.json (text-heavy)
        Output: Structured data + original text
        """
        print(f"\n   📄 Processing: {raw_json_path.name}")
        
        # Load raw data
        with open(raw_json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Extract structured fields
        structured_data = {
            # Basic info
            "supplement_name": raw_data.get("supplement_name", ""),
            "scientific_name": raw_data.get("scientific_name", ""),
            
            # KEEP original text for embeddings/RAG
            "overview_text": raw_data.get("overview", ""),
            "warnings_text": raw_data.get("warnings", ""),
            "effectiveness_text": raw_data.get("effectiveness_text", ""),
            "safety_text": raw_data.get("safety_text", ""),
            "dosing_text": raw_data.get("dosing_text", ""),
            "interactions_text": raw_data.get("interactions_text", ""),
            "mechanism_text": raw_data.get("mechanism_text", ""),
            
            # ADD structured fields for Neo4j
            "drug_interactions": self.extract_drug_interactions(
                raw_data.get("interactions_text", "")
            ),
            "conditions": self.extract_conditions(
                raw_data.get("effectiveness_text", "")
            ),
            "dosage_guidelines": self.extract_dosages(
                raw_data.get("dosing_text", "")
            ),
            "safety_ratings": self.extract_safety_ratings(
                raw_data.get("safety_text", "")
            ),
            
            # Metadata
            "metadata": raw_data.get("metadata", {})
        }
        
        # Summary
        print(f"      ✅ Conditions: {len(structured_data['conditions'])}")
        print(f"      ✅ Drug interactions: {len(structured_data['drug_interactions'])}")
        print(f"      ✅ Dosages: {len(structured_data['dosage_guidelines'])}")
        
        return structured_data


def main():
    """Process all supplements"""
    
    print("="*60)
    print("🔧 STRUCTURE EXTRACTOR")
    print("="*60)
    print("\nThis script will:")
    print("  1. Read your raw scraped data")
    print("  2. Extract structured fields for Neo4j")
    print("  3. Keep original text for embeddings")
    print("  4. Save to data/processed/")
    print()
    
    # Setup directories
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files
    raw_files = list(raw_dir.glob("*.json"))
    
    if not raw_files:
        print("❌ No JSON files found in data/raw/")
        print("   Did you run the scraper first?")
        return
    
    print(f"📦 Found {len(raw_files)} supplement files\n")
    print("="*60)
    
    # Create extractor
    extractor = StructureExtractor()
    
    # Process each supplement
    successful = 0
    failed = []
    
    for raw_file in raw_files:
        try:
            # Extract structure
            structured_data = extractor.process_supplement(raw_file)
            
            # Save to processed directory
            output_file = processed_dir / raw_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            
            print(f"      💾 Saved: {output_file.name}")
            successful += 1
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            failed.append(raw_file.name)
    
    # Summary
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE!")
    print("="*60)
    print(f"\n   Success: {successful}/{len(raw_files)}")
    
    if failed:
        print(f"   Failed: {', '.join(failed)}")
    
    print(f"\n📁 Structured files saved to: {processed_dir.absolute()}")
    print("\n🎯 Next Steps:")
    print("   1. Check processed/*.json to see structured data")
    print("   2. Use these files for Neo4j Knowledge Graph (Phase 2)")
    print("   3. Use text fields for FAISS embeddings (Phase 3)")
    print("="*60)


if __name__ == "__main__":
    main()