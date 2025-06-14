#!/usr/bin/env python3
"""
UMLS Term Typing System - Clean Implementation

A streamlined implementation for medical term classification using UMLS semantic types.
This module consolidates all UMLS functionality into a single, clean interface.
"""

import os
import json
import time
import random
import logging
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class UMLSTermTyper:
    """Main class for UMLS-based medical term classification."""
    
    def __init__(self):
        """Initialize the UMLS term typer."""
        self._setup_api()
        self._load_data()
    
    def _setup_api(self):
        """Setup Google Gemini API."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY in .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        logger.info("✓ API configured successfully")
    
    def _load_data(self):
        """Load UMLS data from files."""
        data_dir = "data"
        
        try:
            # Load UMLS classes/semantic types
            with open(os.path.join(data_dir, "umls_classes.json"), 'r', encoding='utf-8') as f:
                self.umls_classes = json.load(f)
            
            # Load UMLS terms
            with open(os.path.join(data_dir, "umls_terms.txt"), 'r', encoding='utf-8') as f:
                self.umls_terms = [line.strip() for line in f if line.strip() and len(line.strip()) > 2]
            
            # Load UMLS hierarchy
            with open(os.path.join(data_dir, "umls_hierarchy.json"), 'r', encoding='utf-8') as f:
                self.umls_hierarchy = json.load(f)
            
            self.semantic_types = list(self.umls_classes.keys())
            
            logger.info(f"✓ Loaded {len(self.semantic_types)} semantic types")
            logger.info(f"✓ Loaded {len(self.umls_terms)} terms")
            logger.info(f"✓ Loaded hierarchy with {len(self.umls_hierarchy)} entries")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"UMLS data files not found: {e}. Run data preparation scripts first.")
    
    def classify_term(self, term: str) -> str:
        """
        Classify a medical term into UMLS semantic type.
        
        Args:
            term: Medical term to classify
            
        Returns:
            UMLS semantic type or "unknown" if classification fails
        """
        try:
            # Sample semantic types to avoid overly long prompts
            sample_types = random.sample(self.semantic_types, min(20, len(self.semantic_types)))
            
            prompt = f"""Given the medical term "{term}" and the following UMLS semantic types, classify the term into the most appropriate semantic type.

UMLS Semantic Types:
{', '.join(sample_types)}

Consider the medical and biological context of the term. Choose the semantic type that best represents the nature and category of this term in the medical domain.

Term: {term}

Please respond with only the most suitable semantic type for this term, without any additional text or explanation."""

            response = self.model.generate_content(prompt)
            result = response.text.strip()
            
            # Map response to available semantic types
            mapped_result = self._map_response_to_semantic_type(result, term)
            
            logger.info(f"'{term}' classified as '{mapped_result}'")
            return mapped_result
            
        except Exception as e:
            logger.error(f"Error classifying term '{term}': {e}")
            return "unknown"
    
    def _map_response_to_semantic_type(self, response: str, term: str) -> str:
        """Map LLM response to valid UMLS semantic type."""
        response_lower = response.lower().strip()
        
        # Try exact matching first
        for semantic_type in self.semantic_types:
            if semantic_type.lower() in response_lower:
                return semantic_type
        
        # Try special mappings for common responses
        mappings = {
            'organ': 'Body Part, Organ, or Organ Component',
            'body part': 'Body Part, Organ, or Organ Component', 
            'structure': 'Anatomical Structure',
            'anatomical': 'Anatomical Structure',
            'drug': 'Pharmacologic Substance',
            'medication': 'Pharmacologic Substance',
            'medicine': 'Pharmacologic Substance',
            'chemical': 'Chemical',
            'substance': 'Chemical',
            'procedure': 'Therapeutic or Preventive Procedure',
            'treatment': 'Therapeutic or Preventive Procedure',
            'therapy': 'Therapeutic or Preventive Procedure',
            'disease': 'Disease or Syndrome',
            'disorder': 'Disease or Syndrome',
            'condition': 'Disease or Syndrome',
            'illness': 'Disease or Syndrome',
            'syndrome': 'Disease or Syndrome',
            'symptom': 'Sign or Symptom',
            'sign': 'Sign or Symptom',
            'finding': 'Finding',
            'organism': 'Organism',
            'bacteria': 'Bacterium',
            'virus': 'Virus'
        }
        
        for key, mapped_type in mappings.items():
            if key in response_lower and mapped_type in self.semantic_types:
                logger.info(f"'{term}' mapped from '{key}' to '{mapped_type}'")
                return mapped_type
        
        # Try fuzzy matching on key words
        for semantic_type in self.semantic_types:
            type_words = semantic_type.lower().split()
            for word in type_words:
                if len(word) > 3 and word in response_lower:
                    logger.info(f"'{term}' fuzzy matched to '{semantic_type}' via '{word}'")
                    return semantic_type
        
        logger.warning(f"Could not map response '{response}' for term '{term}'")
        return "unknown"
    
    def classify_terms_batch(self, terms: List[str], delay: float = 2.0) -> List[Tuple[str, str]]:
        """
        Classify multiple terms with rate limiting.
        
        Args:
            terms: List of terms to classify
            delay: Delay between API calls in seconds
            
        Returns:
            List of (term, semantic_type) tuples
        """
        results = []
        for term in terms:
            result = self.classify_term(term)
            results.append((term, result))
            if delay > 0:
                time.sleep(delay)
        return results
    
    def check_hierarchical_relationship(self, child: str, parent: str) -> bool:
        """
        Check if there's a hierarchical relationship between two semantic types.
        
        Args:
            child: Child semantic type
            parent: Parent semantic type
            
        Returns:
            True if hierarchical relationship exists
        """
        try:
            prompt = f"""In the UMLS (Unified Medical Language System) semantic network, determine if there is a hierarchical relationship where "{child}" is a subtype or specialization of "{parent}".

Consider the medical and biological meanings of these terms. A hierarchical relationship exists if the first term is a more specific instance or subtype of the second term.

Question: Is "{child}" a subtype/specialization of "{parent}" in medical terminology?

Answer with only "yes" or "no"."""

            response = self.model.generate_content(prompt)
            result = response.text.strip().lower()
            
            return "yes" in result
            
        except Exception as e:
            logger.error(f"Error checking relationship {child} -> {parent}: {e}")
            return False
    
    def get_semantic_type_info(self, semantic_type: str) -> Optional[Dict]:
        """Get information about a specific semantic type."""
        return self.umls_classes.get(semantic_type)
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded UMLS data."""
        return {
            "semantic_types_count": len(self.semantic_types),
            "terms_count": len(self.umls_terms),
            "hierarchy_entries": len(self.umls_hierarchy),
            "semantic_types": self.semantic_types[:10]  # First 10 for preview
        }


def main():
    """Demo function showing basic usage."""
    print("🧬 UMLS Term Typing System Demo")
    print("=" * 40)
    
    try:
        # Initialize the typer
        typer = UMLSTermTyper()
        
        # Show stats
        stats = typer.get_stats()
        print(f"\nSystem loaded successfully!")
        print(f"• Semantic types: {stats['semantic_types_count']}")
        print(f"• Terms: {stats['terms_count']}")
        print(f"• Hierarchy entries: {stats['hierarchy_entries']}")
        
        # Test some terms
        test_terms = ["diabetes", "heart", "aspirin", "surgery", "bacteria"]
        
        print(f"\nTesting {len(test_terms)} medical terms:")
        print("-" * 30)
        
        for term in test_terms:
            result = typer.classify_term(term)
            print(f"'{term}' → {result}")
            time.sleep(1)  # Rate limiting
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
