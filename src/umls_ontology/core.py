"""
Core UMLS ontology learning module.
Consolidates all functionality into clean, focused classes.
"""

import json
import time
import random
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import google.generativeai as genai
from config.settings import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UMLSDataLoader:
    """Handles loading and validation of UMLS data."""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self._data = {}
    
    def load_all_data(self) -> Dict[str, Any]:
        """Load all UMLS data files."""
        if self._data:
            return self._data
            
        self._data = {
            'classes': self._load_json(self.config.UMLS_CLASSES_FILE),
            'hierarchy': self._load_json(self.config.UMLS_HIERARCHY_FILE),
            'relationships': self._load_json(self.config.UMLS_RELATIONSHIPS_FILE),
            'terms': self._load_terms(self.config.UMLS_TERMS_FILE)
        }
        
        logger.info(f"Loaded {len(self._data['classes'])} semantic types")
        logger.info(f"Loaded {len(self._data['terms'])} terms")
        
        return self._data
    
    def _load_json(self, file_path: Path) -> Dict:
        """Load JSON file with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Data file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            raise
    
    def _load_terms(self, file_path: Path) -> List[str]:
        """Load terms from text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"Terms file not found: {file_path}")
            raise


class LLMClient:
    """Handles all LLM interactions with rate limiting and error handling."""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self._setup_client()
    
    def _setup_client(self):
        """Initialize the LLM client."""
        if not self.config.GOOGLE_API_KEY:
            raise ValueError("Google API key not configured")
        
        genai.configure(api_key=self.config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(self.config.MODEL_NAME)
        logger.info(f"Initialized {self.config.MODEL_NAME}")
    
    def generate_response(self, prompt: str, max_retries: int = 3) -> str:
        """Generate response with retry logic and rate limiting."""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                time.sleep(self.config.API_DELAY)
                return response.text.strip()
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.config.API_DELAY * 2)  # Exponential backoff
                else:
                    raise
        
        raise RuntimeError(f"Failed to generate response after {max_retries} attempts")


class UMLSClassifier:
    """Handles medical term classification tasks."""
    
    def __init__(self, data_loader: UMLSDataLoader, llm_client: LLMClient):
        self.data_loader = data_loader
        self.llm_client = llm_client
        self.data = data_loader.load_all_data()
        self.semantic_types = list(self.data['classes'].keys())
    
    def classify_term(self, term: str, sample_size: int = 15) -> str:
        """Classify a medical term into UMLS semantic type."""
        # Sample semantic types to avoid overly long prompts
        sampled_types = random.sample(
            self.semantic_types, 
            min(sample_size, len(self.semantic_types))
        )
        
        template = random.choice(Config.CLASSIFICATION_TEMPLATES)
        prompt = template.format(
            term=term,
            types=", ".join(sampled_types)
        )
        
        try:
            response = self.llm_client.generate_response(prompt)
            mapped_type = self._map_response_to_type(response, sampled_types)
            logger.info(f"Classified '{term}' as '{mapped_type}'")
            return mapped_type
        except Exception as e:
            logger.error(f"Failed to classify '{term}': {e}")
            return "unknown"
    
    def _map_response_to_type(self, response: str, available_types: List[str]) -> str:
        """Map LLM response to valid semantic type."""
        response_lower = response.lower()
        
        # Exact match
        for semantic_type in available_types:
            if semantic_type.lower() in response_lower:
                return semantic_type
        
        # Fuzzy matching
        type_mappings = {
            "organ": "Body Part, Organ, or Organ Component",
            "body": "Body Part, Organ, or Organ Component",
            "drug": "Pharmacologic Substance",
            "medication": "Pharmacologic Substance",
            "disease": "Disease or Syndrome",
            "disorder": "Disease or Syndrome",
            "procedure": "Therapeutic or Preventive Procedure",
            "treatment": "Therapeutic or Preventive Procedure",
            "symptom": "Sign or Symptom",
            "sign": "Sign or Symptom"
        }
        
        for keyword, mapped_type in type_mappings.items():
            if keyword in response_lower and mapped_type in available_types:
                return mapped_type
        
        logger.warning(f"Could not map response: {response}")
        return "unknown"
    
    def classify_batch(self, terms: List[str]) -> List[Tuple[str, str]]:
        """Classify multiple terms with progress tracking."""
        results = []
        for i, term in enumerate(terms, 1):
            logger.info(f"Processing term {i}/{len(terms)}: {term}")
            result = self.classify_term(term)
            results.append((term, result))
        return results


class UMLSRelationshipAnalyzer:
    """Handles hierarchical and semantic relationship analysis."""
    
    def __init__(self, data_loader: UMLSDataLoader, llm_client: LLMClient):
        self.data_loader = data_loader
        self.llm_client = llm_client
        self.data = data_loader.load_all_data()
    
    def check_hierarchy(self, child: str, parent: str) -> bool:
        """Check if hierarchical relationship exists between terms."""
        template = random.choice(Config.HIERARCHY_TEMPLATES)
        prompt = template.format(child=child, parent=parent, answer="[ANSWER]")
        prompt = prompt.replace("[ANSWER]", "")
        prompt += "\n\nRespond with only 'yes' or 'no'."
        
        try:
            response = self.llm_client.generate_response(prompt)
            return self._parse_boolean_response(response)
        except Exception as e:
            logger.error(f"Failed to check hierarchy {child} -> {parent}: {e}")
            return False
    
    def check_semantic_relation(self, subject: str, relation: str, object_term: str) -> bool:
        """Check if semantic relationship holds between terms."""
        template = random.choice(Config.RELATION_TEMPLATES)
        prompt = template.format(
            subject=subject, 
            relation=relation, 
            object=object_term, 
            answer=""
        )
        prompt += "\n\nRespond with only 'yes' or 'no'."
        
        try:
            response = self.llm_client.generate_response(prompt)
            return self._parse_boolean_response(response)
        except Exception as e:
            logger.error(f"Failed to check relation {subject} {relation} {object_term}: {e}")
            return False
    
    def _parse_boolean_response(self, response: str) -> bool:
        """Parse LLM response to boolean value."""
        response_lower = response.lower().strip()
        
        positive_indicators = Config.LABEL_MAPPINGS["correct"]
        negative_indicators = Config.LABEL_MAPPINGS["incorrect"]
        
        if any(indicator in response_lower for indicator in positive_indicators):
            return True
        elif any(indicator in response_lower for indicator in negative_indicators):
            return False
        else:
            logger.warning(f"Ambiguous response: {response}")
            return False


class UMLSOntologyLearner:
    """Main orchestrator class for UMLS ontology learning tasks."""
    
    def __init__(self):
        Config.ensure_directories()
        Config.validate_config()
        
        self.data_loader = UMLSDataLoader()
        self.llm_client = LLMClient()
        self.classifier = UMLSClassifier(self.data_loader, self.llm_client)
        self.relationship_analyzer = UMLSRelationshipAnalyzer(self.data_loader, self.llm_client)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded system."""
        data = self.data_loader.load_all_data()
        return {
            "semantic_types_count": len(data['classes']),
            "terms_count": len(data['terms']),
            "hierarchy_entries": len(data['hierarchy']),
            "sample_types": list(data['classes'].keys())[:10],
            "sample_terms": data['terms'][:10]
        }
    
    def run_classification_demo(self, num_terms: int = 5) -> Dict[str, Any]:
        """Run a demo of term classification."""
        data = self.data_loader.load_all_data()
        sample_terms = random.sample(data['terms'], min(num_terms, len(data['terms'])))
        
        results = self.classifier.classify_batch(sample_terms)
        
        return {
            "task": "term_classification",
            "results": results,
            "summary": {
                "total_terms": len(results),
                "successful_classifications": len([r for r in results if r[1] != "unknown"]),
                "unknown_classifications": len([r for r in results if r[1] == "unknown"])
            }
        }
    
    def run_hierarchy_demo(self, test_pairs: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Run a demo of hierarchical relationship detection."""
        if not test_pairs:
            test_pairs = [
                ("Human", "Animal"),
                ("Disease or Syndrome", "Pathologic Function"),
                ("Body Part, Organ, or Organ Component", "Anatomical Structure"),
                ("Pharmacologic Substance", "Chemical")
            ]
        
        results = []
        for child, parent in test_pairs:
            relationship_exists = self.relationship_analyzer.check_hierarchy(child, parent)
            results.append((child, parent, relationship_exists))
        
        return {
            "task": "hierarchy_detection",
            "results": results,
            "summary": {
                "total_pairs": len(results),
                "positive_relationships": len([r for r in results if r[2]]),
                "negative_relationships": len([r for r in results if not r[2]])
            }
        }
