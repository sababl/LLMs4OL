"""
Configuration management for UMLS Ontology Learning project.
"""

import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralized configuration for the UMLS ontology learning system."""
    
    # Project structure
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
    
    # Data files
    UMLS_CLASSES_FILE = DATA_DIR / "umls_classes.json"
    UMLS_TERMS_FILE = DATA_DIR / "umls_terms.txt"
    UMLS_HIERARCHY_FILE = DATA_DIR / "umls_hierarchy.json"
    UMLS_RELATIONSHIPS_FILE = DATA_DIR / "umls_relationship_samples.json"
    
    # Rate limiting
    API_DELAY = 5.0  # seconds between API calls
    
    # Evaluation settings
    TEST_SAMPLE_SIZE = 50
    RANDOM_SEED = 42
    
    # Templates for different tasks
    CLASSIFICATION_TEMPLATES = [
        "Classify the medical term '{term}' into the most appropriate UMLS semantic type from: {types}",
        "Given the medical concept '{term}', select the best semantic type: {types}",
    ]
    
    HIERARCHY_TEMPLATES = [
        "{child} is a subtype of {parent}. This statement is {answer}.",
        "{parent} is the superclass of {child}. This statement is {answer}.",
    ]
    
    RELATION_TEMPLATES = [
        "{subject} {relation} {object}. This statement is {answer}.",
        "The relationship '{relation}' holds between {subject} and {object}. This statement is {answer}.",
    ]
    
    # Label mappings
    LABEL_MAPPINGS = {
        "correct": ["yes", "true", "correct", "right", "valid"],
        "incorrect": ["no", "false", "incorrect", "wrong", "invalid"]
    }
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        for directory in [cls.RESULTS_DIR, cls.LOGS_DIR]:
            directory.mkdir(exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate configuration and required files."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("Google API key not found in environment variables")
        
        required_files = [
            cls.UMLS_CLASSES_FILE,
            cls.UMLS_TERMS_FILE,
            cls.UMLS_HIERARCHY_FILE
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required data files: {missing_files}")
