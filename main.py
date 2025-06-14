#!/usr/bin/env python3
"""
LLMs4OL - Large Language Models for Ontology Learning
UMLS Domain Implementation (Clean Version)

This script demonstrates ontology learning tasks using the UMLS (Unified Medical Language System)
domain with Google's Gemini model. Uses the consolidated clean implementation.
"""

import time
import os
from umls_typer import UMLSTermTyper
from umls_data_utils import UMLSDataManager

if __name__ == "__main__":
    print("=== LLMs4OL - UMLS Domain Ontology Learning ===\n")
    
    # Initialize UMLS system
    print("Initializing UMLS medical term classification system...")
    try:
        # Initialize the typer and data manager
        typer = UMLSTermTyper()
        data_manager = UMLSDataManager()
        
        # Show stats
        stats = typer.get_stats()
        print(f"✓ Loaded {stats['semantic_types_count']} semantic types and {stats['terms_count']} terms")
        print(f"Sample semantic types: {stats['semantic_types'][:5]}")
        print(f"Sample terms: {typer.umls_terms[:10]}\n")
        
        # Test medical term classification
        print("1. Testing medical term classification...")
        
        sample_terms = typer.umls_terms[:5]  # Use first 5 terms for testing
        
        classification_results = []
        for term in sample_terms:
            print(f"Classifying: {term}")
            classified_type = typer.classify_term(term)
            classification_results.append((term, classified_type))
            print(f"  {term} -> {classified_type}")
            time.sleep(2)  # Rate limiting
        
        print("\n" + "="*50 + "\n")
        
        # Test hierarchical relationship detection
        print("2. Testing medical relationship detection...")
        
        # Test some hierarchical relationships
        test_relationships = [
            ("Human", "Animal"),
            ("Disease or Syndrome", "Pathologic Function"),
            ("Body Part, Organ, or Organ Component", "Anatomical Structure"),
            ("Pharmacologic Substance", "Chemical")
        ]
        
        relationship_results = []
        for child, parent in test_relationships:
            result = typer.check_hierarchical_relationship(child, parent)
            relationship_results.append((child, parent, result))
            print(f"Is '{child}' a subtype of '{parent}'? {result}")
            time.sleep(2)
        
        print("\n" + "="*50 + "\n")
        
        # Show summary
        print("3. Summary and next steps...")
        print("✅ UMLS term classification system is working")
        print("✅ Hierarchical relationship detection is functional")
        print(f"✅ Processed {len(classification_results)} terms")
        print(f"✅ Tested {len(relationship_results)} relationships")
        
        print("\nNext steps:")
        print("- Run 'python test_umls_comprehensive.py' for detailed testing")
        print("- Use 'python umls_typer.py' for interactive demo")
        print("- Check 'python umls_data_utils.py' for data management")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required data files: {e}")
        print("Please run 'python umls_data_utils.py' to create sample data")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check your configuration and data files.")
