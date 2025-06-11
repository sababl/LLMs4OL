#!/usr/bin/env python3

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set GEMINI_API_KEY from GOOGLE_API_KEY if not set
if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

sys.path.append('src')

from term_typing import classify_random_terms, classify_term

def test_comprehensive():
    """Comprehensive test of the term typing implementation"""
    
    wordnet_pos_classes = ["noun", "verb", "adjective", "adverb"]
    
    print("=== Comprehensive Term Typing Test ===\n")
    
    # Test 1: Individual term classification
    print("1. Testing individual terms for different POS types:")
    test_cases = [
        ("cat", "noun"),
        ("run", "verb"), 
        ("beautiful", "adjective"),
        ("quickly", "adverb"),
        ("swimming", "verb"),  # could be noun (gerund) or verb
        ("happy", "adjective"),
        ("tree", "noun")
    ]
    
    for term, expected in test_cases:
        result = classify_term(term, wordnet_pos_classes)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {term} -> {result} (expected: {expected})")
    
    print("\n" + "="*50)
    
    # Test 2: Random terms classification and Prolog output
    print("\n2. Testing random terms classification (10 terms):")
    
    # Clear previous test results from ontology file (keep only the header)
    with open("prolog/ontology.pl", "w") as f:
        f.write("% Generated ontology facts\n")
    
    results = classify_random_terms(
        num_terms=10,
        classes=wordnet_pos_classes,
        terms_file_path="data/terms.txt",
        ontology_file_path="prolog/ontology.pl"
    )
    
    print(f"\nClassification Results:")
    pos_counts = {"noun": 0, "verb": 0, "adjective": 0, "adverb": 0, "unknown": 0}
    
    for term, pos in results:
        pos_counts[pos] += 1
        print(f"  {term:15} -> {pos}")
    
    print(f"\nPOS Distribution:")
    for pos, count in pos_counts.items():
        if count > 0:
            print(f"  {pos:10}: {count} terms")
    
    print("\n" + "="*50)
    
    # Test 3: Verify Prolog file output
    print("\n3. Verifying Prolog ontology file:")
    
    try:
        with open("prolog/ontology.pl", "r") as f:
            ontology_content = f.read()
            
        lines = [line.strip() for line in ontology_content.split('\n') if line.strip() and not line.startswith('%')]
        print(f"  Generated {len(lines)} Prolog facts")
        print("  Sample facts:")
        for i, line in enumerate(lines[:5]):  # Show first 5 facts
            print(f"    {line}")
        if len(lines) > 5:
            print(f"    ... and {len(lines) - 5} more")
            
    except Exception as e:
        print(f"  Error reading ontology file: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_comprehensive()
