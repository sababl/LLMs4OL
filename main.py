#!/usr/bin/env python3
"""
UMLS Ontology Learning - Simplified Main Entry Point

This is the single entry point for the UMLS ontology learning system.
Replaces the cluttered main.py with a clean, focused interface.
"""

import sys
import argparse
from pathlib import Path

from src.umls_ontology.core import UMLSOntologyLearner
from tests.test_suite import UMLSTestSuite


def run_demo():
    """Run a simple demonstration of the system."""
    print("=== UMLS Ontology Learning Demo ===\n")
    
    try:
        # Initialize the system
        learner = UMLSOntologyLearner()
        
        # Show system statistics
        stats = learner.get_system_stats()
        print("System Statistics:")
        print(f"  - Semantic Types: {stats['semantic_types_count']}")
        print(f"  - Medical Terms: {stats['terms_count']}")
        print(f"  - Hierarchy Entries: {stats['hierarchy_entries']}")
        
        # Run classification demo
        print("\n1. Term Classification Demo:")
        print("-" * 30)
        classification_results = learner.run_classification_demo(num_terms=3)
        for term, semantic_type in classification_results['results']:
            print(f"  '{term}' -> {semantic_type}")
        
        # Run hierarchy demo
        print("\n2. Hierarchy Detection Demo:")
        print("-" * 30)
        hierarchy_results = learner.run_hierarchy_demo()
        for child, parent, relationship in hierarchy_results['results']:
            status = "[PASS]" if relationship else "[FAIL]"
            print(f"  {status} '{child}' is subtype of '{parent}': {relationship}")
        
        print("\nDemo completed successfully!")
        print("\nNext steps:")
        print("  - Run full tests: python main.py --test")
        print("  - Interactive mode: python main.py --interactive")
        
    except Exception as e:
        print(f"ERROR: Demo failed: {e}")
        sys.exit(1)


def run_tests():
    """Run the comprehensive test suite."""
    print("Running comprehensive test suite...\n")
    
    try:
        test_suite = UMLSTestSuite()
        test_suite.run_all_tests()
    except Exception as e:
        print(f"ERROR: Tests failed: {e}")
        sys.exit(1)


def run_interactive():
    """Run interactive mode for manual testing."""
    print("=== Interactive UMLS Ontology Learning ===\n")
    
    try:
        learner = UMLSOntologyLearner()
        
        while True:
            print("\nOptions:")
            print("1. Classify a medical term")
            print("2. Check hierarchical relationship")
            print("3. System statistics")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                term = input("Enter medical term to classify: ").strip()
                if term:
                    result = learner.classifier.classify_term(term)
                    print(f"Result: '{term}' -> {result}")
            
            elif choice == "2":
                child = input("Enter child concept: ").strip()
                parent = input("Enter parent concept: ").strip()
                if child and parent:
                    result = learner.relationship_analyzer.check_hierarchy(child, parent)
                    print(f"Result: '{child}' is subtype of '{parent}': {result}")
            
            elif choice == "3":
                stats = learner.get_system_stats()
                print("\nSystem Statistics:")
                for key, value in stats.items():
                    if isinstance(value, list):
                        value = f"{len(value)} items"
                    print(f"  {key}: {value}")
            
            elif choice == "4":
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\n\nExiting interactive mode.")
    except Exception as e:
        print(f"ERROR: Interactive mode failed: {e}")
        sys.exit(1)


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="UMLS Ontology Learning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                 # Run demo
  python main.py --test          # Run full test suite
  python main.py --interactive   # Interactive mode
        """
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run comprehensive test suite"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true", 
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.interactive:
        run_interactive()
    else:
        run_demo()


if __name__ == "__main__":
    main()
