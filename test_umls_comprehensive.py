#!/usr/bin/env python3
"""
Comprehensive Test Suite for UMLS Term Typing System

This module consolidates all testing functionality into a single, clean interface.
Replaces multiple test files with one comprehensive test suite.
"""

import os
import sys
import json
import time
import random
from typing import List, Dict, Tuple
from umls_typer import UMLSTermTyper

class UMLSTestSuite:
    """Comprehensive test suite for UMLS term typing system."""
    
    def __init__(self):
        """Initialize the test suite."""
        try:
            self.typer = UMLSTermTyper()
            self.results = {}
        except Exception as e:
            print(f"❌ Failed to initialize UMLS typer: {e}")
            sys.exit(1)
    
    def test_basic_classification(self) -> Dict:
        """Test basic term classification functionality."""
        print("=== Basic Classification Test ===\n")
        
        test_cases = [
            ("heart", ["Anatomical Structure", "Body Part, Organ, or Organ Component"]),
            ("diabetes", ["Disease or Syndrome"]),
            ("aspirin", ["Pharmacologic Substance", "Organic Chemical"]),
            ("surgery", ["Therapeutic or Preventive Procedure"]),
            ("bacteria", ["Organism", "Bacterium"]),
            ("pain", ["Finding", "Sign or Symptom"]),
            ("muscle", ["Anatomical Structure", "Body Part, Organ, or Organ Component"]),
            ("treatment", ["Therapeutic or Preventive Procedure"]),
            ("insulin", ["Pharmacologic Substance", "Hormone"]),
            ("pneumonia", ["Disease or Syndrome"])
        ]
        
        results = []
        correct = 0
        total = len(test_cases)
        
        print(f"Testing {total} medical terms with expected classifications:")
        
        for term, expected_types in test_cases:
            try:
                result = self.typer.classify_term(term)
                
                # Check if result matches any expected type
                is_correct = any(
                    expected.lower() in result.lower() or result.lower() in expected.lower()
                    for expected in expected_types
                )
                
                if is_correct:
                    correct += 1
                
                status = "✓" if is_correct else "✗"
                expected_str = " or ".join(expected_types)
                print(f"  {status} '{term}' → '{result}' (expected: {expected_str})")
                
                results.append({
                    "term": term,
                    "result": result,
                    "expected": expected_types,
                    "correct": is_correct
                })
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"  ✗ Error classifying '{term}': {e}")
                results.append({
                    "term": term,
                    "result": "ERROR",
                    "expected": expected_types,
                    "correct": False
                })
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"\nBasic Classification Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        
        return {
            "test_name": "basic_classification",
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
    
    def test_domain_specific_categories(self) -> Dict:
        """Test classification across different medical domain categories."""
        print("\n=== Domain-Specific Category Test ===\n")
        
        categories = {
            "Anatomical": ["brain", "liver", "kidney", "lung"],
            "Disease": ["cancer", "tuberculosis", "malaria", "asthma"],
            "Pharmaceutical": ["penicillin", "morphine", "warfarin", "ibuprofen"],
            "Organism": ["virus", "fungus", "parasite"],
            "Procedure": ["biopsy", "vaccination", "chemotherapy"]
        }
        
        category_results = {}
        
        for category, terms in categories.items():
            print(f"Testing {category} terms:")
            category_correct = 0
            category_total = len(terms)
            term_results = []
            
            for term in terms:
                try:
                    result = self.typer.classify_term(term)
                    
                    # Check if result is domain-appropriate
                    is_correct = self._is_domain_appropriate(result, category)
                    
                    if is_correct:
                        category_correct += 1
                    
                    status = "✓" if is_correct else "✗"
                    print(f"  {status} '{term}' → '{result}'")
                    
                    term_results.append({
                        "term": term,
                        "result": result,
                        "correct": is_correct
                    })
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    print(f"  ✗ Error with '{term}': {e}")
                    term_results.append({
                        "term": term,
                        "result": "ERROR",
                        "correct": False
                    })
            
            category_accuracy = (category_correct / category_total) * 100 if category_total > 0 else 0
            print(f"  {category} Accuracy: {category_correct}/{category_total} ({category_accuracy:.1f}%)\n")
            
            category_results[category] = {
                "accuracy": category_accuracy,
                "correct": category_correct,
                "total": category_total,
                "results": term_results
            }
        
        return {
            "test_name": "domain_categories",
            "categories": category_results
        }
    
    def test_random_terms(self, num_terms: int = 10) -> Dict:
        """Test classification of random terms from the UMLS dataset."""
        print(f"\n=== Random Terms Test ({num_terms} terms) ===\n")
        
        # Get random terms from loaded dataset
        random_terms = random.sample(self.typer.umls_terms, min(num_terms, len(self.typer.umls_terms)))
        
        results = []
        predicted_types = set()
        
        print("Classifying random medical terms:")
        
        for term in random_terms:
            try:
                result = self.typer.classify_term(term)
                predicted_types.add(result)
                results.append({"term": term, "result": result})
                print(f"  '{term}' → '{result}'")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"  Error with '{term}': {e}")
                results.append({"term": term, "result": "ERROR"})
        
        coverage = (len(predicted_types) / len(self.typer.semantic_types)) * 100
        print(f"\nSemantic Type Coverage: {len(predicted_types)}/{len(self.typer.semantic_types)} ({coverage:.1f}%)")
        print(f"Predicted types: {sorted(list(predicted_types))}")
        
        return {
            "test_name": "random_terms",
            "coverage": coverage,
            "predicted_types": list(predicted_types),
            "results": results
        }
    
    def test_hierarchical_relationships(self, num_pairs: int = 6) -> Dict:
        """Test hierarchical relationship detection."""
        print(f"\n=== Hierarchical Relationships Test ({num_pairs} pairs) ===\n")
        
        # Create test pairs from known hierarchical relationships
        test_pairs = [
            ("Anatomical Abnormality", "Anatomical Structure", True),
            ("Embryonic Structure", "Anatomical Structure", True), 
            ("Fully Formed Anatomical Structure", "Anatomical Structure", True),
            ("Antibiotic", "Pharmacologic Substance", True),
            ("Hormone", "Pharmacologic Substance", True),
            ("Pharmacologic Substance", "Anatomical Structure", False),
            ("Disease or Syndrome", "Pharmacologic Substance", False),
            ("Organism", "Chemical", False)
        ]
        
        # Sample the pairs
        selected_pairs = random.sample(test_pairs, min(num_pairs, len(test_pairs)))
        
        results = []
        correct = 0
        total = len(selected_pairs)
        
        print("Testing hierarchical relationships:")
        
        for child, parent, expected in selected_pairs:
            try:
                predicted = self.typer.check_hierarchical_relationship(child, parent)
                is_correct = predicted == expected
                
                if is_correct:
                    correct += 1
                
                status = "✓" if is_correct else "✗"
                print(f"  {status} '{child}' isa '{parent}': Expected={expected}, Predicted={predicted}")
                
                results.append({
                    "child": child,
                    "parent": parent,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": is_correct
                })
                
                time.sleep(2)  # Longer delay for more complex queries
                
            except Exception as e:
                print(f"  ✗ Error checking '{child}' -> '{parent}': {e}")
                results.append({
                    "child": child,
                    "parent": parent,
                    "expected": expected,
                    "predicted": None,
                    "correct": False
                })
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"\nHierarchical Relationship Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        
        return {
            "test_name": "hierarchical_relationships",
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
    
    def test_performance_metrics(self) -> Dict:
        """Test performance and timing metrics."""
        print("\n=== Performance Metrics Test ===\n")
        
        test_terms = ["diabetes", "aspirin", "heart", "surgery"]
        performance_data = []
        
        print("Measuring classification timing:")
        
        for term in test_terms:
            start_time = time.time()
            result = self.typer.classify_term(term)
            end_time = time.time()
            
            duration = end_time - start_time
            print(f"  '{term}' → {result} ({duration:.2f}s)")
            
            performance_data.append({
                "term": term,
                "result": result,
                "duration": duration
            })
            
            time.sleep(1)  # Rate limiting
        
        avg_duration = sum(p["duration"] for p in performance_data) / len(performance_data)
        print(f"\nAverage classification time: {avg_duration:.2f}s")
        
        return {
            "test_name": "performance_metrics",
            "average_duration": avg_duration,
            "performance_data": performance_data
        }
    
    def _is_domain_appropriate(self, result: str, category: str) -> bool:
        """Check if classification result is appropriate for the domain category."""
        domain_mappings = {
            "Anatomical": ["anatomical", "structure", "organ", "body part", "tissue"],
            "Disease": ["disease", "syndrome", "disorder", "condition", "dysfunction"],
            "Pharmaceutical": ["pharmacologic", "substance", "chemical", "drug", "medication"],
            "Organism": ["organism", "bacterium", "virus", "animal", "plant"],
            "Procedure": ["procedure", "activity", "treatment", "therapy", "intervention"]
        }
        
        expected_keywords = domain_mappings.get(category, [])
        result_lower = result.lower()
        
        return any(keyword in result_lower for keyword in expected_keywords)
    
    def run_all_tests(self) -> Dict:
        """Run all tests and return comprehensive results."""
        print("🧬 UMLS Term Typing System - Comprehensive Test Suite")
        print("=" * 60)
        
        # Environment check
        if not self._check_environment():
            return {"error": "Environment check failed"}
        
        # Initialize results
        all_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": self.typer.get_stats(),
            "tests": {}
        }
        
        try:
            # Run all test suites
            all_results["tests"]["basic_classification"] = self.test_basic_classification()
            all_results["tests"]["domain_categories"] = self.test_domain_specific_categories()
            all_results["tests"]["random_terms"] = self.test_random_terms(8)
            all_results["tests"]["hierarchical_relationships"] = self.test_hierarchical_relationships(6)
            all_results["tests"]["performance_metrics"] = self.test_performance_metrics()
            
            # Generate summary
            self._print_summary(all_results)
            
            # Save results
            self._save_results(all_results)
            
            return all_results
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Test suite interrupted by user")
            return {"error": "Interrupted by user"}
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            return {"error": str(e)}
    
    def _check_environment(self) -> bool:
        """Check if environment is properly configured."""
        print("=== Environment Check ===\n")
        
        checks_passed = 0
        total_checks = 0
        
        # API Key check
        total_checks += 1
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            print("✓ API key configured")
            checks_passed += 1
        else:
            print("❌ No API key found")
        
        # Data files check
        data_files = [
            "data/umls_terms.txt",
            "data/umls_classes.json",
            "data/umls_hierarchy.json"
        ]
        
        for file_path in data_files:
            total_checks += 1
            if os.path.exists(file_path):
                print(f"✓ {file_path}")
                checks_passed += 1
            else:
                print(f"❌ {file_path}")
        
        success_rate = (checks_passed / total_checks) * 100
        print(f"\nEnvironment Check: {checks_passed}/{total_checks} ({success_rate:.1f}%)")
        
        if checks_passed < total_checks:
            print("Please fix the issues above before running tests.")
        
        print()
        return checks_passed == total_checks
    
    def _print_summary(self, results: Dict):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("🎉 Test Suite Completed!")
        print("\nSummary:")
        
        tests = results.get("tests", {})
        
        if "basic_classification" in tests:
            basic = tests["basic_classification"]
            print(f"• Basic Classification: {basic['correct']}/{basic['total']} ({basic['accuracy']:.1f}%)")
        
        if "domain_categories" in tests:
            domain = tests["domain_categories"]
            for category, data in domain["categories"].items():
                print(f"• {category} Terms: {data['correct']}/{data['total']} ({data['accuracy']:.1f}%)")
        
        if "random_terms" in tests:
            random_test = tests["random_terms"]
            print(f"• Random Terms Coverage: {random_test['coverage']:.1f}%")
        
        if "hierarchical_relationships" in tests:
            hierarchy = tests["hierarchical_relationships"]
            print(f"• Hierarchical Relationships: {hierarchy['correct']}/{hierarchy['total']} ({hierarchy['accuracy']:.1f}%)")
        
        if "performance_metrics" in tests:
            perf = tests["performance_metrics"]
            print(f"• Average Response Time: {perf['average_duration']:.2f}s")
    
    def _save_results(self, results: Dict):
        """Save test results to file."""
        output_file = "umls_comprehensive_test_results.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Test results saved to '{output_file}'")


def main():
    """Main function to run the test suite."""
    test_suite = UMLSTestSuite()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()
