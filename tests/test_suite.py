"""
Unified test suite for UMLS ontology learning system.
Replaces multiple scattered test files with a single comprehensive suite.
"""

import json
import time
import random
import sys
from typing import List, Dict, Tuple, Any
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from umls_ontology.core import UMLSOntologyLearner
from config.settings import Config


class UMLSTestSuite:
    """Comprehensive test suite for all UMLS ontology learning tasks."""
    
    def __init__(self):
        self.learner = UMLSOntologyLearner()
        self.results_dir = Config.RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and return comprehensive results."""
        print("=== UMLS Ontology Learning Test Suite ===\n")
        
        all_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_stats": self.learner.get_system_stats()
        }
        
        # Run individual test suites
        test_suites = [
            ("classification", self.test_term_classification),
            ("hierarchy", self.test_hierarchy_detection),
            ("relations", self.test_semantic_relations),
            ("integration", self.test_system_integration)
        ]
        
        for suite_name, test_function in test_suites:
            print(f"\n{'='*50}")
            print(f"Running {suite_name.title()} Tests")
            print(f"{'='*50}")
            
            try:
                suite_results = test_function()
                all_results[suite_name] = suite_results
                self._print_suite_summary(suite_name, suite_results)
            except Exception as e:
                print(f"ERROR in {suite_name} tests: {e}")
                all_results[suite_name] = {"error": str(e)}
        
        # Save comprehensive results
        self._save_results(all_results)
        self._print_overall_summary(all_results)
        
        return all_results
    
    def test_term_classification(self) -> Dict[str, Any]:
        """Test medical term classification functionality."""
        test_cases = [
            ("heart", ["Body Part, Organ, or Organ Component", "Anatomical Structure"]),
            ("diabetes", ["Disease or Syndrome"]),
            ("aspirin", ["Pharmacologic Substance"]),
            ("surgery", ["Therapeutic or Preventive Procedure"]),
            ("bacteria", ["Organism", "Bacterium"]),
            ("pain", ["Sign or Symptom", "Finding"]),
            ("muscle", ["Body Part, Organ, or Organ Component", "Anatomical Structure"]),
            ("pneumonia", ["Disease or Syndrome"])
        ]
        
        results = []
        correct = 0
        
        for term, expected_types in test_cases:
            predicted_type = self.learner.classifier.classify_term(term)
            is_correct = predicted_type in expected_types
            if is_correct:
                correct += 1
            
            result = {
                "term": term,
                "predicted": predicted_type,
                "expected": expected_types,
                "correct": is_correct
            }
            results.append(result)
            
            status = "[PASS]" if is_correct else "[FAIL]"
            print(f"{status} {term} -> {predicted_type} (expected: {expected_types})")
            
            time.sleep(1)  # Rate limiting
        
        accuracy = correct / len(test_cases)
        
        return {
            "test_cases": results,
            "metrics": {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(test_cases)
            }
        }
    
    def test_hierarchy_detection(self) -> Dict[str, Any]:
        """Test hierarchical relationship detection."""
        test_relationships = [
            ("Human", "Animal", True),
            ("Disease or Syndrome", "Pathologic Function", True),
            ("Body Part, Organ, or Organ Component", "Anatomical Structure", True),
            ("Pharmacologic Substance", "Chemical", True),
            ("Animal", "Human", False),  # Reverse relationship
            ("Chemical", "Pharmacologic Substance", False),  # Reverse relationship
            ("Unrelated Type A", "Unrelated Type B", False)  # No relationship
        ]
        
        results = []
        correct = 0
        
        for child, parent, expected in test_relationships:
            predicted = self.learner.relationship_analyzer.check_hierarchy(child, parent)
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            
            result = {
                "child": child,
                "parent": parent,
                "predicted": predicted,
                "expected": expected,
                "correct": is_correct
            }
            results.append(result)
            
            status = "[PASS]" if is_correct else "[FAIL]"
            print(f"{status} '{child}' -> '{parent}': {predicted} (expected: {expected})")
            
            time.sleep(1)  # Rate limiting
        
        accuracy = correct / len(test_relationships)
        
        return {
            "test_cases": results,
            "metrics": {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(test_relationships)
            }
        }
    
    def test_semantic_relations(self) -> Dict[str, Any]:
        """Test semantic relationship detection."""
        test_relations = [
            ("Pharmacologic Substance", "treats", "Disease or Syndrome", True),
            ("Disease or Syndrome", "affects", "Human", True),
            ("Therapeutic or Preventive Procedure", "treats", "Disease or Syndrome", True),
            ("Sign or Symptom", "indicates", "Disease or Syndrome", True),
            ("Human", "treats", "Pharmacologic Substance", False),  # Wrong direction
            ("Disease or Syndrome", "cures", "Human", False),  # Wrong relation
        ]
        
        results = []
        correct = 0
        
        for subject, relation, obj, expected in test_relations:
            predicted = self.learner.relationship_analyzer.check_semantic_relation(
                subject, relation, obj
            )
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            
            result = {
                "subject": subject,
                "relation": relation,
                "object": obj,
                "predicted": predicted,
                "expected": expected,
                "correct": is_correct
            }
            results.append(result)
            
            status = "[PASS]" if is_correct else "[FAIL]"
            print(f"{status} '{subject}' {relation} '{obj}': {predicted} (expected: {expected})")
            
            time.sleep(1)  # Rate limiting
        
        accuracy = correct / len(test_relations)
        
        return {
            "test_cases": results,
            "metrics": {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(test_relations)
            }
        }
    
    def test_system_integration(self) -> Dict[str, Any]:
        """Test overall system integration and performance."""
        # Test data loading
        stats = self.learner.get_system_stats()
        
        # Test classification demo
        classification_demo = self.learner.run_classification_demo(num_terms=3)
        
        # Test hierarchy demo  
        hierarchy_demo = self.learner.run_hierarchy_demo()
        
        integration_score = 0
        total_checks = 0
        
        # Check data loading
        if stats["semantic_types_count"] > 0:
            integration_score += 1
        total_checks += 1
        
        if stats["terms_count"] > 0:
            integration_score += 1
        total_checks += 1
        
        # Check classification functionality
        if classification_demo["summary"]["successful_classifications"] > 0:
            integration_score += 1
        total_checks += 1
        
        # Check hierarchy functionality
        if hierarchy_demo["summary"]["total_pairs"] > 0:
            integration_score += 1
        total_checks += 1
        
        integration_health = integration_score / total_checks
        
        return {
            "system_stats": stats,
            "classification_demo": classification_demo,
            "hierarchy_demo": hierarchy_demo,
            "integration_health": integration_health,
            "checks_passed": f"{integration_score}/{total_checks}"
        }
    
    def _print_suite_summary(self, suite_name: str, results: Dict[str, Any]):
        """Print summary for a test suite."""
        if "error" in results:
            print(f"\nERROR {suite_name.title()} Suite: ERROR - {results['error']}")
            return
        
        if "metrics" in results:
            metrics = results["metrics"]
            accuracy = metrics.get("accuracy", 0)
            correct = metrics.get("correct", 0)
            total = metrics.get("total", 0)
            
            status = "PASS" if accuracy >= 0.7 else "WARN" if accuracy >= 0.5 else "FAIL"
            print(f"\n{status} {suite_name.title()} Suite: {accuracy:.2%} ({correct}/{total})")
        else:
            print(f"\nPASS {suite_name.title()} Suite: Completed")
    
    def _print_overall_summary(self, results: Dict[str, Any]):
        """Print overall test summary."""
        print(f"\n{'='*60}")
        print("OVERALL TEST SUMMARY")
        print(f"{'='*60}")
        
        total_suites = 0
        successful_suites = 0
        
        for suite_name, suite_results in results.items():
            if suite_name in ["timestamp", "system_stats"]:
                continue
            
            total_suites += 1
            
            if "error" not in suite_results:
                if "metrics" in suite_results and suite_results["metrics"].get("accuracy", 0) >= 0.5:
                    successful_suites += 1
                elif "integration_health" in suite_results and suite_results["integration_health"] >= 0.7:
                    successful_suites += 1
        
        overall_health = successful_suites / total_suites if total_suites > 0 else 0
        
        print(f"Test Suites Passed: {successful_suites}/{total_suites}")
        print(f"Overall System Health: {overall_health:.2%}")
        
        if overall_health >= 0.8:
            print("System is performing well!")
        elif overall_health >= 0.6:
            print("System has some issues that need attention")
        else:
            print("System requires significant improvements")
        
        print(f"\nResults saved to: {self.results_dir}/comprehensive_test_results.json")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save test results to file."""
        output_file = self.results_dir / "comprehensive_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """Main function to run the test suite."""
    try:
        test_suite = UMLSTestSuite()
        test_suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user.")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
