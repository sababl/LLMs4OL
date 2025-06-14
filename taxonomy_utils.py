"""
Configuration and utilities for UMLS Taxonomy Discovery
"""

import json
from pathlib import Path


class Config:
    """Configuration for UMLS taxonomy discovery."""
    
    def __init__(self):
        self.data_dir = Path("/home/saba/university/S&DAI/LLMs4OL/data")
        self.hierarchy_path = self.data_dir / "umls_hierarchy.json"
        self.classes_path = self.data_dir / "umls_classes.json"
        
        # Templates for different prompt styles
        self.templates = [
            "{text_a} is the superclass of {text_b}. This statement is {answer}.",
            "{text_b} is a subclass of {text_a}. This statement is {answer}.",
            "{text_a} is the parent class of {text_b}. This statement is {answer}.",
            "{text_b} is a child class of {text_a}. This statement is {answer}.",
            "{text_a} is a supertype of {text_b}. This statement is {answer}.",
            "{text_b} is a subtype of {text_a}. This statement is {answer}.",
            "{text_a} is an ancestor class of {text_b}. This statement is {answer}.",
            "{text_b} is a descendant class of {text_a}. This statement is {answer}."
        ]
        
        # Label mappings
        self.label_mapper = {
            "correct": ["yes", "true", "correct", "right", "valid"],
            "incorrect": ["no", "false", "incorrect", "wrong", "invalid"]
        }


class DataUtils:
    """Utility functions for data processing."""
    
    @staticmethod
    def load_json(file_path: str) -> dict:
        """Load JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: dict, file_path: str) -> None:
        """Save data to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def normalize_concept_name(name: str) -> str:
        """Normalize concept names for consistency."""
        return name.lower().strip()
    
    @staticmethod
    def build_transitive_closure(hierarchy: dict) -> set:
        """Build transitive closure of hierarchy relationships."""
        pairs = set()
        
        # Add direct relationships
        for child, parents in hierarchy.items():
            for parent in parents:
                pairs.add((DataUtils.normalize_concept_name(parent), 
                          DataUtils.normalize_concept_name(child)))
        
        # Add transitive relationships
        changed = True
        iterations = 0
        max_iterations = 10  # Prevent infinite loops
        
        while changed and iterations < max_iterations:
            changed = False
            new_pairs = set()
            
            for p1 in pairs:
                for p2 in pairs:
                    if p1[1] == p2[0]:  # B in A->B matches A in B->C
                        new_pair = (p1[0], p2[1])  # Add A->C
                        if new_pair not in pairs:
                            new_pairs.add(new_pair)
                            changed = True
            
            pairs.update(new_pairs)
            iterations += 1
        
        return pairs


class EvaluationMetrics:
    """Evaluation metrics for taxonomy discovery."""
    
    @staticmethod
    def calculate_f1(true_labels, predicted_labels):
        """Calculate F1 score."""
        from sklearn.metrics import f1_score
        
        # Convert labels to binary
        true_binary = [1 if label == "correct" else 0 for label in true_labels]
        pred_binary = [1 if label == "correct" else 0 for label in predicted_labels]
        
        return f1_score(true_binary, pred_binary, average='macro')
    
    @staticmethod
    def calculate_accuracy(true_labels, predicted_labels):
        """Calculate accuracy."""
        correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
        return correct / len(true_labels) if true_labels else 0
    
    @staticmethod
    def detailed_report(true_labels, predicted_labels):
        """Generate detailed classification report."""
        from sklearn.metrics import classification_report
        
        # Convert labels to binary
        true_binary = [1 if label == "correct" else 0 for label in true_labels]
        pred_binary = [1 if label == "correct" else 0 for label in predicted_labels]
        
        return classification_report(
            true_binary, pred_binary,
            target_names=['incorrect', 'correct'],
            output_dict=True
        )
