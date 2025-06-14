"""
Simplified Task B: Taxonomy Discovery for UMLS
Identifies "is-a" hierarchies between UMLS semantic types.
"""

import json
import random
from typing import List, Dict, Tuple
from sklearn.metrics import classification_report, f1_score
import itertools


class UMLSTaxonomyDiscovery:
    def __init__(self, hierarchy_path: str, classes_path: str):
        """Initialize with UMLS data paths."""
        self.hierarchy_path = hierarchy_path
        self.classes_path = classes_path
        self.hierarchy = self._load_hierarchy()
        self.classes = self._load_classes()
        self.templates = [
            "{text_a} is the superclass of {text_b}. This statement is {answer}.",
            "{text_b} is a subclass of {text_a}. This statement is {answer}.",
            "{text_a} is the parent class of {text_b}. This statement is {answer}.",
            "{text_b} is a child class of {text_a}. This statement is {answer}."
        ]
        self.label_mapper = {
            "correct": ["correct", "true", "yes", "valid"],
            "incorrect": ["incorrect", "false", "no", "invalid"]
        }
    
    def _load_hierarchy(self) -> Dict[str, List[str]]:
        """Load UMLS hierarchy data."""
        with open(self.hierarchy_path, 'r') as f:
            return json.load(f)
    
    def _load_classes(self) -> Dict[str, Dict]:
        """Load UMLS classes data."""
        with open(self.classes_path, 'r') as f:
            return json.load(f)
    
    def build_taxonomy_pairs(self) -> List[Dict]:
        """Build positive and negative examples for taxonomy discovery."""
        positive_pairs = []
        all_concepts = set()
        
        # Extract positive pairs from hierarchy
        for child, parents in self.hierarchy.items():
            all_concepts.add(child.lower())
            for parent in parents:
                all_concepts.add(parent.lower())
                positive_pairs.append((parent.lower(), child.lower()))
        
        # Add transitive relationships (if A->B and B->C, then A->C)
        extended_pairs = set(positive_pairs)
        for _ in range(3):  # Apply transitivity multiple times
            new_pairs = set()
            for p1 in extended_pairs:
                for p2 in extended_pairs:
                    if p1[1] == p2[0]:  # B in A->B matches A in B->C
                        new_pairs.add((p1[0], p2[1]))  # Add A->C
            extended_pairs.update(new_pairs)
        
        positive_pairs = list(extended_pairs)
        all_concepts = list(all_concepts)
        
        # Create dataset with positive and negative examples
        dataset = []
        
        # Add positive examples
        for parent, child in positive_pairs:
            dataset.append({
                "text_a": parent,
                "text_b": child,
                "label": "correct"
            })
        
        # Create negative examples by randomly pairing concepts
        negative_count = min(len(positive_pairs), 1000)  # Limit negative examples
        positive_set = set(positive_pairs)
        
        negative_pairs = []
        attempts = 0
        while len(negative_pairs) < negative_count and attempts < negative_count * 10:
            concept1 = random.choice(all_concepts)
            concept2 = random.choice(all_concepts)
            if (concept1, concept2) not in positive_set and concept1 != concept2:
                negative_pairs.append((concept1, concept2))
            attempts += 1
        
        # Add negative examples
        for parent, child in negative_pairs:
            dataset.append({
                "text_a": parent,
                "text_b": child,
                "label": "incorrect"
            })
        
        # Shuffle dataset
        random.shuffle(dataset)
        return dataset
    
    def create_prompts(self, dataset: List[Dict], template_idx: int = 0) -> List[str]:
        """Create prompts using specified template."""
        template = self.templates[template_idx]
        prompts = []
        
        for item in dataset:
            # For evaluation, we'll use a masked version
            prompt = template.format(
                text_a=item["text_a"],
                text_b=item["text_b"],
                answer="[MASK]"
            )
            prompts.append(prompt)
        
        return prompts
    
    def simulate_predictions(self, dataset: List[Dict]) -> List[str]:
        """Simulate model predictions (placeholder for actual model inference)."""
        predictions = []
        
        for item in dataset:
            # Simple heuristic: check if there's a known hierarchy relationship
            parent, child = item["text_a"], item["text_b"]
            
            # Check direct relationship
            if parent in self.hierarchy.get(child, []):
                predictions.append("correct")
            elif child in self.hierarchy.get(parent, []):
                predictions.append("incorrect")  # Wrong direction
            else:
                # Random prediction for unknown relationships
                predictions.append(random.choice(["correct", "incorrect"]))
        
        return predictions
    
    def evaluate(self, true_labels: List[str], predictions: List[str]) -> Dict:
        """Evaluate predictions against true labels."""
        # Map predictions to binary labels
        pred_binary = [1 if pred == "correct" else 0 for pred in predictions]
        true_binary = [1 if true == "correct" else 0 for true in true_labels]
        
        # Calculate metrics
        f1 = f1_score(true_binary, pred_binary, average='macro')
        report = classification_report(true_binary, pred_binary, 
                                     target_names=['incorrect', 'correct'], 
                                     output_dict=True)
        
        return {
            "f1_score": f1,
            "classification_report": report,
            "accuracy": report['accuracy']
        }
    
    def run_experiment(self, template_idx: int = 0, test_size: float = 0.3) -> Dict:
        """Run complete taxonomy discovery experiment."""
        print("Building taxonomy pairs...")
        dataset = self.build_taxonomy_pairs()
        print(f"Created {len(dataset)} examples")
        
        # Split into train/test
        split_idx = int(len(dataset) * (1 - test_size))
        train_data = dataset[:split_idx]
        test_data = dataset[split_idx:]
        
        print(f"Train: {len(train_data)}, Test: {len(test_data)}")
        
        # Create prompts
        test_prompts = self.create_prompts(test_data, template_idx)
        
        # Get predictions (simulated)
        print("Generating predictions...")
        predictions = self.simulate_predictions(test_data)
        true_labels = [item["label"] for item in test_data]
        
        # Evaluate
        results = self.evaluate(true_labels, predictions)
        
        return {
            "template_used": self.templates[template_idx],
            "dataset_size": len(dataset),
            "test_size": len(test_data),
            "results": results,
            "sample_prompts": test_prompts[:5],  # Show first 5 prompts
            "sample_predictions": list(zip(true_labels[:5], predictions[:5]))
        }


def main():
    """Main function to run taxonomy discovery."""
    # Initialize with data paths
    hierarchy_path = "/home/saba/university/S&DAI/LLMs4OL/data/umls_hierarchy.json"
    classes_path = "/home/saba/university/S&DAI/LLMs4OL/data/umls_classes.json"
    
    # Create taxonomy discovery instance
    taxonomy_discovery = UMLSTaxonomyDiscovery(hierarchy_path, classes_path)
    
    # Run experiment with different templates
    for template_idx in range(len(taxonomy_discovery.templates)):
        print(f"\n{'='*60}")
        print(f"Running experiment with template {template_idx + 1}")
        print(f"Template: {taxonomy_discovery.templates[template_idx]}")
        print(f"{'='*60}")
        
        results = taxonomy_discovery.run_experiment(template_idx=template_idx)
        
        print(f"\nResults:")
        print(f"F1-Score: {results['results']['f1_score']:.4f}")
        print(f"Accuracy: {results['results']['accuracy']:.4f}")
        
        print(f"\nSample prompts:")
        for i, prompt in enumerate(results['sample_prompts'][:3]):
            print(f"{i+1}. {prompt}")
        
        print(f"\nSample predictions (true, predicted):")
        for true_label, pred_label in results['sample_predictions'][:3]:
            print(f"  True: {true_label}, Predicted: {pred_label}")


if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    main()
