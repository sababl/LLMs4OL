"""
Simplified Task C: Non-Taxonomic Relation Extraction for UMLS
Identifies semantic relationships between UMLS semantic types (non "is-a").
"""

import json
import random
from typing import List, Dict, Tuple, Set
from sklearn.metrics import classification_report, f1_score
from collections import defaultdict


class UMLSRelationExtraction:
    def __init__(self, hierarchy_path: str, classes_path: str, relationships_path: str):
        """Initialize with UMLS data paths."""
        self.hierarchy_path = hierarchy_path
        self.classes_path = classes_path
        self.relationships_path = relationships_path
        self.hierarchy = self._load_hierarchy()
        self.classes = self._load_classes()
        self.relationships = self._load_relationships()
        
        # Templates for relation extraction
        self.templates = [
            "{h} {r} {t}. This statement is {answer}.",
            "The relationship '{r}' holds between {h} and {t}. This statement is {answer}.",
            "{h} has the relation '{r}' with {t}. This statement is {answer}.",
            "'{h}' {r} '{t}'. This statement is {answer}.",
            "In the semantic network, {h} {r} {t}. This statement is {answer}.",
            "The semantic relation '{r}' connects {h} to {t}. This statement is {answer}.",
            "{h} and {t} are related by '{r}'. This statement is {answer}.",
            "There exists a '{r}' relationship from {h} to {t}. This statement is {answer}."
        ]
        
        # Known UMLS semantic relations (non-taxonomic)
        self.semantic_relations = {
            "affects", "associated_with", "treats", "prevents", "diagnoses",
            "causes", "manifestation_of", "result_of", "location_of", "part_of",
            "connected_to", "adjacent_to", "surrounds", "traverses", "contains",
            "uses", "performs", "carries_out", "occurs_in", "produced_by",
            "exhibits", "indicates", "measures", "method_of", "process_of",
            "property_of", "conceptually_related_to", "functionally_related_to",
            "temporally_related_to", "spatially_related_to", "causally_related_to"
        }
        
        self.label_mapper = {
            "correct": ["correct", "true", "yes", "valid", "accurate"],
            "incorrect": ["incorrect", "false", "no", "invalid", "inaccurate"]
        }
    
    def _load_hierarchy(self) -> Dict[str, List[str]]:
        """Load UMLS hierarchy data."""
        with open(self.hierarchy_path, 'r') as f:
            return json.load(f)
    
    def _load_classes(self) -> Dict[str, Dict]:
        """Load UMLS classes data."""
        with open(self.classes_path, 'r') as f:
            return json.load(f)
    
    def _load_relationships(self) -> List[Dict]:
        """Load UMLS relationship samples."""
        try:
            with open(self.relationships_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Relationship samples file not found. Will generate synthetic data.")
            return []
    
    def extract_concepts_from_hierarchy(self) -> Set[str]:
        """Extract all concept names from hierarchy."""
        concepts = set()
        for child, parents in self.hierarchy.items():
            concepts.add(child.lower())
            for parent in parents:
                concepts.add(parent.lower())
        return concepts
    
    def generate_synthetic_relations(self, concepts: List[str], num_relations: int = 1000) -> List[Dict]:
        """Generate synthetic non-taxonomic relations between concepts."""
        relations = []
        
        # Create some domain-specific relation mappings
        medical_relations = {
            "treats": ["disease", "syndrome", "disorder", "pathology"],
            "causes": ["disease", "disorder", "pathology", "abnormality"],
            "prevents": ["disease", "disorder", "pathology"],
            "diagnoses": ["disease", "syndrome", "disorder"],
            "affects": ["function", "process", "activity"],
            "location_of": ["structure", "component", "part"],
            "part_of": ["system", "structure", "entity"],
            "uses": ["substance", "chemical", "device", "method"],
            "produces": ["substance", "chemical", "product"],
            "performs": ["activity", "function", "process"],
            "occurs_in": ["location", "region", "space", "structure"],
            "contains": ["substance", "component", "element"],
            "connected_to": ["structure", "component", "part"],
            "associated_with": ["concept", "entity", "phenomenon"]
        }
        
        for _ in range(num_relations):
            h = random.choice(concepts)
            t = random.choice(concepts)
            
            if h != t:  # Avoid self-relations
                # Select appropriate relation based on concept types
                relation = random.choice(list(self.semantic_relations))
                
                # Create positive example
                relations.append({
                    "h": h,
                    "r": relation,
                    "t": t,
                    "label": "correct"
                })
        
        return relations
    
    def build_relation_dataset(self, max_examples: int = 2000) -> List[Dict]:
        """Build dataset of positive and negative relation examples."""
        concepts = list(self.extract_concepts_from_hierarchy())
        
        # Load existing relationships or generate synthetic ones
        if self.relationships:
            positive_relations = [
                {
                    "h": rel.get("child", "").lower(),
                    "r": rel.get("relationship", "unknown"),
                    "t": rel.get("parent", "").lower(),
                    "label": "correct"
                }
                for rel in self.relationships 
                if rel.get("relationship", "") != "isa" and rel.get("label", "") == "positive"
            ]
        else:
            positive_relations = self.generate_synthetic_relations(concepts, max_examples // 2)
        
        # Limit positive examples
        if len(positive_relations) > max_examples // 2:
            positive_relations = random.sample(positive_relations, max_examples // 2)
        
        dataset = positive_relations.copy()
        
        # Generate negative examples
        negative_count = len(positive_relations)
        positive_triples = {(rel["h"], rel["r"], rel["t"]) for rel in positive_relations}
        
        negative_relations = []
        attempts = 0
        max_attempts = negative_count * 10
        
        while len(negative_relations) < negative_count and attempts < max_attempts:
            h = random.choice(concepts)
            r = random.choice(list(self.semantic_relations))
            t = random.choice(concepts)
            
            if h != t and (h, r, t) not in positive_triples:
                negative_relations.append({
                    "h": h,
                    "r": r,
                    "t": t,
                    "label": "incorrect"
                })
            
            attempts += 1
        
        dataset.extend(negative_relations)
        
        # Shuffle dataset
        random.shuffle(dataset)
        return dataset
    
    def create_prompts(self, dataset: List[Dict], template_idx: int = 0) -> List[str]:
        """Create prompts using specified template."""
        template = self.templates[template_idx]
        prompts = []
        
        for item in dataset:
            prompt = template.format(
                h=item["h"],
                r=item["r"],
                t=item["t"],
                answer="[MASK]"
            )
            prompts.append(prompt)
        
        return prompts
    
    def simple_relation_classifier(self, dataset: List[Dict]) -> List[str]:
        """Simple baseline classifier using heuristics."""
        predictions = []
        
        # Simple heuristics based on concept types and relations
        for item in dataset:
            h, r, t = item["h"], item["r"], item["t"]
            
            # Heuristic 1: Common medical relations
            if any(keyword in h for keyword in ["disease", "disorder", "syndrome"]) and \
               r in ["treats", "prevents", "causes"] and \
               any(keyword in t for keyword in ["substance", "therapy", "treatment"]):
                predictions.append("correct")
            
            # Heuristic 2: Anatomical relations
            elif any(keyword in h for keyword in ["organ", "structure", "component"]) and \
                 r in ["part_of", "location_of", "connected_to"] and \
                 any(keyword in t for keyword in ["system", "body", "anatomical"]):
                predictions.append("correct")
            
            # Heuristic 3: Process relations
            elif any(keyword in h for keyword in ["process", "function", "activity"]) and \
                 r in ["occurs_in", "affects", "performs"] and \
                 any(keyword in t for keyword in ["organism", "cell", "tissue"]):
                predictions.append("correct")
            
            # Default: random guess weighted by relation frequency
            else:
                predictions.append(random.choice(["correct", "incorrect"]))
        
        return predictions
    
    def random_classifier(self, dataset: List[Dict]) -> List[str]:
        """Random baseline classifier."""
        return [random.choice(["correct", "incorrect"]) for _ in dataset]
    
    def evaluate(self, true_labels: List[str], predictions: List[str]) -> Dict:
        """Evaluate predictions against true labels."""
        pred_binary = [1 if pred == "correct" else 0 for pred in predictions]
        true_binary = [1 if true == "correct" else 0 for true in true_labels]
        
        f1 = f1_score(true_binary, pred_binary, average='macro')
        report = classification_report(true_binary, pred_binary, 
                                     target_names=['incorrect', 'correct'], 
                                     output_dict=True)
        
        return {
            "f1_score": f1,
            "classification_report": report,
            "accuracy": report['accuracy']
        }
    
    def analyze_relations(self, dataset: List[Dict]) -> Dict:
        """Analyze the distribution of relations in the dataset."""
        relation_counts = defaultdict(int)
        label_counts = defaultdict(int)
        
        for item in dataset:
            relation_counts[item["r"]] += 1
            label_counts[item["label"]] += 1
        
        return {
            "total_examples": len(dataset),
            "unique_relations": len(relation_counts),
            "relation_distribution": dict(relation_counts),
            "label_distribution": dict(label_counts),
            "most_common_relations": sorted(relation_counts.items(), 
                                          key=lambda x: x[1], reverse=True)[:10]
        }
    
    def run_experiment(self, template_idx: int = 0, classifier_type: str = "heuristic", 
                      test_size: float = 0.3) -> Dict:
        """Run complete relation extraction experiment."""
        print("Building relation dataset...")
        dataset = self.build_relation_dataset()
        print(f"Created {len(dataset)} examples")
        
        # Analyze dataset
        analysis = self.analyze_relations(dataset)
        print(f"Dataset contains {analysis['unique_relations']} unique relations")
        
        # Split into train/test
        split_idx = int(len(dataset) * (1 - test_size))
        train_data = dataset[:split_idx]
        test_data = dataset[split_idx:]
        
        print(f"Train: {len(train_data)}, Test: {len(test_data)}")
        
        # Create prompts
        test_prompts = self.create_prompts(test_data, template_idx)
        
        # Get predictions
        print("Generating predictions...")
        if classifier_type == "heuristic":
            predictions = self.simple_relation_classifier(test_data)
        else:
            predictions = self.random_classifier(test_data)
        
        true_labels = [item["label"] for item in test_data]
        
        # Evaluate
        results = self.evaluate(true_labels, predictions)
        
        return {
            "template_used": self.templates[template_idx],
            "classifier_type": classifier_type,
            "dataset_size": len(dataset),
            "test_size": len(test_data),
            "dataset_analysis": analysis,
            "results": results,
            "sample_prompts": test_prompts[:5],
            "sample_predictions": list(zip(true_labels[:5], predictions[:5]))
        }


def main():
    """Main function to run relation extraction."""
    # Initialize with data paths
    hierarchy_path = "/home/saba/university/S&DAI/LLMs4OL/data/umls_hierarchy.json"
    classes_path = "/home/saba/university/S&DAI/LLMs4OL/data/umls_classes.json"
    relationships_path = "/home/saba/university/S&DAI/LLMs4OL/data/umls_relationship_samples.json"
    
    # Create relation extraction instance
    relation_extractor = UMLSRelationExtraction(hierarchy_path, classes_path, relationships_path)
    
    # Run experiment with different templates and classifiers
    classifiers = ["heuristic", "random"]
    
    for classifier_type in classifiers:
        print(f"\n{'='*80}")
        print(f"TESTING CLASSIFIER: {classifier_type.upper()}")
        print(f"{'='*80}")
        
        for template_idx in range(min(4, len(relation_extractor.templates))):  # Test first 4 templates
            print(f"\n{'='*60}")
            print(f"Template {template_idx + 1}: {relation_extractor.templates[template_idx]}")
            print(f"{'='*60}")
            
            results = relation_extractor.run_experiment(
                template_idx=template_idx,
                classifier_type=classifier_type
            )
            
            print(f"\nResults:")
            print(f"F1-Score: {results['results']['f1_score']:.4f}")
            print(f"Accuracy: {results['results']['accuracy']:.4f}")
            
            print(f"\nDataset Analysis:")
            analysis = results['dataset_analysis']
            print(f"Total examples: {analysis['total_examples']}")
            print(f"Unique relations: {analysis['unique_relations']}")
            print(f"Label distribution: {analysis['label_distribution']}")
            
            print(f"\nTop relations:")
            for rel, count in analysis['most_common_relations'][:5]:
                print(f"  {rel}: {count}")
            
            print(f"\nSample prompts:")
            for i, prompt in enumerate(results['sample_prompts'][:3]):
                print(f"{i+1}. {prompt}")
            
            print(f"\nSample predictions (true, predicted):")
            for true_label, pred_label in results['sample_predictions'][:3]:
                print(f"  True: {true_label}, Predicted: {pred_label}")


if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    main()
