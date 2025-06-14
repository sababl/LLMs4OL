#!/usr/bin/env python3
"""
UMLS Non-Taxonomic Relation Extraction - Task C Implementation
A simplified version for discovering semantic relationships between UMLS types (non "is-a").
"""

import argparse
import json
import random
from pathlib import Path
from taxonomy_utils import Config, DataUtils, EvaluationMetrics
from collections import defaultdict


class RelationConfig(Config):
    """Extended configuration for relation extraction."""
    
    def __init__(self):
        super().__init__()
        self.relationships_path = self.data_dir / "umls_relationship_samples.json"
        
        # Templates for relation extraction
        self.relation_templates = [
            "{h} {r} {t}. This statement is {answer}.",
            "The relationship '{r}' holds between {h} and {t}. This statement is {answer}.",
            "{h} has the relation '{r}' with {t}. This statement is {answer}.",
            "'{h}' {r} '{t}'. This statement is {answer}.",
            "In the semantic network, {h} {r} {t}. This statement is {answer}.",
            "The semantic relation '{r}' connects {h} to {t}. This statement is {answer}.",
            "{h} and {t} are related by '{r}'. This statement is {answer}.",
            "There exists a '{r}' relationship from {h} to {t}. This statement is {answer}."
        ]
        
        # Common UMLS semantic relations (non-taxonomic)
        self.semantic_relations = [
            "affects", "associated_with", "treats", "prevents", "diagnoses",
            "causes", "manifestation_of", "result_of", "location_of", "part_of",
            "connected_to", "adjacent_to", "surrounds", "traverses", "contains",
            "uses", "performs", "carries_out", "occurs_in", "produced_by",
            "exhibits", "indicates", "measures", "method_of", "process_of",
            "property_of", "conceptually_related_to", "functionally_related_to",
            "temporally_related_to", "spatially_related_to", "interacts_with"
        ]


def extract_concepts_from_hierarchy(hierarchy_data):
    """Extract all concept names from hierarchy."""
    concepts = set()
    for child, parents in hierarchy_data.items():
        concepts.add(DataUtils.normalize_concept_name(child))
        for parent in parents:
            concepts.add(DataUtils.normalize_concept_name(parent))
    return list(concepts)


def generate_semantic_relations(concepts, semantic_relations, num_relations=1000):
    """Generate synthetic semantic relations between concepts."""
    relations = []
    
    # Medical domain relation patterns
    medical_patterns = {
        "treats": (["disease", "disorder", "syndrome", "pathology"], 
                  ["substance", "therapy", "treatment", "drug"]),
        "causes": (["pathology", "disorder", "dysfunction"], 
                  ["disease", "syndrome", "abnormality"]),
        "prevents": (["substance", "therapy", "intervention"], 
                    ["disease", "disorder", "pathology"]),
        "location_of": (["structure", "organ", "component"], 
                       ["process", "function", "activity"]),
        "part_of": (["component", "element", "structure"], 
                   ["system", "whole", "entity"]),
        "affects": (["substance", "agent", "process"], 
                   ["function", "system", "organism"]),
        "uses": (["process", "activity", "organism"], 
                ["substance", "tool", "method"]),
        "occurs_in": (["process", "function", "activity"], 
                     ["location", "structure", "environment"])
    }
    
    for _ in range(num_relations):
        h = random.choice(concepts)
        t = random.choice(concepts)
        r = random.choice(semantic_relations)
        
        if h != t:  # Avoid self-relations
            relations.append({
                "h": h,
                "r": r,
                "t": t,
                "label": "correct"
            })
    
    return relations


def create_relation_dataset(hierarchy_data, relationships_data=None, max_examples=2000):
    """Create balanced dataset of relation examples."""
    config = RelationConfig()
    concepts = extract_concepts_from_hierarchy(hierarchy_data)
    
    # Load existing relationships or generate synthetic ones
    positive_relations = []
    
    if relationships_data:
        # Filter non-taxonomic relationships
        for rel in relationships_data:
            if (rel.get("relationship", "") != "isa" and 
                rel.get("label", "") == "positive"):
                positive_relations.append({
                    "h": DataUtils.normalize_concept_name(rel.get("child", "")),
                    "r": rel.get("relationship", "related_to"),
                    "t": DataUtils.normalize_concept_name(rel.get("parent", "")),
                    "label": "correct"
                })
    
    # Generate additional synthetic relations if needed
    target_positive = max_examples // 2
    if len(positive_relations) < target_positive:
        additional_needed = target_positive - len(positive_relations)
        synthetic_relations = generate_semantic_relations(
            concepts, config.semantic_relations, additional_needed
        )
        positive_relations.extend(synthetic_relations)
    
    # Limit positive examples
    if len(positive_relations) > target_positive:
        positive_relations = random.sample(positive_relations, target_positive)
    
    # Create negative examples
    positive_triples = {(rel["h"], rel["r"], rel["t"]) for rel in positive_relations}
    negative_relations = []
    
    attempts = 0
    max_attempts = target_positive * 10
    
    while len(negative_relations) < len(positive_relations) and attempts < max_attempts:
        h = random.choice(concepts)
        r = random.choice(config.semantic_relations)
        t = random.choice(concepts)
        
        if h != t and (h, r, t) not in positive_triples:
            negative_relations.append({
                "h": h,
                "r": r,
                "t": t,
                "label": "incorrect"
            })
        
        attempts += 1
    
    # Combine and shuffle
    dataset = positive_relations + negative_relations
    random.shuffle(dataset)
    return dataset


def heuristic_relation_classifier(test_data):
    """Heuristic-based classifier for semantic relations."""
    predictions = []
    
    # Domain-specific patterns
    medical_patterns = {
        ("disease", "disorder", "syndrome"): {
            "treats": ("substance", "therapy", "treatment", "drug"),
            "causes": ("pathology", "dysfunction", "abnormality"),
            "prevents": ("intervention", "vaccine", "therapy")
        },
        ("organ", "structure", "component"): {
            "part_of": ("system", "body", "organism"),
            "location_of": ("process", "function"),
            "connected_to": ("structure", "component")
        },
        ("process", "function", "activity"): {
            "occurs_in": ("organism", "cell", "tissue"),
            "affects": ("system", "function"),
            "uses": ("substance", "energy")
        }
    }
    
    for item in test_data:
        h, r, t = item["h"], item["r"], item["t"]
        prediction = "incorrect"  # Default
        
        # Check medical patterns
        for h_patterns, relations in medical_patterns.items():
            if any(pattern in h for pattern in h_patterns):
                if r in relations:
                    t_patterns = relations[r]
                    if any(pattern in t for pattern in t_patterns):
                        prediction = "correct"
                        break
        
        # Additional heuristics
        if prediction == "incorrect":
            # Common semantic patterns
            if r == "associated_with":
                prediction = random.choices(["correct", "incorrect"], weights=[0.6, 0.4])[0]
            elif r in ["interacts_with", "related_to"]:
                prediction = random.choices(["correct", "incorrect"], weights=[0.7, 0.3])[0]
            else:
                prediction = random.choice(["correct", "incorrect"])
        
        predictions.append(prediction)
    
    return predictions


def analyze_relation_dataset(dataset):
    """Analyze the distribution of relations in the dataset."""
    relation_counts = defaultdict(int)
    label_counts = defaultdict(int)
    concept_counts = defaultdict(int)
    
    for item in dataset:
        relation_counts[item["r"]] += 1
        label_counts[item["label"]] += 1
        concept_counts[item["h"]] += 1
        concept_counts[item["t"]] += 1
    
    return {
        "total_examples": len(dataset),
        "unique_relations": len(relation_counts),
        "unique_concepts": len(concept_counts),
        "relation_distribution": dict(relation_counts),
        "label_distribution": dict(label_counts),
        "most_common_relations": sorted(relation_counts.items(), 
                                      key=lambda x: x[1], reverse=True)[:10],
        "most_common_concepts": sorted(concept_counts.items(), 
                                     key=lambda x: x[1], reverse=True)[:10]
    }


def run_relation_experiment(template_idx=0, classifier_type="heuristic", test_ratio=0.3):
    """Run relation extraction experiment."""
    config = RelationConfig()
    
    print(f"Loading UMLS data...")
    hierarchy_data = DataUtils.load_json(config.hierarchy_path)
    
    # Try to load relationship data
    relationships_data = []
    try:
        relationships_data = DataUtils.load_json(config.relationships_path)
        print(f"Loaded {len(relationships_data)} relationship samples")
    except FileNotFoundError:
        print("No relationship samples found, will generate synthetic data")
    
    print(f"Creating relation dataset...")
    dataset = create_relation_dataset(hierarchy_data, relationships_data)
    print(f"Dataset size: {len(dataset)}")
    
    # Analyze dataset
    analysis = analyze_relation_dataset(dataset)
    
    # Split dataset
    split_idx = int(len(dataset) * (1 - test_ratio))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Get template
    template = config.relation_templates[template_idx]
    print(f"Using template: {template}")
    
    # Generate predictions
    if classifier_type == "heuristic":
        predictions = heuristic_relation_classifier(test_data)
    else:  # random
        predictions = [random.choice(["correct", "incorrect"]) for _ in test_data]
    
    # Get true labels
    true_labels = [item["label"] for item in test_data]
    
    # Evaluate
    f1 = EvaluationMetrics.calculate_f1(true_labels, predictions)
    accuracy = EvaluationMetrics.calculate_accuracy(true_labels, predictions)
    detailed_report = EvaluationMetrics.detailed_report(true_labels, predictions)
    
    # Create sample prompts
    sample_prompts = []
    for item in test_data[:5]:
        prompt = template.format(
            h=item["h"],
            r=item["r"],
            t=item["t"],
            answer="[MASK]"
        )
        sample_prompts.append(prompt)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Template {template_idx + 1}: {template}")
    print(f"Classifier: {classifier_type}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    print(f"\nDataset Analysis:")
    print(f"Total examples: {analysis['total_examples']}")
    print(f"Unique relations: {analysis['unique_relations']}")
    print(f"Unique concepts: {analysis['unique_concepts']}")
    print(f"Label distribution: {analysis['label_distribution']}")
    
    print(f"\nTop 5 relations:")
    for rel, count in analysis['most_common_relations'][:5]:
        print(f"  {rel}: {count}")
    
    print(f"\nSample test examples:")
    for i, (prompt, pred) in enumerate(zip(sample_prompts[:3], predictions[:3])):
        print(f"{i+1}. {prompt}")
        print(f"   True: {true_labels[i]}, Predicted: {pred}")
        print()
    
    return {
        "template_idx": template_idx,
        "classifier_type": classifier_type,
        "f1_score": f1,
        "accuracy": accuracy,
        "detailed_report": detailed_report,
        "dataset_analysis": analysis,
        "dataset_size": len(dataset),
        "test_size": len(test_data)
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="UMLS Non-Taxonomic Relation Extraction - Task C")
    parser.add_argument("--template", type=int, default=0, 
                       help="Template index (0-7)")
    parser.add_argument("--classifier", choices=["heuristic", "random"], 
                       default="heuristic", help="Classifier type")
    parser.add_argument("--test_ratio", type=float, default=0.3, 
                       help="Test set ratio")
    parser.add_argument("--all_templates", action="store_true", 
                       help="Run with all templates")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    if args.all_templates:
        config = RelationConfig()
        all_results = []
        
        print("Running experiments with all relation templates...")
        for template_idx in range(len(config.relation_templates)):
            print(f"\n{'='*80}")
            print(f"TEMPLATE {template_idx + 1}/{len(config.relation_templates)}")
            print(f"{'='*80}")
            
            result = run_relation_experiment(
                template_idx=template_idx,
                classifier_type=args.classifier,
                test_ratio=args.test_ratio
            )
            all_results.append(result)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY OF ALL TEMPLATES")
        print(f"{'='*80}")
        
        avg_f1 = sum(r["f1_score"] for r in all_results) / len(all_results)
        avg_acc = sum(r["accuracy"] for r in all_results) / len(all_results)
        
        print(f"Average F1-Score: {avg_f1:.4f}")
        print(f"Average Accuracy: {avg_acc:.4f}")
        
        print(f"\nPer-template results:")
        for i, result in enumerate(all_results):
            print(f"Template {i+1}: F1={result['f1_score']:.4f}, Acc={result['accuracy']:.4f}")
        
        # Save results
        DataUtils.save_json(all_results, "relation_extraction_results.json")
        print(f"\nResults saved to relation_extraction_results.json")
        
    else:
        run_relation_experiment(
            template_idx=args.template,
            classifier_type=args.classifier,
            test_ratio=args.test_ratio
        )


if __name__ == "__main__":
    main()
