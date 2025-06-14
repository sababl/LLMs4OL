#!/usr/bin/env python3
"""
UMLS Taxonomy Discovery - Task B Implementation
A simplified version for discovering "is-a" hierarchies in UMLS semantic types.
"""

import argparse
import json
import random
from pathlib import Path
from taxonomy_utils import Config, DataUtils, EvaluationMetrics


def create_dataset(hierarchy_data, max_examples=2000):
    """Create balanced dataset of positive and negative taxonomy examples."""
    config = Config()
    
    # Build transitive closure for positive examples
    positive_pairs = DataUtils.build_transitive_closure(hierarchy_data)
    positive_pairs = list(positive_pairs)
    
    # Limit positive examples
    if len(positive_pairs) > max_examples // 2:
        positive_pairs = random.sample(positive_pairs, max_examples // 2)
    
    # Get all unique concepts
    all_concepts = set()
    for parent, child in positive_pairs:
        all_concepts.add(parent)
        all_concepts.add(child)
    all_concepts = list(all_concepts)
    
    # Create negative examples
    positive_set = set(positive_pairs)
    negative_pairs = []
    
    while len(negative_pairs) < len(positive_pairs):
        concept1 = random.choice(all_concepts)
        concept2 = random.choice(all_concepts)
        
        if (concept1, concept2) not in positive_set and concept1 != concept2:
            negative_pairs.append((concept1, concept2))
    
    # Build final dataset
    dataset = []
    
    # Add positive examples
    for parent, child in positive_pairs:
        dataset.append({
            "text_a": parent,
            "text_b": child,
            "label": "correct"
        })
    
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


def simple_baseline_classifier(test_data, hierarchy_data):
    """Simple baseline that uses hierarchy lookup."""
    hierarchy_pairs = DataUtils.build_transitive_closure(hierarchy_data)
    predictions = []
    
    for item in test_data:
        parent = item["text_a"]
        child = item["text_b"]
        
        if (parent, child) in hierarchy_pairs:
            predictions.append("correct")
        else:
            predictions.append("incorrect")
    
    return predictions


def random_baseline_classifier(test_data):
    """Random baseline classifier."""
    return [random.choice(["correct", "incorrect"]) for _ in test_data]


def run_experiment(template_idx=0, classifier_type="hierarchy", test_ratio=0.3):
    """Run taxonomy discovery experiment."""
    config = Config()
    
    print(f"Loading UMLS data...")
    hierarchy_data = DataUtils.load_json(config.hierarchy_path)
    
    print(f"Creating dataset...")
    dataset = create_dataset(hierarchy_data)
    print(f"Dataset size: {len(dataset)}")
    
    # Split dataset
    split_idx = int(len(dataset) * (1 - test_ratio))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Get template
    template = config.templates[template_idx]
    print(f"Using template: {template}")
    
    # Generate predictions
    if classifier_type == "hierarchy":
        predictions = simple_baseline_classifier(test_data, hierarchy_data)
    else:
        predictions = random_baseline_classifier(test_data)
    
    # Get true labels
    true_labels = [item["label"] for item in test_data]
    
    # Evaluate
    f1 = EvaluationMetrics.calculate_f1(true_labels, predictions)
    accuracy = EvaluationMetrics.calculate_accuracy(true_labels, predictions)
    detailed_report = EvaluationMetrics.detailed_report(true_labels, predictions)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Template {template_idx + 1}: {template}")
    print(f"Classifier: {classifier_type}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Show some examples
    print(f"\nSample test examples:")
    for i, (item, pred) in enumerate(zip(test_data[:5], predictions[:5])):
        formatted_prompt = template.format(
            text_a=item["text_a"],
            text_b=item["text_b"],
            answer="[MASK]"
        )
        print(f"{i+1}. {formatted_prompt}")
        print(f"   True: {item['label']}, Predicted: {pred}")
        print()
    
    return {
        "template_idx": template_idx,
        "classifier_type": classifier_type,
        "f1_score": f1,
        "accuracy": accuracy,
        "detailed_report": detailed_report,
        "dataset_size": len(dataset),
        "test_size": len(test_data)
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="UMLS Taxonomy Discovery - Task B")
    parser.add_argument("--template", type=int, default=0, 
                       help="Template index (0-7)")
    parser.add_argument("--classifier", choices=["hierarchy", "random"], 
                       default="hierarchy", help="Classifier type")
    parser.add_argument("--test_ratio", type=float, default=0.3, 
                       help="Test set ratio")
    parser.add_argument("--all_templates", action="store_true", 
                       help="Run with all templates")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    if args.all_templates:
        config = Config()
        all_results = []
        
        print("Running experiments with all templates...")
        for template_idx in range(len(config.templates)):
            print(f"\n{'='*80}")
            print(f"TEMPLATE {template_idx + 1}/{len(config.templates)}")
            print(f"{'='*80}")
            
            result = run_experiment(
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
        DataUtils.save_json(all_results, "taxonomy_discovery_results.json")
        print(f"\nResults saved to taxonomy_discovery_results.json")
        
    else:
        run_experiment(
            template_idx=args.template,
            classifier_type=args.classifier,
            test_ratio=args.test_ratio
        )


if __name__ == "__main__":
    main()
