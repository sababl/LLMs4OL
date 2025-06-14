#!/usr/bin/env python3
"""
UMLS Data Utilities

Consolidated utilities for UMLS data management, dataset creation, and processing.
This module replaces multiple build scripts with a single, clean interface.
"""

import os
import json
import csv
import random
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class UMLSDataManager:
    """Manager class for UMLS data operations."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data manager."""
        self.data_dir = data_dir
        self.ensure_data_dir()
    
    def ensure_data_dir(self):
        """Ensure data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def create_sample_umls_data(self):
        """Create sample UMLS data for development/testing purposes."""
        print("Creating sample UMLS data...")
        
        # Sample semantic types based on real UMLS structure
        sample_classes = {
            "Organism": {
                "ui": "T001",
                "name": "Organism", 
                "definition": "Generally, a living individual, including all plants and animals.",
                "parents": ["Physical Object"],
                "tree_number": "A1.1"
            },
            "Animal": {
                "ui": "T008",
                "name": "Animal",
                "definition": "A living organism characterized by voluntary movement.",
                "parents": ["Organism"],
                "tree_number": "A1.1.1"
            },
            "Human": {
                "ui": "T016", 
                "name": "Human",
                "definition": "Modern man, the only living species of the genus Homo.",
                "parents": ["Mammal"],
                "tree_number": "A1.1.1.2"
            },
            "Anatomical Structure": {
                "ui": "T017",
                "name": "Anatomical Structure",
                "definition": "A normal or abnormal part of the anatomy or structural organization of an organism.",
                "parents": ["Physical Object"],
                "tree_number": "A1.2"
            },
            "Body Part, Organ, or Organ Component": {
                "ui": "T023",
                "name": "Body Part, Organ, or Organ Component", 
                "definition": "Anatomical structure which is a subdivision of a whole organism.",
                "parents": ["Anatomical Structure"],
                "tree_number": "A1.2.1"
            },
            "Disease or Syndrome": {
                "ui": "T047",
                "name": "Disease or Syndrome",
                "definition": "A condition which alters or interferes with a normal process.",
                "parents": ["Pathologic Function"],
                "tree_number": "B2.2.1"
            },
            "Pharmacologic Substance": {
                "ui": "T121",
                "name": "Pharmacologic Substance", 
                "definition": "A substance used in the treatment or prevention of disease.",
                "parents": ["Chemical Viewed Functionally"],
                "tree_number": "D1.2.1"
            },
            "Therapeutic or Preventive Procedure": {
                "ui": "T061",
                "name": "Therapeutic or Preventive Procedure",
                "definition": "A procedure, method, or technique designed to prevent disease.",
                "parents": ["Health Care Activity"],
                "tree_number": "E1.1.1"
            },
            "Sign or Symptom": {
                "ui": "T184",
                "name": "Sign or Symptom",
                "definition": "An observable manifestation of a disease or condition.",
                "parents": ["Finding"],
                "tree_number": "B2.1.1"
            },
            "Chemical": {
                "ui": "T103",
                "name": "Chemical",
                "definition": "A substance with a defined molecular composition.",
                "parents": ["Physical Object"],
                "tree_number": "D1"
            }
        }
        
        # Sample terms for each semantic type
        sample_terms = [
            "heart", "brain", "liver", "kidney", "lung", "muscle", "bone", "blood",
            "diabetes", "cancer", "pneumonia", "arthritis", "hypertension", "asthma",
            "aspirin", "insulin", "penicillin", "morphine", "warfarin", "metformin",
            "surgery", "biopsy", "vaccination", "chemotherapy", "treatment", "therapy",
            "fever", "pain", "nausea", "fatigue", "headache", "cough",
            "bacteria", "virus", "fungus", "parasite", "human", "animal",
            "sodium", "glucose", "protein", "vitamin", "hormone", "enzyme"
        ]
        
        # Create hierarchy relationships
        sample_hierarchy = {
            "Animal": ["Organism"],
            "Human": ["Animal", "Organism"],
            "Body Part, Organ, or Organ Component": ["Anatomical Structure"],
            "Sign or Symptom": ["Finding"],
            "Pharmacologic Substance": ["Chemical"]
        }
        
        # Save the data
        self.save_umls_classes(sample_classes)
        self.save_umls_terms(sample_terms)
        self.save_umls_hierarchy(sample_hierarchy)
        
        print(f"✓ Sample UMLS data created in {self.data_dir}/")
        print(f"  - {len(sample_classes)} semantic types")
        print(f"  - {len(sample_terms)} terms")
        print(f"  - {len(sample_hierarchy)} hierarchy relationships")
    
    def save_umls_classes(self, classes: Dict):
        """Save UMLS classes to JSON file."""
        file_path = os.path.join(self.data_dir, "umls_classes.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(classes, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved UMLS classes to {file_path}")
    
    def save_umls_terms(self, terms: List[str]):
        """Save UMLS terms to text file."""
        file_path = os.path.join(self.data_dir, "umls_terms.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            for term in terms:
                f.write(f"{term}\n")
        logger.info(f"Saved UMLS terms to {file_path}")
    
    def save_umls_hierarchy(self, hierarchy: Dict):
        """Save UMLS hierarchy to JSON file."""
        file_path = os.path.join(self.data_dir, "umls_hierarchy.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(hierarchy, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved UMLS hierarchy to {file_path}")
    
    def load_umls_data(self) -> Tuple[Dict, List[str], Dict]:
        """Load all UMLS data."""
        # Load classes
        classes_path = os.path.join(self.data_dir, "umls_classes.json")
        with open(classes_path, 'r', encoding='utf-8') as f:
            classes = json.load(f)
        
        # Load terms
        terms_path = os.path.join(self.data_dir, "umls_terms.txt")
        with open(terms_path, 'r', encoding='utf-8') as f:
            terms = [line.strip() for line in f if line.strip()]
        
        # Load hierarchy
        hierarchy_path = os.path.join(self.data_dir, "umls_hierarchy.json")
        with open(hierarchy_path, 'r', encoding='utf-8') as f:
            hierarchy = json.load(f)
        
        return classes, terms, hierarchy
    
    def validate_data_files(self) -> bool:
        """Validate that all required UMLS data files exist and are valid."""
        required_files = [
            ("umls_classes.json", "JSON"),
            ("umls_terms.txt", "text"),
            ("umls_hierarchy.json", "JSON")
        ]
        
        all_valid = True
        
        for filename, file_type in required_files:
            file_path = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(file_path):
                print(f"❌ Missing file: {file_path}")
                all_valid = False
                continue
            
            try:
                if file_type == "JSON":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if not data:
                            print(f"⚠️  Empty JSON file: {file_path}")
                        else:
                            print(f"✓ Valid {file_type}: {file_path} ({len(data)} entries)")
                else:  # text file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        if not lines:
                            print(f"⚠️  Empty text file: {file_path}")
                        else:
                            print(f"✓ Valid {file_type}: {file_path} ({len(lines)} lines)")
                            
            except Exception as e:
                print(f"❌ Invalid {file_type} file {file_path}: {e}")
                all_valid = False
        
        return all_valid
    
    def create_llms4ol_compatible_dataset(self, output_dir: str = "data/LLMs4OL_UMLS"):
        """Create LLMs4OL compatible dataset structure."""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            classes, terms, hierarchy = self.load_umls_data()
            
            # Create label mapper
            label_mapper = {}
            for semantic_type, class_info in classes.items():
                labels = [
                    semantic_type,
                    class_info.get("name", semantic_type),
                    semantic_type.lower(),
                    semantic_type.replace(" ", "_").lower()
                ]
                
                # Add definition keywords if available
                if "definition" in class_info:
                    definition = class_info["definition"]
                    # Extract key words (simple approach)
                    key_words = [word for word in definition.split() 
                               if len(word) > 4 and word.isalpha()][:3]
                    labels.extend(key_words)
                
                # Remove duplicates and empty strings
                labels = list(set([label for label in labels if label and len(label) > 1]))
                label_mapper[semantic_type] = labels[:8]  # Limit to 8 labels
            
            # Create templates for entity typing
            templates = [
                "The medical term {term} is classified as {semantic_type}.",
                "{term} is a type of {semantic_type}.",
                "In medical terminology, {term} belongs to the category {semantic_type}.",
                "The UMLS semantic type of {term} is {semantic_type}.",
                "{term} is classified under {semantic_type} in the medical domain."
            ]
            
            # Create hierarchy structure
            hierarchy_structure = {
                "umls": {
                    "semantic_types": list(classes.keys()),
                    "relationships": hierarchy,
                    "total_types": len(classes),
                    "description": "UMLS Semantic Network hierarchy"
                }
            }
            
            # Save files
            files_to_save = [
                ("label_mapper.json", label_mapper),
                ("templates.json", {"templates": templates}),
                ("hierarchy.json", hierarchy_structure),
                ("umls_stats.json", {
                    "semantic_types_count": len(classes),
                    "terms_count": len(terms),
                    "hierarchy_relations": len(hierarchy),
                    "created_by": "UMLS Data Manager"
                })
            ]
            
            for filename, data in files_to_save:
                file_path = os.path.join(output_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"✓ Created {file_path}")
            
            # Create README
            readme_content = """# LLMs4OL Compatible UMLS Dataset

This directory contains UMLS data in LLMs4OL compatible format.

## Files:
- `label_mapper.json`: Semantic type label mappings
- `templates.json`: Prompt templates for entity typing
- `hierarchy.json`: Hierarchical relationships
- `umls_stats.json`: Dataset statistics

## Usage:
Load these files for LLMs4OL Task A (Entity Typing) with UMLS domain focus.
"""
            
            readme_path = os.path.join(output_dir, "README.md")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            print(f"\n LLMs4OL compatible dataset created in {output_dir}/")
            
        except Exception as e:
            print(f" Error creating LLMs4OL dataset: {e}")
    
    def get_data_statistics(self) -> Dict:
        """Get comprehensive statistics about the UMLS data."""
        try:
            classes, terms, hierarchy = self.load_umls_data()
            
            # Analyze semantic types by category
            type_categories = defaultdict(int)
            for semantic_type, class_info in classes.items():
                # Simple categorization based on keywords
                name_lower = semantic_type.lower()
                if any(word in name_lower for word in ['anatomical', 'body', 'organ', 'structure']):
                    type_categories['Anatomical'] += 1
                elif any(word in name_lower for word in ['disease', 'syndrome', 'disorder']):
                    type_categories['Disease'] += 1
                elif any(word in name_lower for word in ['pharmacologic', 'chemical', 'substance']):
                    type_categories['Chemical'] += 1
                elif any(word in name_lower for word in ['organism', 'animal', 'human']):
                    type_categories['Organism'] += 1
                elif any(word in name_lower for word in ['procedure', 'activity', 'treatment']):
                    type_categories['Procedure'] += 1
                else:
                    type_categories['Other'] += 1
            
            stats = {
                "total_semantic_types": len(classes),
                "total_terms": len(terms),
                "total_hierarchy_relations": len(hierarchy),
                "semantic_type_categories": dict(type_categories),
                "sample_semantic_types": list(classes.keys())[:10],
                "sample_terms": terms[:20],
                "hierarchy_depth": max(len(parents) for parents in hierarchy.values()) if hierarchy else 0
            }
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}


def main():
    """Main function for data management operations."""
    print("🗃️  UMLS Data Manager")
    print("=" * 30)
    
    manager = UMLSDataManager()
    
    # Check if data exists
    print("Checking existing data files...")
    if manager.validate_data_files():
        print("\n All data files are valid!")
        
        # Show statistics
        stats = manager.get_data_statistics()
        print(f"\nData Statistics:")
        print(f"• Semantic Types: {stats.get('total_semantic_types', 0)}")
        print(f"• Terms: {stats.get('total_terms', 0)}")
        print(f"• Hierarchy Relations: {stats.get('total_hierarchy_relations', 0)}")
        
        if 'semantic_type_categories' in stats:
            print(f"\nSemantic Type Categories:")
            for category, count in stats['semantic_type_categories'].items():
                print(f"  • {category}: {count}")
        
        # Create LLMs4OL compatible dataset
        print(f"\nCreating LLMs4OL compatible dataset...")
        manager.create_llms4ol_compatible_dataset()
        
    else:
        print("\n Some data files are missing or invalid.")
        print("Creating sample UMLS data for development...")
        manager.create_sample_umls_data()
        
        print("\nValidating created data...")
        if manager.validate_data_files():
            print("Sample data created successfully!")
        else:
            print("Failed to create valid sample data.")


if __name__ == "__main__":
    main()
