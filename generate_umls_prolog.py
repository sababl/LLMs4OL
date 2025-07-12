#!/usr/bin/env python3
"""
UMLS to Prolog Ontology Generator
Automatically generates a comprehensive Prolog representation of the UMLS ontology
from the JSON data files used in Task B and Task C implementations.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re


class UMLSPrologGenerator:
    def __init__(self, data_dir: str = "data"):
        """Initialize with data directory path."""
        self.data_dir = Path(data_dir)
        self.hierarchy_path = self.data_dir / "umls_dataset" / "umls_hierarchy.json"
        self.classes_path = self.data_dir / "umls_dataset" / "umls_classes.json"
        self.relationships_path = self.data_dir / "umls_relationship_samples.json"
        
        # Load data
        self.hierarchy = self._load_json(self.hierarchy_path)
        self.classes = self._load_json(self.classes_path)
        self.relationships = self._load_json(self.relationships_path, default=[])
        
        # Prolog generation statistics
        self.stats = {
            "concepts": 0,
            "hierarchical_relations": 0,
            "semantic_relations": 0,
            "definitions": 0,
            "properties": 0
        }
    
    def _load_json(self, file_path: Path, default=None):
        """Load JSON file with error handling."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
            return default if default is not None else {}
    
    def _normalize_atom(self, text: str) -> str:
        """Normalize text for use as Prolog atom."""
        if not text:
            return "unknown"
        
        # Convert to lowercase and replace spaces/special chars with underscores
        normalized = re.sub(r'[^a-zA-Z0-9_]', '_', text.lower())
        # Remove multiple consecutive underscores
        normalized = re.sub(r'_+', '_', normalized)
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        
        # Ensure it doesn't start with a number
        if normalized and normalized[0].isdigit():
            normalized = f"concept_{normalized}"
        
        return normalized if normalized else "unknown"
    
    def _escape_string(self, text: str) -> str:
        """Escape string for Prolog representation."""
        if not text:
            return "''"
        
        # Escape single quotes and wrap in quotes
        escaped = text.replace("'", "''")
        return f"'{escaped}'"
    
    def generate_concept_facts(self) -> list:
        """Generate concept/semantic type facts."""
        facts = []
        facts.append("% UMLS Semantic Type Concepts")
        facts.append("% concept(ConceptName, UI, TreeNumber).")
        facts.append("")
        
        # Generate from classes data
        for concept_name, concept_data in self.classes.items():
            atom_name = self._normalize_atom(concept_name)
            ui = concept_data.get("ui", "")
            tree_number = concept_data.get("tree_number", "")
            
            fact = f"concept({atom_name}, '{ui}', '{tree_number}')."
            facts.append(fact)
            self.stats["concepts"] += 1
        
        # Also add concepts from hierarchy that might not be in classes
        all_concepts = set()
        for child, parents in self.hierarchy.items():
            all_concepts.add(child)
            all_concepts.update(parents)
        
        for concept in all_concepts:
            atom_name = self._normalize_atom(concept)
            if concept not in self.classes:
                fact = f"concept({atom_name}, '', '')."
                facts.append(fact)
                self.stats["concepts"] += 1
        
        facts.append("")
        return facts
    
    def generate_definition_facts(self) -> list:
        """Generate concept definition facts."""
        facts = []
        facts.append("% UMLS Concept Definitions")
        facts.append("% definition(ConceptName, Definition).")
        facts.append("")
        
        for concept_name, concept_data in self.classes.items():
            definition = concept_data.get("definition", "")
            if definition:
                atom_name = self._normalize_atom(concept_name)
                escaped_def = self._escape_string(definition)
                fact = f"definition({atom_name}, {escaped_def})."
                facts.append(fact)
                self.stats["definitions"] += 1
        
        facts.append("")
        return facts
    
    def generate_hierarchy_facts(self) -> list:
        """Generate hierarchical (is-a) relationship facts."""
        facts = []
        facts.append("% UMLS Hierarchical Relationships (is-a)")
        facts.append("% isa(Child, Parent) - Child is a subclass of Parent.")
        facts.append("")
        
        for child, parents in self.hierarchy.items():
            child_atom = self._normalize_atom(child)
            for parent in parents:
                parent_atom = self._normalize_atom(parent)
                fact = f"isa({child_atom}, {parent_atom})."
                facts.append(fact)
                self.stats["hierarchical_relations"] += 1
        
        facts.append("")
        return facts
    
    def generate_semantic_relation_facts(self) -> list:
        """Generate semantic (non-is-a) relationship facts."""
        facts = []
        facts.append("% UMLS Semantic Relationships (non-is-a)")
        facts.append("% semantic_relation(Subject, Relation, Object).")
        facts.append("")
        
        # From relationship samples
        for rel_data in self.relationships:
            if rel_data.get("relationship", "") != "isa" and rel_data.get("label", "") == "positive":
                subject = self._normalize_atom(rel_data.get("child", ""))
                relation = self._normalize_atom(rel_data.get("relationship", ""))
                object_name = self._normalize_atom(rel_data.get("parent", ""))
                
                fact = f"semantic_relation({subject}, {relation}, {object_name})."
                facts.append(fact)
                self.stats["semantic_relations"] += 1
        
        facts.append("")
        return facts
    
    def generate_property_facts(self) -> list:
        """Generate additional property facts."""
        facts = []
        facts.append("% UMLS Concept Properties")
        facts.append("% has_ui(ConceptName, UI).")
        facts.append("% has_tree_number(ConceptName, TreeNumber).")
        facts.append("")
        
        for concept_name, concept_data in self.classes.items():
            atom_name = self._normalize_atom(concept_name)
            ui = concept_data.get("ui", "")
            tree_number = concept_data.get("tree_number", "")
            
            if ui:
                fact = f"has_ui({atom_name}, '{ui}')."
                facts.append(fact)
                self.stats["properties"] += 1
            
            if tree_number:
                fact = f"has_tree_number({atom_name}, '{tree_number}')."
                facts.append(fact)
                self.stats["properties"] += 1
        
        facts.append("")
        return facts
    
    def generate_rules(self) -> list:
        """Generate useful Prolog rules for querying the ontology."""
        rules = []
        rules.append("% UMLS Ontology Rules")
        rules.append("% Useful rules for querying the ontology")
        rules.append("")
        
        # Transitive closure for is-a relationships
        rules.append("% Transitive closure of is-a relationships")
        rules.append("% subclass(X, Y) - X is a subclass of Y (direct or indirect)")
        rules.append("subclass(X, Y) :- isa(X, Y).")
        rules.append("subclass(X, Z) :- isa(X, Y), subclass(Y, Z).")
        rules.append("")
        
        # Sibling relationships
        rules.append("% Sibling relationships")
        rules.append("% sibling(X, Y) - X and Y share a common parent")
        rules.append("sibling(X, Y) :- isa(X, Z), isa(Y, Z), X \\= Y.")
        rules.append("")
        
        # Ancestor relationships
        rules.append("% Ancestor relationships")
        rules.append("% ancestor(X, Y) - X is an ancestor of Y")
        rules.append("ancestor(X, Y) :- isa(Y, X).")
        rules.append("ancestor(X, Z) :- isa(Y, X), ancestor(Y, Z).")
        rules.append("")
        
        # Descendant relationships
        rules.append("% Descendant relationships")
        rules.append("% descendant(X, Y) - X is a descendant of Y")
        rules.append("descendant(X, Y) :- ancestor(Y, X).")
        rules.append("")
        
        # Common ancestor
        rules.append("% Common ancestor")
        rules.append("% common_ancestor(X, Y, Z) - Z is a common ancestor of X and Y")
        rules.append("common_ancestor(X, Y, Z) :- ancestor(Z, X), ancestor(Z, Y), X \\= Y.")
        rules.append("")
        
        # Related concepts (direct semantic relations)
        rules.append("% Related concepts")
        rules.append("% related(X, Y, R) - X is related to Y by relation R")
        rules.append("related(X, Y, R) :- semantic_relation(X, R, Y).")
        rules.append("related(X, Y, R) :- semantic_relation(Y, R, X).")
        rules.append("")
        
        # Concept with definition
        rules.append("% Concepts with definitions")
        rules.append("% has_definition(X) - X has a definition")
        rules.append("has_definition(X) :- definition(X, _).")
        rules.append("")
        
        return rules
    
    def generate_queries(self) -> list:
        """Generate example queries for the ontology."""
        queries = []
        queries.append("% Example Queries")
        queries.append("% Use these queries to explore the UMLS ontology")
        queries.append("")
        
        queries.append("% Find all concepts")
        queries.append("% ?- concept(X, _, _).")
        queries.append("")
        
        queries.append("% Find all subclasses of a concept")
        queries.append("% ?- subclass(X, organism).")
        queries.append("")
        
        queries.append("% Find all ancestors of a concept")
        queries.append("% ?- ancestor(X, plant).")
        queries.append("")
        
        queries.append("% Find siblings of a concept")
        queries.append("% ?- sibling(plant, X).")
        queries.append("")
        
        queries.append("% Find concepts with definitions")
        queries.append("% ?- has_definition(X), definition(X, Def).")
        queries.append("")
        
        queries.append("% Find semantic relations involving a concept")
        queries.append("% ?- related(organism, Y, R).")
        queries.append("")
        
        queries.append("% Find common ancestors")
        queries.append("% ?- common_ancestor(plant, animal, X).")
        queries.append("")
        
        return queries
    
    def generate_prolog_file(self, output_path: str = "prolog/umls_ontology.pl") -> str:
        """Generate complete Prolog file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        # Generate all sections
        header = self._generate_header()
        concepts = self.generate_concept_facts()
        definitions = self.generate_definition_facts()
        hierarchy = self.generate_hierarchy_facts()
        semantics = self.generate_semantic_relation_facts()
        properties = self.generate_property_facts()
        rules = self.generate_rules()
        queries = self.generate_queries()
        footer = self._generate_footer()
        
        # Combine all sections
        prolog_content = []
        prolog_content.extend(header)
        prolog_content.extend(concepts)
        prolog_content.extend(definitions)
        prolog_content.extend(hierarchy)
        prolog_content.extend(semantics)
        prolog_content.extend(properties)
        prolog_content.extend(rules)
        prolog_content.extend(queries)
        prolog_content.extend(footer)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(prolog_content))
        
        return str(output_file)
    
    def _generate_header(self) -> list:
        """Generate file header."""
        header = []
        header.append("% UMLS Ontology in Prolog")
        header.append("% Automatically generated from UMLS JSON data")
        header.append(f"% Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        header.append("%")
        header.append("% This file contains:")
        header.append("% - UMLS semantic type concepts")
        header.append("% - Concept definitions")
        header.append("% - Hierarchical (is-a) relationships")
        header.append("% - Semantic (non-is-a) relationships")
        header.append("% - Useful rules for querying")
        header.append("% - Example queries")
        header.append("")
        header.append(":- encoding(utf8).")
        header.append("")
        return header
    
    def _generate_footer(self) -> list:
        """Generate file footer with statistics."""
        footer = []
        footer.append("")
        footer.append("% Generation Statistics")
        footer.append(f"% Concepts: {self.stats['concepts']}")
        footer.append(f"% Hierarchical relations: {self.stats['hierarchical_relations']}")
        footer.append(f"% Semantic relations: {self.stats['semantic_relations']}")
        footer.append(f"% Definitions: {self.stats['definitions']}")
        footer.append(f"% Properties: {self.stats['properties']}")
        footer.append("")
        footer.append("% End of UMLS Ontology")
        return footer
    
    def print_statistics(self):
        """Print generation statistics."""
        print("\n" + "="*60)
        print("UMLS PROLOG GENERATION STATISTICS")
        print("="*60)
        print(f"Concepts generated: {self.stats['concepts']}")
        print(f"Hierarchical relations: {self.stats['hierarchical_relations']}")
        print(f"Semantic relations: {self.stats['semantic_relations']}")
        print(f"Definitions: {self.stats['definitions']}")
        print(f"Properties: {self.stats['properties']}")
        print(f"Total facts: {sum(self.stats.values())}")
        print("="*60)


def main():
    """Main function to generate Prolog ontology."""
    parser = argparse.ArgumentParser(description="Generate UMLS Prolog Ontology")
    parser.add_argument("--data_dir", default="data", 
                       help="Directory containing UMLS JSON files")
    parser.add_argument("--output", default="prolog/umls_ontology.pl",
                       help="Output Prolog file path")
    parser.add_argument("--stats", action="store_true",
                       help="Show generation statistics")
    
    args = parser.parse_args()
    
    print("UMLS to Prolog Ontology Generator")
    print("Generating comprehensive Prolog representation...")
    
    # Check if data files exist
    data_dir = Path(args.data_dir)
    required_files = ["umls_dataset/umls_hierarchy.json", "umls_dataset/umls_classes.json"]
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    
    if missing_files:
        print(f"ERROR: Missing required files: {missing_files}")
        print(f"Please ensure files exist in {data_dir}")
        return
    
    # Generate Prolog ontology
    generator = UMLSPrologGenerator(args.data_dir)
    output_file = generator.generate_prolog_file(args.output)
    
    print(f"SUCCESS: Prolog ontology generated: {output_file}")
    
    if args.stats:
        generator.print_statistics()
    
    # Show some example usage
    print(f"\nTo use the generated ontology:")
    print(f"1. Load in SWI-Prolog: ?- consult('{output_file}').")
    print(f"2. Query concepts: ?- concept(X, _, _).")
    print(f"3. Find hierarchies: ?- subclass(X, organism).")
    print(f"4. Check the file for more example queries!")


if __name__ == "__main__":
    main()
