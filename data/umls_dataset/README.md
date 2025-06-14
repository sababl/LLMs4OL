# UMLS Dataset for Medical Term Classification

This dataset contains medical terminology and semantic types from the Unified Medical Language System (UMLS) for use with the term classification system.

## Dataset Overview

- **Domain**: Medical/Biomedical (UMLS)
- **Task**: Medical term classification and semantic typing
- **Total Semantic Types**: 127
- **Total Terms**: 297
- **Entity Typing Samples**: 297
- **Training Samples**: 594

## Files Description

### Core Data Files
- `umls_classes.json` - Complete UMLS semantic types with definitions and metadata
- `umls_terms.txt` - Medical terms extracted from UMLS
- `umls_hierarchy.json` - Hierarchical relationships between semantic types

### Dataset Files
- `umls_entity_typing.json` - Entity typing dataset with term-semantic type mappings
- `umls_training_data.json` - Training data for classification models
- `umls_semantic_types.json` - Comprehensive reference of semantic types
- `umls_templates.json` - Template prompts for classification tasks

### Metadata
- `dataset_stats.json` - Dataset statistics and metadata
- `README.md` - This documentation file

## Usage

### Basic Classification
```python
from src.term_typing import classify_term, load_umls_data

# Load data
umls_classes, umls_terms, umls_hierarchy = load_umls_data()
semantic_types = list(umls_classes.keys())

# Classify a medical term
result = classify_term("diabetes", semantic_types)
print(result)  # Should return: Disease or Syndrome
```

### Using the Entity Typing Dataset
```python
import json

# Load entity typing dataset
with open('data/umls_dataset/umls_entity_typing.json', 'r') as f:
    entity_data = json.load(f)

# Example entry
print(entity_data[0])
# {
#     "term": "diabetes",
#     "semantic_type": "Disease or Syndrome",
#     "tree_number": "B2.2.1.2.1",
#     "definition": "A condition which alters or interferes...",
#     "context": "Medical term from UMLS domain: diabetes"
# }
```

## Semantic Type Categories

The dataset includes semantic types across these categories:
- **Anatomy**: Body parts, organs, tissues, cells
- **Biology**: Organisms, genes, molecular structures  
- **Chemistry**: Chemical compounds, substances, drugs
- **Pathology**: Diseases, disorders, abnormalities
- **Procedures**: Therapeutic, diagnostic, laboratory procedures
- **Findings**: Clinical observations, symptoms, test results
- **Concepts**: Temporal, spatial, qualitative concepts
- **Organizations**: Healthcare organizations, professional groups

## Examples

### Common Medical Terms and Their Classifications:
- heart → Body Part, Organ, or Organ Component
- diabetes → Disease or Syndrome
- aspirin → Pharmacologic Substance  
- surgery → Therapeutic or Preventive Procedure
- fever → Sign or Symptom
- bacteria → Bacterium

## Integration

This dataset is designed to work with:
- The term classification system in `src/term_typing.py`
- Medical ontology learning experiments
- Healthcare AI applications
- Biomedical NLP research

## Data Sources

- **UMLS (Unified Medical Language System)**: Semantic types and hierarchies
- **Medical terminology**: Extracted from UMLS semantic network
- **Definitions**: From official UMLS documentation

## License

This dataset is derived from UMLS data which is freely available for research purposes.
Please cite UMLS in any research using this dataset.

## Updates

Version 1.0 - Initial release (2025-06-13)
- Complete UMLS semantic type coverage
- 297 entity typing samples
- 594 training samples
- Comprehensive documentation and examples
