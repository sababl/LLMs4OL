# LLMs4OL - Large Language Models for Ontology Learning (UMLS Domain)

A research project exploring the use of Large Language Models for ontology learning tasks in the medical domain, specifically focusing on UMLS (Unified Medical Language System) term classification and hierarchical relationship detection using Google's Gemini model.

## Overview

This project investigates how LLMs can be leveraged for medical ontology learning by:
- Classifying medical terms by their UMLS semantic types
- Detecting hierarchical relationships between medical concepts
- Evaluating model performance on medical taxonomic relationship identification
- Creating datasets compatible with the LLMs4OL framework

## Features

- **Medical Term Classification**: Automatically classify medical terms according to UMLS semantic types
- **Hierarchical Relationship Detection**: Identify superclass-subclass relationships between medical concepts
- **UMLS Integration**: Leverage UMLS semantic network for generating test datasets
- **Google Gemini Integration**: Use Google's Gemini model for medical language understanding tasks
- **LLMs4OL Compatibility**: Generate datasets in LLMs4OL format for standardized evaluation

## Setup

### Prerequisites

- Python 3.7+
- Google API key for Gemini model access
- Required Python packages (install via pip):
  ```bash
  pip install google-generativeai python-dotenv
  ```

### Configuration

1. Clone this repository
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Add your Google API key to the `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   GOOGLE_API_KEY=your_api_key_here
   MODEL_NAME=gemini-2.0-flash
   ```

### UMLS Data

The project includes UMLS data files in the `data/` directory:
- `umls_classes.json`: UMLS semantic types and their definitions
- `umls_terms.txt`: Medical terms from UMLS
- `umls_hierarchy.json`: Hierarchical relationships between semantic types

## Usage

### Quick Start

Run the main script to execute the ontology learning tasks:

```bash
python main.py
```

### Interactive Demo

For an interactive demonstration:

```bash
python umls_typer.py
```

### Comprehensive Testing

Run the full test suite:

```bash
python test_umls_comprehensive.py
```

### Data Management

Create or validate UMLS data:

```bash
python umls_data_utils.py
```

## Project Structure (Clean Version)

```
├── main.py                        # Main execution script
├── umls_typer.py                  # Clean UMLS term classification module
├── umls_data_utils.py             # Data management utilities
├── test_umls_comprehensive.py     # Comprehensive test suite
├── quick_umls_demo.py             # Quick demo script
├── data/                          # UMLS data files
│   ├── umls_classes.json          # Semantic types and definitions
│   ├── umls_terms.txt             # Medical terms
│   ├── umls_hierarchy.json        # Hierarchical relationships
│   └── LLMs4OL_UMLS/             # LLMs4OL compatible dataset
├── LLMs4OL/                       # Original LLMs4OL framework
├── .env.example                   # Environment configuration template
├── .env                           # Environment configuration (not tracked)
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Core Modules

### `UMLSTermTyper` Class
Main class for medical term classification with methods:
- `classify_term(term)`: Classify a single medical term
- `classify_terms_batch(terms)`: Classify multiple terms with rate limiting
- `check_hierarchical_relationship(child, parent)`: Check if hierarchical relationship exists
- `get_stats()`: Get statistics about loaded data

### `UMLSDataManager` Class
Data management operations:
- `create_sample_umls_data()`: Create sample data for development
- `validate_data_files()`: Validate data integrity
- `load_umls_data()`: Load all UMLS data
- `create_llms4ol_compatible_dataset()`: Generate LLMs4OL format data

### `UMLSTestSuite` Class
Comprehensive testing framework:
- `test_basic_classification()`: Basic term classification tests
- `test_domain_specific_categories()`: Domain-specific testing
- `test_random_terms()`: Random term testing
- `test_hierarchical_relationships()`: Relationship testing
Classifies a given term's part of speech using the LLM.

### `create_samples(neg_seed, pos_seed, n: int)`
Generates positive and negative test pairs for hierarchical relationship detection using WordNet's hypernym chains.

### `is_subclass(test_pairs)`
Evaluates whether the LLM can correctly identify superclass-subclass relationships between term pairs.

## Research Context

This project is part of research in the intersection of:
- **Semantic Web & Data Analysis and Intelligence (S&DAI)**
- **Natural Language Processing**
- **Ontology Engineering**
- **Large Language Model Applications**

## Rate Limiting

The code includes built-in delays (5 seconds between API calls) to respect Google's API rate limits. Adjust as needed based on your API quota.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

This is a research project. For questions or collaboration opportunities, please open an issue or contact the repository owner.

## Acknowledgments

- Uses Google's Gemini model for natural language understanding
- Leverages NLTK's WordNet corpus for taxonomic ground truth
- Built for academic research in ontology learning