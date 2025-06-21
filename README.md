# UMLS Ontology Learning with LLMs

A clean, efficient implementation for medical ontology learning using Large Language Models, specifically focused on the UMLS (Unified Medical Language System) domain with Google's Gemini model.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project demonstrates how Large Language Models can be leveraged for medical ontology learning tasks including:

- **Medical Term Classification**: Classify medical terms into UMLS semantic types
- **Hierarchical Relationship Detection**: Identify parent-child relationships between medical concepts  
- **Semantic Relation Extraction**: Detect non-taxonomic relationships between medical entities

## Features

- **LLM-Powered**: Uses Google Gemini for medical language understanding
- **UMLS Integration**: Built on the Unified Medical Language System
- **Clean Architecture**: Professional code structure with proper separation of concerns
- **Comprehensive Testing**: Unified test suite with detailed evaluation metrics
- **Configurable**: Centralized configuration management
- **Easy to Use**: Simple CLI interface with demo, test, and interactive modes

## Quick Start

### Prerequisites

- Python 3.8+
- Google API key for Gemini model access

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LLMs4OL
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   cp .env.example .env
   # Add your Google API key to .env file
   echo "GOOGLE_API_KEY=your_api_key_here" >> .env
   ```

### Usage

#### Quick Demo
```bash
python main_clean.py
```

#### Run Comprehensive Tests
```bash
python main_clean.py --test
```

#### Interactive Mode
```bash
python main_clean.py --interactive
```

#### Install as Package
```bash
pip install -e .
```

## Project Structure

```
├── src/umls_ontology/          # Core package
│   ├── __init__.py            # Package exports
│   └── core.py               # Main functionality
├── config/                   # Configuration
│   └── settings.py          # Centralized settings
├── tests/                   # Testing
│   └── test_suite.py       # Comprehensive test suite
├── data/                   # UMLS data files
├── results/               # Output directory
├── main_clean.py         # Main entry point
├── requirements.txt     # Dependencies
└── setup.py            # Package installation
```

## Configuration

The system uses centralized configuration in `config/settings.py`. Key settings include:

- **API Configuration**: Google API key and model selection
- **Rate Limiting**: API call delays to respect quotas
- **Data Paths**: UMLS data file locations
- **Templates**: Customizable prompts for different tasks

## UMLS Data

The project includes sample UMLS data:

- **Semantic Types**: 127 medical concept categories
- **Medical Terms**: 297+ medical terms for classification
- **Hierarchical Relationships**: Parent-child relationships between concepts
- **Semantic Relations**: Non-taxonomic relationships between entities

## Core Components

### UMLSOntologyLearner
Main orchestrator class that coordinates all functionality.

### UMLSClassifier  
Handles medical term classification into UMLS semantic types.

### UMLSRelationshipAnalyzer
Manages hierarchical and semantic relationship detection.

### UMLSDataLoader
Handles loading and validation of UMLS data files.

### LLMClient
Manages all interactions with the Google Gemini API.

## Testing

The project includes comprehensive testing with evaluation metrics:

```bash
# Run all tests
python main_clean.py --test

# Expected output:
# PASS: Hierarchy Detection: ~85% accuracy
# WARN: Semantic Relations: ~67% accuracy  
# FAIL: Term Classification: Needs optimization
# PASS: System Integration: Fully functional
```

## Example Usage

### Term Classification
```python
from src.umls_ontology import UMLSOntologyLearner

learner = UMLSOntologyLearner()
result = learner.classifier.classify_term("diabetes")
print(f"diabetes -> {result}")  # diabetes -> Disease or Syndrome
```

### Hierarchy Detection
```python
is_subtype = learner.relationship_analyzer.check_hierarchy(
    "Human", "Animal"
)
print(f"Human is subtype of Animal: {is_subtype}")  # True
```

## Performance

Current system performance on test datasets:

| Task | Accuracy | Status |
|------|----------|---------|
| Hierarchy Detection | 85.7% | Good |
| Semantic Relations | 66.7% | Moderate |
| Term Classification | Variable | Needs optimization |
| System Integration | 100% | Excellent |

## API Requirements

- **Google Gemini API**: Required for LLM functionality
- **Rate Limits**: Respects free tier quotas (15 requests/minute)
- **Error Handling**: Automatic retry logic with exponential backoff

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **UMLS**: Unified Medical Language System for medical terminology
- **Google Gemini**: Large Language Model for natural language understanding
- **LLMs4OL Framework**: Inspiration for ontology learning evaluation

## Citation

If you use this work in your research, please cite:

```bibtex
@software{umls_ontology_learning,
  title={UMLS Ontology Learning with Large Language Models},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/LLMs4OL}
}
```
