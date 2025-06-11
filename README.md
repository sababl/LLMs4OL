# LLMs4OL - Large Language Models for Ontology Learning

A research project exploring the use of Large Language Models for ontology learning tasks, specifically focusing on term classification and hierarchical relationship detection using Google's Gemini model.

## Overview

This project investigates how LLMs can be leveraged for ontology learning by:
- Classifying terms by their part of speech
- Detecting hypernym-hyponym relationships between concepts
- Evaluating model performance on taxonomic relationship identification

## Features

- **Term Classification**: Automatically classify terms as nouns, verbs, adjectives, or adverbs
- **Hierarchical Relationship Detection**: Identify superclass-subclass relationships between concepts
- **WordNet Integration**: Leverage WordNet's taxonomic structure for generating test datasets
- **Google Gemini Integration**: Use Google's Gemini model for natural language understanding tasks

## Setup

### Prerequisites

- Python 3.7+
- Google API key for Gemini model access
- Required Python packages (install via pip):
  ```bash
  pip install nltk google-generativeai python-dotenv
  ```

### Configuration

1. Clone this repository
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Add your Google API key to the `.env` file:
   ```
   GOOGLE_API_KEY=your_api_key_here
   MODEL_NAME=gemini-1.5-flash-latest
   ```

### NLTK Setup

Download required NLTK data:
```python
import nltk
nltk.download('wordnet')
```

## Usage

Run the main script to execute the ontology learning tasks:

```bash
python main.py
```

The script will:
1. Extract random noun lemmas from WordNet
2. Classify each term's part of speech using the LLM
3. Generate test pairs for hierarchical relationship detection
4. Evaluate the model's ability to identify superclass-subclass relationships

## Project Structure

```
├── main.py          # Main execution script
├── tasks.py         # Core functionality for LLM-based ontology tasks
├── .env.example     # Environment configuration template
├── .env             # Environment configuration (not tracked)
├── .gitignore       # Git ignore rules
├── LICENSE          # MIT License
└── README.md        # This file
```

## Core Functions

### `type_term(term: str)`
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