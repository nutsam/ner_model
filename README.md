# Text Processing and Named Entity Recognition (NER) Toolkit

This repository provides a Python-based toolkit for text preprocessing, text masking, and Named Entity Recognition (NER) using both English (spaCy) and Chinese (CKIP Transformers). The project is managed using Poetry for dependency management.

## Features
- **Text Preprocessing**: Cleans text by removing URLs, HTML tags, punctuation, and stopwords.
- **Text Masking**: Replaces Chinese and English words with underscores for privacy protection.
- **Named Entity Recognition (NER)**:
  - Uses **spaCy** for English NER.
  - Uses **CKIP Transformers** for Chinese NER.
  - Merges the results for a comprehensive entity extraction.

## Installation

### Prerequisites
Ensure you have Python installed (>=3.8).

### Using Poetry

1. **Install Poetry** (if not already installed):
   ```sh
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
2. **Install dependencies using Poetry**:
   ```sh
   poetry install
   ```
 
## Usage

Download the model first:

```sh
python -m spacy download en_core_web_sm
```

Run the script to preprocess text, mask sensitive data, and extract named entities:

```sh
poetry run python main.py
```
