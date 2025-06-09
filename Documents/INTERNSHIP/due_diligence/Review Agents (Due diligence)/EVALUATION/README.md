# BERTje Evaluation Framework for Dutch Legal Documents

This framework provides a comprehensive evaluation suite for Dutch legal documents using BERTje, a Dutch BERT model. It implements six different metrics to assess various aspects of legal document quality and accuracy.

## Metrics

The framework includes the following six metrics:

### 1. Legal Reference Verification Score (LRVS)

**Formula:** LRVS = (Verifiable BW Articles) / (Total BW Articles Cited) × 100%

This metric measures the accuracy of legal references to the Dutch Civil Code (Burgerlijk Wetboek, BW). It uses regex pattern matching and fuzzy text matching with the fuzzywuzzy package to identify and verify legal references.

### 2. Risk Classification Consistency (RCC)

**Formula:** RCC = 1 - (Standard Deviation of Risk Levels for Similar Questions)

This metric evaluates the consistency of risk classifications across similar questions or scenarios. It groups similar questions using semantic similarity and calculates the standard deviation of risk levels within each group.

### 3. Evidence Retrieval Precision (ERP)

**Formula:** ERP = (Verified Evidence Citations) / (Total Evidence Citations) × 100%

This metric measures the precision of evidence citations in legal documents. It uses semantic similarity with sentence-transformers for robust matching between citations and a database of evidence.

### 4. Evidence-Conclusion Alignment Score (ECAS)

**Formula:** ECAS = MLM_prediction_score(evidence → conclusion)

This metric evaluates how well evidence supports the conclusions in legal documents. It uses BERTje to assess the alignment between evidence and conclusions through masked language modeling and natural language inference.

### 5. Logical Consistency Score (LCS)

**Formula:** LCS = 1 - (Contradictions / Total Statement Pairs)

This metric measures the logical consistency of statements in a document. It uses AllenNLP for contradiction detection and spaCy for linguistic preprocessing to identify contradictory statement pairs.

### 6. Legal Rule Application Coherence (LRAC)

**Formula:** LRAC = MLM_coherence_score(rule, application)

This metric evaluates how coherently legal rules are applied to specific cases. It uses BERTje to score the coherence between a rule and its application through masked language modeling.

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the required models:

```bash
python -m spacy download nl_core_news_md
```

## Usage

### Command Line Interface

The framework provides a command-line interface for easy evaluation:

```bash
# Evaluate a single document
python evaluate.py --document path/to/document.txt --output results.json

# Evaluate a batch of documents
python evaluate.py --batch path/to/documents/ --output batch_results.json

# Generate a visual report
python evaluate.py --document path/to/document.txt --report

# Specify resources for evaluation
python evaluate.py --document path/to/document.txt \
                  --civil-code path/to/civil_code.pdf \
                  --evidence path/to/evidence/ \
                  --questions-risks path/to/questions_risks.json
```

### Python API

You can also use the framework programmatically:

```python
from evaluate import BERTjeEvaluator

# Initialize evaluator
evaluator = BERTjeEvaluator(
    civil_code_path="path/to/civil_code.pdf",
    evidence_path="path/to/evidence/",
    nli_model_path=None,  # Use default
    output_dir="results"
)

# Evaluate a document
results = evaluator.evaluate_document(
    document_path="path/to/document.txt",
    questions_risks_path="path/to/questions_risks.json"
)

# Save results
output_path = evaluator.save_results(results)

# Generate report
report_path = evaluator.generate_report(results)
```

## Input Formats

### Document

The document to evaluate should be a text file containing the legal document text.

### Questions and Risk Levels (for RCC)

For the Risk Classification Consistency metric, you need to provide a JSON file with questions and their risk levels:

```json
[
  {
    "question": "Is er sprake van een contractbreuk?",
    "risk_level": 4
  },
  {
    "question": "Wat zijn de fiscale gevolgen van deze transactie?",
    "risk_level": 3
  }
]
```

### Dutch Civil Code (for LRVS)

For the Legal Reference Verification Score, you can provide the Dutch Civil Code as either:

- A PDF file containing the full text of the Dutch Civil Code
- A JSON file with pre-extracted articles

### Evidence Database (for ERP)

For the Evidence Retrieval Precision metric, you can provide evidence as either:

- A directory containing evidence files (text or JSON)
- A JSON file mapping evidence IDs to content

## Output Format

The evaluation results are saved as a JSON file with the following structure:

```json
{
  "document_path": "path/to/document.txt",
  "timestamp": "2025-06-06 10:00:00",
  "metrics": {
    "lrvs": {
      "metric": "Legal Reference Verification Score (LRVS)",
      "score": 85.7,
      "details": { ... }
    },
    "rcc": { ... },
    "erp": { ... },
    "ecas": { ... },
    "lcs": { ... },
    "lrac": { ... }
  },
  "overall_score": 82.3
}
```

## Visualization

The framework can generate visual reports of the evaluation results, showing the scores for each metric and the overall score.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Sentence-Transformers
- AllenNLP
- spaCy
- fuzzywuzzy
- numpy
- pandas
- matplotlib
- seaborn

## License

This project is licensed under the MIT License - see the LICENSE file for details.