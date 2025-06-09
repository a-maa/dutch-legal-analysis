"""
Logical Consistency Score (LCS) Implementation

This module calculates the Logical Consistency Score (LCS), which measures
the logical consistency of statements in legal documents.

Formula: LCS = 1 - (Contradictions / Total Statement Pairs)

The implementation uses:
1. AllenNLP for contradiction detection
2. Spacy for linguistic preprocessing and sentence segmentation
"""

import os
import json
import logging
import re
import numpy as np
import spacy
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from allennlp.predictors.predictor import Predictor
import torch
from itertools import combinations
from tqdm import tqdm
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LogicalConsistencyScorer:
    """Class to evaluate logical consistency in legal documents."""
    
    def __init__(self, 
                 nli_model_path: Optional[str] = None,
                 spacy_model: str = "nl_core_news_md",
                 contradiction_threshold: float = 0.7,
                 max_pairs: int = 1000):
        """
        Initialize the Logical Consistency Scorer.
        
        Args:
            nli_model_path: Path to AllenNLP NLI model (if None, will use default)
            spacy_model: Name of the spaCy model to use
            contradiction_threshold: Threshold for classifying as contradiction (0-1)
            max_pairs: Maximum number of statement pairs to evaluate
        """
        self.contradiction_threshold = contradiction_threshold
        self.max_pairs = max_pairs
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            # Try to download the model
            try:
                os.system(f"python -m spacy download {spacy_model}")
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Downloaded and loaded spaCy model: {spacy_model}")
            except Exception as e2:
                logger.error(f"Error downloading spaCy model: {e2}")
                self.nlp = None
        
        # Load AllenNLP NLI model
        try:
            if nli_model_path:
                self.predictor = Predictor.from_path(nli_model_path)
            else:
                # Use default model
                self.predictor = Predictor.from_path(
                    "https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz"
                )
            logger.info("Loaded AllenNLP NLI model")
        except Exception as e:
            logger.error(f"Error loading AllenNLP NLI model: {e}")
            self.predictor = None
    
    def _extract_statements(self, text: str) -> List[str]:
        """
        Extract statements from text using spaCy.
        
        Args:
            text: The text to extract statements from
            
        Returns:
            List of statements
        """
        if not self.nlp:
            # Fallback to simple sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]
        
        try:
            doc = self.nlp(text)
            statements = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
            return statements
        except Exception as e:
            logger.error(f"Error extracting statements with spaCy: {e}")
            # Fallback to simple sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _filter_statements(self, statements: List[str]) -> List[str]:
        """
        Filter statements to keep only those that make factual claims.
        
        Args:
            statements: List of statements to filter
            
        Returns:
            Filtered list of statements
        """
        # This is a simplified implementation
        # A real implementation would use more sophisticated NLP techniques
        
        # Filter out questions
        filtered = [s for s in statements if not s.endswith('?')]
        
        # Filter out very short statements
        filtered = [s for s in filtered if len(s.split()) >= 5]
        
        # Filter out statements that are likely not factual claims
        non_factual_patterns = [
            r'^(Ik|Wij|U|Jij|Je) (denk|denken|vindt|vinden|meent|menen)',
            r'^(Misschien|Wellicht|Mogelijk|Waarschijnlijk)',
            r'^(Zou|Zouden|Kan|Kunnen|Mag|Mogen)'
        ]
        
        for pattern in non_factual_patterns:
            filtered = [s for s in filtered if not re.match(pattern, s, re.IGNORECASE)]
        
        return filtered
    
    def _check_contradiction(self, statement1: str, statement2: str) -> Tuple[bool, float]:
        """
        Check if two statements contradict each other.
        
        Args:
            statement1: First statement
            statement2: Second statement
            
        Returns:
            Tuple of (is_contradiction, confidence_score)
        """
        if not self.predictor:
            logger.warning("NLI predictor not available, returning default")
            return False, 0.0
        
        try:
            # Get prediction
            result = self.predictor.predict(
                premise=statement1,
                hypothesis=statement2
            )
            
            # Get contradiction probability
            contradiction_prob = result["label_probs"][0]  # Assuming [contradiction, neutral, entailment]
            
            # Check if it exceeds threshold
            is_contradiction = contradiction_prob >= self.contradiction_threshold
            
            return is_contradiction, contradiction_prob
        
        except Exception as e:
            logger.error(f"Error checking contradiction: {e}")
            return False, 0.0
    
    def _translate_to_english(self, text: str) -> str:
        """
        Translate Dutch text to English for compatibility with English NLI models.
        
        This is a placeholder. In a real implementation, you would use a
        translation service or model.
        
        Args:
            text: Dutch text to translate
            
        Returns:
            English translation
        """
        # This is a placeholder
        # In a real implementation, you would use a translation API or model
        # For example, you could use the Helsinki-NLP/opus-mt-nl-en model from Hugging Face
        
        # For now, we'll just return the original text
        # This assumes you're using a multilingual model that can handle Dutch
        return text
    
    def calculate_lcs(self, text: str, translate_to_english: bool = False) -> Dict:
        """
        Calculate the Logical Consistency Score.
        
        Args:
            text: The text to analyze
            translate_to_english: Whether to translate to English before analysis
            
        Returns:
            Dictionary with LCS results
        """
        # Extract statements
        statements = self._extract_statements(text)
        
        # Filter statements
        statements = self._filter_statements(statements)
        
        if len(statements) < 2:
            return {
                "score": 100.0,  # No pairs means no contradictions
                "num_statements": len(statements),
                "num_pairs": 0,
                "contradictions": []
            }
        
        # Generate pairs of statements
        pairs = list(combinations(statements, 2))
        
        # Limit number of pairs to evaluate
        if len(pairs) > self.max_pairs:
            # Randomly sample pairs
            indices = np.random.choice(len(pairs), self.max_pairs, replace=False)
            pairs = [pairs[i] for i in indices]
        
        num_pairs = len(pairs)
        contradictions = []
        
        # Check each pair for contradictions
        for stmt1, stmt2 in tqdm(pairs, desc="Checking statement pairs", disable=None):
            # Translate if needed
            if translate_to_english:
                stmt1_proc = self._translate_to_english(stmt1)
                stmt2_proc = self._translate_to_english(stmt2)
            else:
                stmt1_proc = stmt1
                stmt2_proc = stmt2
            
            # Check for contradiction
            is_contradiction, confidence = self._check_contradiction(stmt1_proc, stmt2_proc)
            
            # Also check the reverse direction
            is_contradiction_rev, confidence_rev = self._check_contradiction(stmt2_proc, stmt1_proc)
            
            # If either direction shows contradiction, count it
            if is_contradiction or is_contradiction_rev:
                # Use the higher confidence score
                final_confidence = max(confidence, confidence_rev)
                
                contradictions.append({
                    "statement1": stmt1,
                    "statement2": stmt2,
                    "confidence": final_confidence
                })
        
        # Calculate LCS score
        num_contradictions = len(contradictions)
        lcs_score = 1.0 - (num_contradictions / num_pairs) if num_pairs > 0 else 1.0
        
        return {
            "score": lcs_score * 100,  # Convert to percentage
            "num_statements": len(statements),
            "num_pairs": num_pairs,
            "num_contradictions": num_contradictions,
            "contradictions": contradictions
        }


def evaluate_document(document_text: str, nli_model_path: Optional[str] = None) -> Dict:
    """
    Evaluate a document using the Logical Consistency Score.
    
    Args:
        document_text: The text of the document to evaluate
        nli_model_path: Path to AllenNLP NLI model (optional)
        
    Returns:
        Dictionary with evaluation results
    """
    # Initialize scorer
    scorer = LogicalConsistencyScorer(nli_model_path=nli_model_path)
    
    # Calculate LCS
    results = scorer.calculate_lcs(document_text)
    
    return {
        "metric": "Logical Consistency Score (LCS)",
        "score": results["score"],
        "details": {
            "num_statements": results["num_statements"],
            "num_pairs_evaluated": results["num_pairs"],
            "num_contradictions": results["num_contradictions"],
            "contradiction_examples": [c["statement1"] + " ↔ " + c["statement2"] 
                                     for c in results["contradictions"][:3]]
        }
    }


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Het bedrijf heeft in 2023 een omzet behaald van €1.2 miljoen.
    De jaaromzet over 2023 bedroeg €1.5 miljoen volgens het jaarverslag.
    Het personeelsbestand is in 2023 gegroeid van 15 naar 20 medewerkers.
    De onderneming had eind 2023 in totaal 25 mensen in dienst.
    Het kantoorpand in Amsterdam is eigendom van de vennootschap.
    De vennootschap huurt het kantoorpand in Amsterdam van een derde partij.
    """
    
    # Evaluate
    results = evaluate_document(sample_text)
    print(json.dumps(results, indent=2))