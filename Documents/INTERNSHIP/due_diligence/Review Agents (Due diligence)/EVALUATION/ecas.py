"""
Evidence-Conclusion Alignment Score (ECAS) Implementation

This module calculates the Evidence-Conclusion Alignment Score (ECAS), which measures
how well evidence supports the conclusions in legal documents.

Formula: ECAS = MLM_prediction_score(evidence → conclusion)

The implementation uses:
1. BERTje (Dutch BERT model) to evaluate the alignment between evidence and conclusions
2. Natural language inference techniques to assess entailment
"""

import os
import json
import logging
import re
import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvidenceConclusionAlignmentScorer:
    """Class to evaluate alignment between evidence and conclusions."""
    
    def __init__(self, 
                 model_name: str = "wietsedv/bertje",
                 nli_model_name: str = "wietsedv/bert-base-dutch-cased-finetuned-nli",
                 device: Optional[str] = None):
        """
        Initialize the Evidence-Conclusion Alignment Scorer.
        
        Args:
            model_name: Name of the BERTje model to use
            nli_model_name: Name of the NLI model to use
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load BERTje model for masked language modeling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info(f"Loaded BERTje model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Error loading BERTje model: {e}")
            self.tokenizer = None
            self.model = None
        
        # Load NLI model for entailment prediction
        try:
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
            self.nli_model.eval()
            logger.info(f"Loaded NLI model: {nli_model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Error loading NLI model: {e}")
            self.nli_tokenizer = None
            self.nli_model = None
    
    def _extract_evidence_conclusion_pairs(self, text: str) -> List[Dict[str, str]]:
        """
        Extract evidence-conclusion pairs from text.
        
        This is a simplified implementation. In a real-world scenario,
        this would use more sophisticated NLP techniques to identify
        evidence and the conclusions they support.
        
        Args:
            text: The text to extract pairs from
            
        Returns:
            List of dictionaries with evidence and conclusion
        """
        # Simple pattern to identify conclusions
        # In Dutch legal documents, conclusions often follow certain phrases
        conclusion_patterns = [
            r'(?:concluderen|concludeert|concludeerde|geconcludeerd)\s+(?:dat|:)\s+([^.;!?]+[.;!?])',
            r'(?:de conclusie is|als conclusie geldt|concluderend)\s+(?:dat|:)\s+([^.;!?]+[.;!?])',
            r'(?:daarom|derhalve|dus|bijgevolg|zodoende|daardoor)\s+([^.;!?]+[.;!?])',
            r'(?:wij zijn van mening|wij zijn van oordeel|het oordeel is)\s+(?:dat|:)\s+([^.;!?]+[.;!?])'
        ]
        
        # Extract all sentences as potential evidence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Extract conclusions
        conclusions = []
        for pattern in conclusion_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                conclusion = match.group(1).strip()
                if conclusion and len(conclusion) > 10:
                    conclusions.append({
                        "conclusion": conclusion,
                        "position": match.start()
                    })
        
        # If no conclusions found, try to use the last sentence as conclusion
        if not conclusions and sentences:
            conclusions.append({
                "conclusion": sentences[-1],
                "position": text.rfind(sentences[-1])
            })
        
        # Create evidence-conclusion pairs
        pairs = []
        for conclusion_info in conclusions:
            conclusion = conclusion_info["conclusion"]
            position = conclusion_info["position"]
            
            # Find sentences that appear before the conclusion
            evidence_candidates = [s for s in sentences if text.find(s) < position]
            
            # Use up to 3 sentences before the conclusion as evidence
            evidence = " ".join(evidence_candidates[-3:]) if evidence_candidates else ""
            
            if evidence and conclusion:
                pairs.append({
                    "evidence": evidence,
                    "conclusion": conclusion
                })
        
        return pairs
    
    def _calculate_mlm_score(self, evidence: str, conclusion: str) -> float:
        """
        Calculate the masked language model prediction score.
        
        This measures how well the evidence predicts the conclusion
        by masking words in the conclusion and seeing if the model
        can predict them given the evidence.
        
        Args:
            evidence: The evidence text
            conclusion: The conclusion text
            
        Returns:
            Prediction score (0-1)
        """
        if not self.tokenizer or not self.model:
            logger.warning("MLM model not available, returning default score")
            return 0.5
        
        try:
            # Tokenize the conclusion
            tokens = self.tokenizer.tokenize(conclusion)
            
            # Skip if too few tokens
            if len(tokens) < 3:
                return 0.5
            
            # Select words to mask (excluding stopwords and punctuation)
            # In a real implementation, you'd have a proper Dutch stopwords list
            dutch_stopwords = {'de', 'het', 'een', 'en', 'van', 'in', 'is', 'op', 'dat', 'die', 'te', 'zijn'}
            mask_candidates = [i for i, token in enumerate(tokens) 
                              if token not in dutch_stopwords 
                              and not all(c in '.,;:!?()[]{}' for c in token)
                              and not token.startswith('##')]
            
            # Skip if no suitable tokens to mask
            if not mask_candidates:
                return 0.5
            
            # Limit to max 5 tokens to mask
            num_to_mask = min(5, len(mask_candidates))
            indices_to_mask = np.random.choice(mask_candidates, num_to_mask, replace=False)
            
            scores = []
            
            # For each masked token, calculate prediction score
            for mask_idx in indices_to_mask:
                # Create a copy of tokens and mask the selected token
                masked_tokens = tokens.copy()
                original_token = masked_tokens[mask_idx]
                masked_tokens[mask_idx] = self.tokenizer.mask_token
                
                # Create input text with context
                masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)
                input_text = f"{evidence} {masked_text}"
                
                # Tokenize for model
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
                
                # Find position of mask token
                mask_token_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
                
                # Skip if mask token not found
                if len(mask_token_index) == 0:
                    continue
                
                # Get model predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Get predicted token probabilities
                logits = outputs.logits
                mask_token_logits = logits[0, mask_token_index, :]
                
                # Get the token ID of the original token
                original_token_id = self.tokenizer.convert_tokens_to_ids(original_token)
                
                # Get probability of the original token
                probs = torch.nn.functional.softmax(mask_token_logits, dim=-1)
                original_token_prob = probs[0, original_token_id].item()
                
                scores.append(original_token_prob)
            
            # Return average score
            return np.mean(scores) if scores else 0.5
        
        except Exception as e:
            logger.error(f"Error calculating MLM score: {e}")
            return 0.5
    
    def _calculate_nli_score(self, evidence: str, conclusion: str) -> float:
        """
        Calculate the natural language inference score.
        
        This measures entailment between evidence and conclusion.
        
        Args:
            evidence: The evidence text
            conclusion: The conclusion text
            
        Returns:
            Entailment score (0-1)
        """
        if not self.nli_tokenizer or not self.nli_model:
            logger.warning("NLI model not available, returning default score")
            return 0.5
        
        try:
            # Tokenize
            inputs = self.nli_tokenizer(evidence, conclusion, return_tensors="pt", 
                                       truncation=True, max_length=512).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Assuming the model outputs [contradiction, neutral, entailment]
            # Return the entailment probability
            entailment_score = probs[0, 2].item()
            
            return entailment_score
        
        except Exception as e:
            logger.error(f"Error calculating NLI score: {e}")
            return 0.5
    
    def calculate_ecas(self, text: str) -> Dict:
        """
        Calculate the Evidence-Conclusion Alignment Score.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with ECAS results
        """
        # Extract evidence-conclusion pairs
        pairs = self._extract_evidence_conclusion_pairs(text)
        
        if not pairs:
            return {
                "score": 0.0,  # No pairs means we can't evaluate
                "num_pairs": 0,
                "pairs": []
            }
        
        pair_scores = []
        
        for pair in pairs:
            evidence = pair["evidence"]
            conclusion = pair["conclusion"]
            
            # Calculate MLM score
            mlm_score = self._calculate_mlm_score(evidence, conclusion)
            
            # Calculate NLI score
            nli_score = self._calculate_nli_score(evidence, conclusion)
            
            # Combine scores (weighted average)
            # NLI is more directly relevant to the task, so we weight it higher
            combined_score = (0.3 * mlm_score) + (0.7 * nli_score)
            
            pair_scores.append({
                "evidence": evidence,
                "conclusion": conclusion,
                "mlm_score": mlm_score,
                "nli_score": nli_score,
                "combined_score": combined_score
            })
        
        # Calculate overall score (average of all pairs)
        overall_score = np.mean([p["combined_score"] for p in pair_scores]) * 100
        
        return {
            "score": overall_score,
            "num_pairs": len(pairs),
            "pairs": pair_scores
        }


def evaluate_document(document_text: str) -> Dict:
    """
    Evaluate a document using the Evidence-Conclusion Alignment Score.
    
    Args:
        document_text: The text of the document to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    # Initialize scorer
    scorer = EvidenceConclusionAlignmentScorer()
    
    # Calculate ECAS
    results = scorer.calculate_ecas(document_text)
    
    return {
        "metric": "Evidence-Conclusion Alignment Score (ECAS)",
        "score": results["score"],
        "details": {
            "num_evidence_conclusion_pairs": results["num_pairs"],
            "average_mlm_score": np.mean([p["mlm_score"] for p in results["pairs"]]) * 100 if results["pairs"] else 0,
            "average_nli_score": np.mean([p["nli_score"] for p in results["pairs"]]) * 100 if results["pairs"] else 0
        }
    }


if __name__ == "__main__":
    # Example usage
    sample_text = """
    De facturen tonen aan dat er in de periode januari-maart 2024 in totaal €75.000 
    is gefactureerd aan de klant. Uit de bankafschriften blijkt dat slechts €25.000 
    is ontvangen. De klant heeft in een e-mail d.d. 15 april 2024 aangegeven dat 
    de geleverde diensten niet voldeden aan de afgesproken kwaliteitseisen.
    
    Daarom concluderen wij dat er een betalingsachterstand is van €50.000, maar 
    dat er een reëel risico bestaat dat deze vordering betwist zal worden op basis 
    van de kwaliteit van de geleverde diensten.
    """
    
    # Evaluate
    results = evaluate_document(sample_text)
    print(json.dumps(results, indent=2))