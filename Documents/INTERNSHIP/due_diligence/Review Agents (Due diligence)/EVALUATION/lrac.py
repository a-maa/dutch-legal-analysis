"""
Legal Rule Application Coherence (LRAC) Implementation

This module calculates the Legal Rule Application Coherence (LRAC), which measures
how coherently legal rules are applied to specific cases.

Formula: LRAC = MLM_coherence_score(rule, application)

The implementation uses:
1. BERTje (Dutch BERT model) to evaluate the coherence between rules and their applications
2. Rule extraction techniques to identify legal rules and their applications
"""

import os
import json
import logging
import re
import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LegalRuleApplicationCoherence:
    """Class to evaluate coherence between legal rules and their applications."""
    
    # Patterns to identify legal rules in Dutch legal text
    RULE_PATTERNS = [
        r'(?:volgens|op grond van|op basis van|krachtens|ingevolge)\s+(?:artikel|art\.)\s+[^\s,;:]+\s+(?:van|BW|Burgerlijk\s+Wetboek)[^.;:!?]+',
        r'(?:artikel|art\.)\s+[^\s,;:]+\s+(?:van|BW|Burgerlijk\s+Wetboek)[^.;:!?]+(?:bepaalt|stelt|schrijft voor|vereist)',
        r'(?:de wet|het recht|de jurisprudentie|de rechtspraak)\s+(?:bepaalt|stelt|schrijft voor|vereist)[^.;:!?]+',
        r'(?:volgens|op grond van|op basis van|krachtens|ingevolge)\s+(?:vaste|constante)\s+(?:jurisprudentie|rechtspraak)[^.;:!?]+'
    ]
    
    # Patterns to identify rule applications
    APPLICATION_PATTERNS = [
        r'(?:in dit geval|in casu|in de onderhavige zaak|in deze situatie)[^.;:!?]+',
        r'(?:toegepast op|toepassing op|toegepast in|toepassing in)[^.;:!?]+',
        r'(?:dit betekent|dat betekent|hieruit volgt|daaruit volgt)[^.;:!?]+',
        r'(?:derhalve|daarom|dus|bijgevolg|zodoende|daardoor)[^.;:!?]+'
    ]
    
    def __init__(self, 
                 model_name: str = "wietsedv/bertje",
                 device: Optional[str] = None,
                 max_context_length: int = 512,
                 coherence_threshold: float = 0.6):
        """
        Initialize the Legal Rule Application Coherence evaluator.
        
        Args:
            model_name: Name of the BERTje model to use
            device: Device to run the model on ('cpu' or 'cuda')
            max_context_length: Maximum context length for the model
            coherence_threshold: Threshold for considering rule-application coherent (0-1)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_context_length = max_context_length
        self.coherence_threshold = coherence_threshold
        
        # Load BERTje model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info(f"Loaded BERTje model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Error loading BERTje model: {e}")
            self.tokenizer = None
            self.model = None
    
    def _extract_rules_and_applications(self, text: str) -> List[Dict[str, str]]:
        """
        Extract legal rules and their applications from text.
        
        Args:
            text: The text to extract from
            
        Returns:
            List of dictionaries with rule and application
        """
        # Extract rules
        rules = []
        for pattern in self.RULE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                rule_text = match.group(0).strip()
                position = match.start()
                rules.append({
                    "text": rule_text,
                    "position": position
                })
        
        # Extract applications
        applications = []
        for pattern in self.APPLICATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                application_text = match.group(0).strip()
                position = match.start()
                applications.append({
                    "text": application_text,
                    "position": position
                })
        
        # Match rules with applications
        pairs = []
        
        # Sort by position
        rules.sort(key=lambda x: x["position"])
        applications.sort(key=lambda x: x["position"])
        
        # For each rule, find the closest application that follows it
        for rule in rules:
            rule_pos = rule["position"]
            closest_app = None
            min_distance = float('inf')
            
            for app in applications:
                app_pos = app["position"]
                
                # Application must come after rule
                if app_pos > rule_pos:
                    distance = app_pos - rule_pos
                    
                    # Application should be within a reasonable distance
                    if distance < min_distance and distance < 1000:
                        min_distance = distance
                        closest_app = app
            
            if closest_app:
                pairs.append({
                    "rule": rule["text"],
                    "application": closest_app["text"],
                    "distance": min_distance
                })
        
        return pairs
    
    def _calculate_coherence_score(self, rule: str, application: str) -> float:
        """
        Calculate coherence score between a rule and its application.
        
        Args:
            rule: The legal rule
            application: The application of the rule
            
        Returns:
            Coherence score (0-1)
        """
        if not self.tokenizer or not self.model:
            logger.warning("Model not available, returning default score")
            return 0.5
        
        try:
            # Combine rule and application
            combined_text = f"{rule} [MASK] {application}"
            
            # Tokenize
            inputs = self.tokenizer(combined_text, return_tensors="pt").to(self.device)
            
            # Truncate if needed
            if inputs["input_ids"].shape[1] > self.max_context_length:
                inputs = self.tokenizer(
                    combined_text, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_context_length
                ).to(self.device)
            
            # Find position of mask token
            mask_token_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
            
            # If no mask token found, insert one
            if len(mask_token_index) == 0:
                # Split tokens
                rule_tokens = self.tokenizer.tokenize(rule)
                app_tokens = self.tokenizer.tokenize(application)
                
                # Insert mask token between
                all_tokens = rule_tokens + [self.tokenizer.mask_token] + app_tokens
                
                # Convert back to string
                combined_text = self.tokenizer.convert_tokens_to_string(all_tokens)
                
                # Tokenize again
                inputs = self.tokenizer(
                    combined_text, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_context_length
                ).to(self.device)
                
                # Find mask token position
                mask_token_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
                
                if len(mask_token_index) == 0:
                    logger.warning("Could not insert mask token, returning default score")
                    return 0.5
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predicted token probabilities
            logits = outputs.logits
            mask_token_logits = logits[0, mask_token_index, :]
            
            # Get top predicted tokens
            top_tokens = torch.topk(mask_token_logits, 5, dim=1)
            
            # Get coherence words (connectors that indicate logical flow)
            coherence_words = [
                "dus", "daarom", "derhalve", "bijgevolg", "zodoende",
                "waardoor", "zodat", "daardoor", "hierdoor", "aldus"
            ]
            
            # Convert to token IDs
            coherence_token_ids = [self.tokenizer.convert_tokens_to_ids(word) for word in coherence_words]
            
            # Check if any coherence words are in top predictions
            coherence_score = 0.0
            for token_id in coherence_token_ids:
                if token_id in top_tokens.indices[0]:
                    # Get the probability of this token
                    idx = (top_tokens.indices[0] == token_id).nonzero(as_tuple=True)[0]
                    if len(idx) > 0:
                        token_prob = torch.nn.functional.softmax(mask_token_logits, dim=1)[0, token_id].item()
                        coherence_score = max(coherence_score, token_prob)
            
            # If no coherence words found, use perplexity as fallback
            if coherence_score == 0.0:
                # Calculate perplexity
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                
                # Convert to a score between 0 and 1 (lower perplexity = higher coherence)
                # Typical perplexity range for Dutch text with BERTje might be 5-50
                coherence_score = max(0.0, min(1.0, 1.0 - (perplexity - 5) / 45))
            
            return coherence_score
        
        except Exception as e:
            logger.error(f"Error calculating coherence score: {e}")
            return 0.5
    
    def calculate_lrac(self, text: str) -> Dict:
        """
        Calculate the Legal Rule Application Coherence score.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with LRAC results
        """
        # Extract rule-application pairs
        pairs = self._extract_rules_and_applications(text)
        
        if not pairs:
            return {
                "score": 0.0,  # No pairs means we can't evaluate
                "num_pairs": 0,
                "pairs": []
            }
        
        pair_scores = []
        
        for pair in pairs:
            rule = pair["rule"]
            application = pair["application"]
            
            # Calculate coherence score
            coherence_score = self._calculate_coherence_score(rule, application)
            
            pair_scores.append({
                "rule": rule,
                "application": application,
                "coherence_score": coherence_score,
                "is_coherent": coherence_score >= self.coherence_threshold
            })
        
        # Calculate overall score (average of all pairs)
        overall_score = np.mean([p["coherence_score"] for p in pair_scores]) * 100
        
        # Calculate percentage of coherent pairs
        coherent_pairs = sum(1 for p in pair_scores if p["is_coherent"])
        coherent_percentage = (coherent_pairs / len(pair_scores)) * 100 if pair_scores else 0
        
        return {
            "score": overall_score,
            "num_pairs": len(pairs),
            "coherent_pairs_percentage": coherent_percentage,
            "pairs": pair_scores
        }


def evaluate_document(document_text: str) -> Dict:
    """
    Evaluate a document using the Legal Rule Application Coherence score.
    
    Args:
        document_text: The text of the document to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    # Initialize evaluator
    evaluator = LegalRuleApplicationCoherence()
    
    # Calculate LRAC
    results = evaluator.calculate_lrac(document_text)
    
    return {
        "metric": "Legal Rule Application Coherence (LRAC)",
        "score": results["score"],
        "details": {
            "num_rule_application_pairs": results["num_pairs"],
            "coherent_pairs_percentage": results["coherent_pairs_percentage"],
            "examples": [(p["rule"], p["application"]) for p in results["pairs"][:3]]
        }
    }


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Volgens artikel 6:162 BW is degene die een onrechtmatige daad pleegt, 
    verplicht de schade te vergoeden die de ander daardoor lijdt. In dit geval 
    heeft gedaagde door het niet nakomen van de afspraken schade veroorzaakt 
    bij eiser, waardoor gedaagde aansprakelijk is voor de geleden schade.
    
    Op grond van artikel 7:17 BW dient een zaak aan de overeenkomst te beantwoorden. 
    Toegepast op deze casus betekent dit dat de geleverde goederen moeten voldoen 
    aan de specificaties zoals overeengekomen in het contract.
    
    Ingevolge vaste jurisprudentie van de Hoge Raad moet bij de uitleg van 
    overeenkomsten worden gekeken naar de bedoeling van partijen. In de onderhavige 
    zaak blijkt uit de correspondentie tussen partijen dat zij beoogden een 
    duurzame samenwerking aan te gaan.
    """
    
    # Evaluate
    results = evaluate_document(sample_text)
    print(json.dumps(results, indent=2))