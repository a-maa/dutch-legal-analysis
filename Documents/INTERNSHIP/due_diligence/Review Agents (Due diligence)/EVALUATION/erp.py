"""
Evidence Retrieval Precision (ERP) Implementation

This module calculates the Evidence Retrieval Precision (ERP), which measures
the accuracy of evidence citations in legal documents.

Formula: ERP = (Verified Evidence Citations) / (Total Evidence Citations) × 100%

The implementation uses:
1. Semantic similarity with sentence-transformers for robust matching
2. Advanced text processing to identify evidence citations
"""

import os
import json
import logging
import re
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvidenceRetriever:
    """Class to evaluate evidence retrieval precision in legal documents."""
    
    # Patterns to identify evidence citations in Dutch legal text
    EVIDENCE_PATTERNS = [
        r'(?:zie|volgens|blijkt uit|blijkens|conform|zoals vermeld in|zoals beschreven in|zoals aangegeven in|zoals gesteld in|zoals bepaald in|zoals vastgelegd in|zoals opgenomen in)\s+([^.;:!?]+)',
        r'(?:verwijst naar|refereert aan|citeert|haalt aan)\s+([^.;:!?]+)',
        r'(?:op basis van|gebaseerd op|met verwijzing naar|onder verwijzing naar)\s+([^.;:!?]+)',
        r'(?:bewijs|bewijsstuk|document|stuk|bijlage|exhibit)\s+([A-Z0-9]+)',
        r'(?:pagina|bladzijde|pagina\'s|bladzijden)\s+(\d+(?:\s*-\s*\d+)?)',
    ]
    
    def __init__(self, 
                 evidence_database: Optional[Dict[str, str]] = None,
                 evidence_files_dir: Optional[str] = None,
                 model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 similarity_threshold: float = 0.75):
        """
        Initialize the Evidence Retriever.
        
        Args:
            evidence_database: Dictionary mapping evidence IDs to content
            evidence_files_dir: Directory containing evidence files
            model_name: Name of the sentence transformer model to use
            similarity_threshold: Threshold for semantic similarity matching (0-1)
        """
        self.evidence_database = evidence_database or {}
        self.similarity_threshold = similarity_threshold
        
        # Load evidence from directory if provided
        if evidence_files_dir and os.path.isdir(evidence_files_dir):
            self._load_evidence_from_dir(evidence_files_dir)
        
        # Load model for semantic similarity
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def _load_evidence_from_dir(self, directory: str) -> None:
        """
        Load evidence from files in a directory.
        
        Args:
            directory: Path to directory containing evidence files
        """
        try:
            file_count = 0
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                
                if os.path.isfile(file_path):
                    try:
                        # Try to load as JSON first
                        if filename.endswith('.json'):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                                if isinstance(data, dict):
                                    # If it's a dictionary, add each key-value pair
                                    for key, value in data.items():
                                        if isinstance(value, str):
                                            self.evidence_database[key] = value
                                        elif isinstance(value, dict) and 'content' in value:
                                            self.evidence_database[key] = value['content']
                                
                                file_count += 1
                        
                        # Otherwise load as text
                        else:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                # Use filename as the evidence ID
                                evidence_id = os.path.splitext(filename)[0]
                                self.evidence_database[evidence_id] = content
                                file_count += 1
                    
                    except Exception as e:
                        logger.warning(f"Could not load evidence file {filename}: {e}")
            
            logger.info(f"Loaded {file_count} evidence files from {directory}")
            logger.info(f"Total evidence items in database: {len(self.evidence_database)}")
        
        except Exception as e:
            logger.error(f"Error loading evidence from directory: {e}")
    
    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract evidence citations from text.
        
        Args:
            text: The text to extract citations from
            
        Returns:
            List of dictionaries with citation details
        """
        citations = []
        
        # Extract citations using patterns
        for pattern in self.EVIDENCE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citation_text = match.group(1).strip()
                full_citation = match.group(0).strip()
                
                # Skip very short citations
                if len(citation_text) < 3:
                    continue
                
                # Create citation object
                citation = {
                    "citation_text": citation_text,
                    "full_citation": full_citation,
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "context": text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                }
                
                citations.append(citation)
        
        return citations
    
    def verify_citation(self, citation: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Verify if a citation exists in the evidence database.
        
        Args:
            citation: The citation to verify
            
        Returns:
            Tuple of (is_verified, evidence_id, similarity_score)
        """
        if not self.evidence_database or not self.model:
            return False, None, None
        
        citation_text = citation["citation_text"]
        
        # Encode the citation text
        citation_embedding = self.model.encode(citation_text, convert_to_tensor=True)
        
        best_match = None
        best_score = 0.0
        best_evidence_id = None
        
        # Compare with each evidence item
        for evidence_id, evidence_content in self.evidence_database.items():
            # For long evidence content, split into chunks
            if len(evidence_content) > 1000:
                chunks = [evidence_content[i:i+1000] for i in range(0, len(evidence_content), 500)]
            else:
                chunks = [evidence_content]
            
            for chunk in chunks:
                # Encode the evidence chunk
                evidence_embedding = self.model.encode(chunk, convert_to_tensor=True)
                
                # Calculate cosine similarity
                similarity = util.pytorch_cos_sim(citation_embedding, evidence_embedding).item()
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = chunk
                    best_evidence_id = evidence_id
        
        # Check if the best match exceeds the threshold
        if best_score >= self.similarity_threshold:
            return True, best_evidence_id, best_score
        
        return False, None, best_score
    
    def calculate_erp(self, text: str) -> Dict:
        """
        Calculate the Evidence Retrieval Precision.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with ERP results
        """
        citations = self.extract_citations(text)
        total_citations = len(citations)
        
        if total_citations == 0:
            return {
                "score": 100.0,  # No citations means no errors
                "total_citations": 0,
                "verified_citations": 0,
                "unverified_citations": [],
                "verified_details": []
            }
        
        verified_count = 0
        verified_details = []
        unverified_citations = []
        
        for citation in citations:
            is_verified, evidence_id, similarity_score = self.verify_citation(citation)
            
            if is_verified:
                verified_count += 1
                verified_details.append({
                    "citation": citation["citation_text"],
                    "evidence_id": evidence_id,
                    "similarity_score": similarity_score,
                    "context": citation["context"]
                })
            else:
                unverified_citations.append({
                    "citation": citation["citation_text"],
                    "context": citation["context"],
                    "best_similarity": similarity_score
                })
        
        erp_score = (verified_count / total_citations) * 100 if total_citations > 0 else 100.0
        
        return {
            "score": erp_score,
            "total_citations": total_citations,
            "verified_citations": verified_count,
            "unverified_citations": unverified_citations,
            "verified_details": verified_details
        }


def evaluate_document(document_text: str, evidence_path: Optional[str] = None) -> Dict:
    """
    Evaluate a document using the Evidence Retrieval Precision metric.
    
    Args:
        document_text: The text of the document to evaluate
        evidence_path: Path to evidence database (directory or JSON file)
        
    Returns:
        Dictionary with evaluation results
    """
    evidence_database = None
    evidence_dir = None
    
    # Load evidence if provided
    if evidence_path:
        if os.path.isdir(evidence_path):
            evidence_dir = evidence_path
        elif os.path.isfile(evidence_path) and evidence_path.endswith('.json'):
            try:
                with open(evidence_path, 'r', encoding='utf-8') as f:
                    evidence_database = json.load(f)
            except Exception as e:
                logger.error(f"Error loading evidence database: {e}")
    
    # Initialize retriever
    retriever = EvidenceRetriever(
        evidence_database=evidence_database,
        evidence_files_dir=evidence_dir
    )
    
    # Calculate ERP
    results = retriever.calculate_erp(document_text)
    
    return {
        "metric": "Evidence Retrieval Precision (ERP)",
        "score": results["score"],
        "details": {
            "total_citations": results["total_citations"],
            "verified_citations": results["verified_citations"],
            "unverified_count": len(results["unverified_citations"])
        }
    }


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Volgens het rapport van de accountant (bijlage A) is er sprake van een tekort
    van €50.000. Zoals vermeld in de jaarrekening 2024 bedroeg de omzet €1.2 miljoen.
    Op basis van de getuigenverklaring van dhr. Jansen blijkt dat de leveringen niet
    conform de overeenkomst zijn uitgevoerd. Zie document B12 voor de specificaties
    van de geleverde goederen.
    """
    
    # Path to evidence database (replace with actual path)
    evidence_path = "../data/evidence"
    
    # Evaluate
    results = evaluate_document(sample_text, evidence_path)
    print(json.dumps(results, indent=2))