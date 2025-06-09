"""
Legal Reference Verification Score (LRVS) Implementation

This module calculates the Legal Reference Verification Score (LRVS), which measures
the accuracy of legal references to the Dutch Civil Code (Burgerlijk Wetboek, BW).

Formula: LRVS = (Verifiable BW Articles) / (Total BW Articles Cited) × 100%

The implementation uses a combination of:
1. Regex pattern matching to identify BW article references
2. Fuzzy text matching using fuzzywuzzy to verify references against a known database
"""

import re
import os
import json
import logging
from typing import Dict, List, Tuple, Set, Optional, Union
import pandas as pd
from fuzzywuzzy import fuzz, process
import PyPDF2
import pdfplumber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LegalReferenceVerifier:
    """Class to verify legal references in Dutch legal documents."""
    
    # Regex patterns for Dutch Civil Code (BW) references
    BW_PATTERNS = [
        r'artikel\s+(\d+(?::\d+)?(?:a|b|c|d|e|f|g|h|i|j|k|l|m)?)\s+(?:van het|BW|Burgerlijk\s+Wetboek)',
        r'art\.?\s+(\d+(?::\d+)?(?:a|b|c|d|e|f|g|h|i|j|k|l|m)?)\s+(?:van het|BW|Burgerlijk\s+Wetboek)',
        r'(?:BW|Burgerlijk\s+Wetboek)\s+artikel\s+(\d+(?::\d+)?(?:a|b|c|d|e|f|g|h|i|j|k|l|m)?)',
        r'(?:BW|Burgerlijk\s+Wetboek)\s+art\.?\s+(\d+(?::\d+)?(?:a|b|c|d|e|f|g|h|i|j|k|l|m)?)'
    ]
    
    def __init__(self, civil_code_pdf_path: Optional[str] = None, 
                 civil_code_json_path: Optional[str] = None,
                 similarity_threshold: int = 80):
        """
        Initialize the Legal Reference Verifier.
        
        Args:
            civil_code_pdf_path: Path to the Dutch Civil Code PDF file
            civil_code_json_path: Path to a JSON file containing pre-extracted articles
            similarity_threshold: Threshold for fuzzy matching (0-100)
        """
        self.similarity_threshold = similarity_threshold
        self.valid_articles = set()
        self.article_content = {}
        
        # Load civil code references
        if civil_code_json_path and os.path.exists(civil_code_json_path):
            self._load_from_json(civil_code_json_path)
        elif civil_code_pdf_path and os.path.exists(civil_code_pdf_path):
            self._extract_from_pdf(civil_code_pdf_path)
        else:
            logger.warning("No valid civil code source provided. Verification will be limited.")
    
    def _load_from_json(self, json_path: str) -> None:
        """Load civil code articles from a JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                self.article_content = data
                self.valid_articles = set(data.keys())
            elif isinstance(data, list):
                for article in data:
                    if 'id' in article and 'content' in article:
                        article_id = article['id']
                        self.valid_articles.add(article_id)
                        self.article_content[article_id] = article['content']
            
            logger.info(f"Loaded {len(self.valid_articles)} articles from JSON")
        except Exception as e:
            logger.error(f"Error loading civil code from JSON: {e}")
    
    def _extract_from_pdf(self, pdf_path: str) -> None:
        """Extract civil code articles from a PDF file."""
        try:
            # This is a simplified implementation
            # A real implementation would need more sophisticated PDF parsing
            text = ""
            
            # First try with PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}. Trying pdfplumber...")
                
                # Fallback to pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or "" + "\n"
            
            # Extract articles using regex patterns
            # This is a simplified approach and would need refinement for production
            article_patterns = [
                r'Artikel\s+(\d+(?::\d+)?(?:a|b|c|d|e|f|g|h|i|j|k|l|m)?)[.\s\n]+(.*?)(?=Artikel\s+\d+|$)',
                r'Art\.\s+(\d+(?::\d+)?(?:a|b|c|d|e|f|g|h|i|j|k|l|m)?)[.\s\n]+(.*?)(?=Art\.\s+\d+|$)'
            ]
            
            for pattern in article_patterns:
                matches = re.finditer(pattern, text, re.DOTALL)
                for match in matches:
                    article_id = match.group(1).strip()
                    content = match.group(2).strip()
                    self.valid_articles.add(article_id)
                    self.article_content[article_id] = content
            
            logger.info(f"Extracted {len(self.valid_articles)} articles from PDF")
        except Exception as e:
            logger.error(f"Error extracting civil code from PDF: {e}")
    
    def extract_references(self, text: str) -> List[str]:
        """
        Extract BW article references from text.
        
        Args:
            text: The text to extract references from
            
        Returns:
            List of article references
        """
        references = []
        
        for pattern in self.BW_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                article_ref = match.group(1).strip()
                references.append(article_ref)
        
        return references
    
    def verify_reference(self, reference: str) -> Tuple[bool, Optional[str]]:
        """
        Verify if a reference exists in the civil code.
        
        Args:
            reference: The article reference to verify
            
        Returns:
            Tuple of (is_valid, matched_reference)
        """
        # Direct match
        if reference in self.valid_articles:
            return True, reference
        
        # Fuzzy match if we have valid articles
        if self.valid_articles:
            best_match, score = process.extractOne(
                reference, 
                self.valid_articles, 
                scorer=fuzz.token_sort_ratio
            )
            
            if score >= self.similarity_threshold:
                return True, best_match
        
        # No match found
        return False, None
    
    def calculate_lrvs(self, text: str) -> Dict:
        """
        Calculate the Legal Reference Verification Score.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with LRVS results
        """
        references = self.extract_references(text)
        total_references = len(references)
        
        if total_references == 0:
            return {
                "score": 100.0,  # No references means no errors
                "total_references": 0,
                "verified_references": 0,
                "unverified_references": [],
                "verified_details": []
            }
        
        verified_count = 0
        verified_details = []
        unverified_references = []
        
        for ref in references:
            is_valid, matched_ref = self.verify_reference(ref)
            
            if is_valid:
                verified_count += 1
                verified_details.append({
                    "reference": ref,
                    "matched_to": matched_ref,
                    "content": self.article_content.get(matched_ref, "Content not available")
                })
            else:
                unverified_references.append(ref)
        
        lrvs_score = (verified_count / total_references) * 100 if total_references > 0 else 100.0
        
        return {
            "score": lrvs_score,
            "total_references": total_references,
            "verified_references": verified_count,
            "unverified_references": unverified_references,
            "verified_details": verified_details
        }


def evaluate_document(document_text: str, civil_code_path: Optional[str] = None) -> Dict:
    """
    Evaluate a document using the Legal Reference Verification Score.
    
    Args:
        document_text: The text of the document to evaluate
        civil_code_path: Path to the Dutch Civil Code (PDF or JSON)
        
    Returns:
        Dictionary with evaluation results
    """
    verifier = LegalReferenceVerifier(
        civil_code_pdf_path=civil_code_path if civil_code_path and civil_code_path.endswith('.pdf') else None,
        civil_code_json_path=civil_code_path if civil_code_path and civil_code_path.endswith('.json') else None
    )
    
    results = verifier.calculate_lrvs(document_text)
    
    return {
        "metric": "Legal Reference Verification Score (LRVS)",
        "score": results["score"],
        "details": {
            "total_references": results["total_references"],
            "verified_references": results["verified_references"],
            "unverified_references": results["unverified_references"]
        }
    }


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Volgens artikel 6:162 BW is degene die een onrechtmatige daad pleegt, 
    verplicht de schade te vergoeden. Artikel 3:13 van het Burgerlijk Wetboek 
    stelt grenzen aan de uitoefening van bevoegdheden. Art. 6:74 BW regelt 
    de wanprestatie.
    """
    
    # Path to Dutch Civil Code (replace with actual path)
    civil_code_path = "../data/dutch_civil_code.pdf"
    
    # Evaluate
    results = evaluate_document(sample_text, civil_code_path)
    print(json.dumps(results, indent=2))