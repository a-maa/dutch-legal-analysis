"""
Risk Classification Consistency (RCC) Implementation

This module calculates the Risk Classification Consistency (RCC), which measures
the consistency of risk classifications across similar questions or scenarios.

Formula: RCC = 1 - (Standard Deviation of Risk Levels for Similar Questions)

The implementation:
1. Groups similar questions/scenarios using semantic similarity
2. Calculates the standard deviation of risk levels within each group
3. Computes the overall consistency score
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, fcluster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RiskClassificationConsistency:
    """Class to evaluate consistency of risk classifications."""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 similarity_threshold: float = 0.75,
                 risk_scale_min: int = 1,
                 risk_scale_max: int = 5):
        """
        Initialize the Risk Classification Consistency evaluator.
        
        Args:
            model_name: Name of the sentence transformer model to use
            similarity_threshold: Threshold for considering questions similar (0-1)
            risk_scale_min: Minimum value on the risk scale
            risk_scale_max: Maximum value on the risk scale
        """
        self.similarity_threshold = similarity_threshold
        self.risk_scale_min = risk_scale_min
        self.risk_scale_max = risk_scale_max
        self.risk_scale_range = risk_scale_max - risk_scale_min
        
        # Load model for semantic similarity
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to TF-IDF if model loading fails
            self.model = None
            logger.info("Will use TF-IDF for similarity calculation instead")
    
    def _normalize_risk_level(self, risk_level: Union[int, float, str]) -> float:
        """
        Normalize risk level to a 0-1 scale.
        
        Args:
            risk_level: The risk level to normalize
            
        Returns:
            Normalized risk level (0-1)
        """
        try:
            # Convert to float if it's a string
            if isinstance(risk_level, str):
                risk_level = float(risk_level.strip())
            
            # Normalize to 0-1 scale
            normalized = (risk_level - self.risk_scale_min) / self.risk_scale_range
            return max(0.0, min(1.0, normalized))  # Clamp to 0-1
        except (ValueError, TypeError):
            logger.warning(f"Could not normalize risk level: {risk_level}, using 0.5")
            return 0.5
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts.
        
        Args:
            texts: List of texts to compute embeddings for
            
        Returns:
            Array of embeddings
        """
        if self.model is not None:
            # Use sentence transformer
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            # Fallback to TF-IDF
            vectorizer = TfidfVectorizer(max_features=100)
            return vectorizer.fit_transform(texts).toarray()
    
    def _cluster_similar_questions(self, 
                                  questions: List[str], 
                                  embeddings: Optional[np.ndarray] = None) -> List[int]:
        """
        Cluster similar questions based on semantic similarity.
        
        Args:
            questions: List of questions to cluster
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            List of cluster IDs for each question
        """
        if embeddings is None:
            embeddings = self._compute_embeddings(questions)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Convert to distance matrix (1 - similarity)
        distance_matrix = 1 - similarity_matrix
        
        # Perform hierarchical clustering
        condensed_distance = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]
        Z = linkage(condensed_distance, method='average')
        
        # Form flat clusters
        # Distance threshold is (1 - similarity_threshold)
        distance_threshold = 1 - self.similarity_threshold
        clusters = fcluster(Z, distance_threshold, criterion='distance')
        
        return clusters
    
    def calculate_rcc(self, 
                     questions: List[str], 
                     risk_levels: List[Union[int, float, str]]) -> Dict:
        """
        Calculate the Risk Classification Consistency score.
        
        Args:
            questions: List of questions or scenarios
            risk_levels: Corresponding risk levels
            
        Returns:
            Dictionary with RCC results
        """
        if len(questions) != len(risk_levels):
            raise ValueError("Length of questions and risk_levels must be the same")
        
        if len(questions) == 0:
            return {
                "score": 1.0,  # Perfect consistency when no data
                "cluster_details": [],
                "overall_std_dev": 0.0
            }
        
        # Normalize risk levels
        normalized_risks = [self._normalize_risk_level(level) for level in risk_levels]
        
        # Compute embeddings and cluster similar questions
        embeddings = self._compute_embeddings(questions)
        clusters = self._cluster_similar_questions(questions, embeddings)
        
        # Calculate standard deviation within each cluster
        unique_clusters = set(clusters)
        cluster_details = []
        weighted_std_devs = []
        
        for cluster_id in unique_clusters:
            # Get indices of questions in this cluster
            indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            
            if len(indices) <= 1:
                # Skip clusters with only one question (std dev is 0)
                continue
            
            # Get questions and risk levels for this cluster
            cluster_questions = [questions[i] for i in indices]
            cluster_risks = [normalized_risks[i] for i in indices]
            
            # Calculate standard deviation
            std_dev = np.std(cluster_risks)
            
            # Weight by cluster size
            weight = len(indices) / len(questions)
            weighted_std_devs.append(std_dev * weight)
            
            cluster_details.append({
                "cluster_id": int(cluster_id),
                "size": len(indices),
                "questions": cluster_questions,
                "risk_levels": [risk_levels[i] for i in indices],
                "normalized_risks": cluster_risks,
                "std_dev": float(std_dev)
            })
        
        # Calculate overall weighted standard deviation
        overall_std_dev = sum(weighted_std_devs) if weighted_std_devs else 0.0
        
        # Calculate RCC score: 1 - std_dev
        # Ensure the score is between 0 and 1
        rcc_score = 1.0 - min(1.0, overall_std_dev)
        
        return {
            "score": rcc_score * 100,  # Convert to percentage
            "cluster_details": cluster_details,
            "overall_std_dev": overall_std_dev
        }


def evaluate_document(questions_risks: List[Dict[str, Union[str, int, float]]]) -> Dict:
    """
    Evaluate a document using the Risk Classification Consistency metric.
    
    Args:
        questions_risks: List of dictionaries with 'question' and 'risk_level' keys
        
    Returns:
        Dictionary with evaluation results
    """
    # Extract questions and risk levels
    questions = [item.get('question', '') for item in questions_risks]
    risk_levels = [item.get('risk_level', 0) for item in questions_risks]
    
    # Initialize evaluator
    evaluator = RiskClassificationConsistency()
    
    # Calculate RCC
    results = evaluator.calculate_rcc(questions, risk_levels)
    
    return {
        "metric": "Risk Classification Consistency (RCC)",
        "score": results["score"],
        "details": {
            "num_questions": len(questions),
            "num_clusters": len(results["cluster_details"]),
            "overall_std_dev": results["overall_std_dev"]
        }
    }


if __name__ == "__main__":
    # Example usage
    sample_data = [
        {"question": "Is er sprake van een contractbreuk?", "risk_level": 4},
        {"question": "Heeft de wederpartij het contract geschonden?", "risk_level": 4},
        {"question": "Is er een schending van de overeenkomst?", "risk_level": 5},
        {"question": "Wat zijn de fiscale gevolgen van deze transactie?", "risk_level": 3},
        {"question": "Wat zijn de belastingtechnische implicaties?", "risk_level": 2},
        {"question": "Is de deadline voor indiening verstreken?", "risk_level": 1},
    ]
    
    # Evaluate
    results = evaluate_document(sample_data)
    print(json.dumps(results, indent=2))