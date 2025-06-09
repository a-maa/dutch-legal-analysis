"""
BERTje Evaluation Framework

This is the main evaluation runner that integrates all six evaluation metrics
for Dutch legal document analysis using BERTje.

The six metrics are:
1. Legal Reference Verification Score (LRVS)
2. Risk Classification Consistency (RCC)
3. Evidence Retrieval Precision (ERP)
4. Evidence-Conclusion Alignment Score (ECAS)
5. Logical Consistency Score (LCS)
6. Legal Rule Application Coherence (LRAC)

Usage:
    python evaluate.py --document path/to/document.txt --output results.json
"""

import os
import sys
import json
import logging
import argparse
import time
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import metric modules
from lrvs import evaluate_document as evaluate_lrvs
from rcc import evaluate_document as evaluate_rcc
from erp import evaluate_document as evaluate_erp
from ecas import evaluate_document as evaluate_ecas
from lcs import evaluate_document as evaluate_lcs
from lrac import evaluate_document as evaluate_lrac

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluation.log")
    ]
)
logger = logging.getLogger(__name__)

class BERTjeEvaluator:
    """Main evaluator class that integrates all metrics."""
    
    def __init__(self, 
                 civil_code_path: Optional[str] = None,
                 evidence_path: Optional[str] = None,
                 nli_model_path: Optional[str] = None,
                 output_dir: str = "results"):
        """
        Initialize the BERTje Evaluator.
        
        Args:
            civil_code_path: Path to Dutch Civil Code (PDF or JSON)
            evidence_path: Path to evidence database (directory or JSON)
            nli_model_path: Path to NLI model for contradiction detection
            output_dir: Directory to save results
        """
        self.civil_code_path = civil_code_path
        self.evidence_path = evidence_path
        self.nli_model_path = nli_model_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def _load_document(self, document_path: str) -> str:
        """
        Load document text from file.
        
        Args:
            document_path: Path to document file
            
        Returns:
            Document text
        """
        try:
            with open(document_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            raise
    
    def _load_questions_risks(self, questions_risks_path: str) -> List[Dict]:
        """
        Load questions and risk levels from file.
        
        Args:
            questions_risks_path: Path to questions/risks JSON file
            
        Returns:
            List of dictionaries with questions and risk levels
        """
        try:
            with open(questions_risks_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading questions/risks: {e}")
            raise
    
    def evaluate_document(self, 
                         document_path: str, 
                         questions_risks_path: Optional[str] = None) -> Dict:
        """
        Evaluate a document using all metrics.
        
        Args:
            document_path: Path to document file
            questions_risks_path: Path to questions/risks JSON file (for RCC)
            
        Returns:
            Dictionary with evaluation results
        """
        # Load document
        document_text = self._load_document(document_path)
        
        # Initialize results
        results = {
            "document_path": document_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {}
        }
        
        # Evaluate LRVS
        logger.info("Evaluating Legal Reference Verification Score (LRVS)...")
        try:
            lrvs_results = evaluate_lrvs(document_text, self.civil_code_path)
            results["metrics"]["lrvs"] = lrvs_results
            logger.info(f"LRVS Score: {lrvs_results['score']:.2f}%")
        except Exception as e:
            logger.error(f"Error evaluating LRVS: {e}")
            results["metrics"]["lrvs"] = {"error": str(e)}
        
        # Evaluate RCC (if questions/risks provided)
        if questions_risks_path:
            logger.info("Evaluating Risk Classification Consistency (RCC)...")
            try:
                questions_risks = self._load_questions_risks(questions_risks_path)
                rcc_results = evaluate_rcc(questions_risks)
                results["metrics"]["rcc"] = rcc_results
                logger.info(f"RCC Score: {rcc_results['score']:.2f}%")
            except Exception as e:
                logger.error(f"Error evaluating RCC: {e}")
                results["metrics"]["rcc"] = {"error": str(e)}
        
        # Evaluate ERP
        logger.info("Evaluating Evidence Retrieval Precision (ERP)...")
        try:
            erp_results = evaluate_erp(document_text, self.evidence_path)
            results["metrics"]["erp"] = erp_results
            logger.info(f"ERP Score: {erp_results['score']:.2f}%")
        except Exception as e:
            logger.error(f"Error evaluating ERP: {e}")
            results["metrics"]["erp"] = {"error": str(e)}
        
        # Evaluate ECAS
        logger.info("Evaluating Evidence-Conclusion Alignment Score (ECAS)...")
        try:
            ecas_results = evaluate_ecas(document_text)
            results["metrics"]["ecas"] = ecas_results
            logger.info(f"ECAS Score: {ecas_results['score']:.2f}%")
        except Exception as e:
            logger.error(f"Error evaluating ECAS: {e}")
            results["metrics"]["ecas"] = {"error": str(e)}
        
        # Evaluate LCS
        logger.info("Evaluating Logical Consistency Score (LCS)...")
        try:
            lcs_results = evaluate_lcs(document_text, self.nli_model_path)
            results["metrics"]["lcs"] = lcs_results
            logger.info(f"LCS Score: {lcs_results['score']:.2f}%")
        except Exception as e:
            logger.error(f"Error evaluating LCS: {e}")
            results["metrics"]["lcs"] = {"error": str(e)}
        
        # Evaluate LRAC
        logger.info("Evaluating Legal Rule Application Coherence (LRAC)...")
        try:
            lrac_results = evaluate_lrac(document_text)
            results["metrics"]["lrac"] = lrac_results
            logger.info(f"LRAC Score: {lrac_results['score']:.2f}%")
        except Exception as e:
            logger.error(f"Error evaluating LRAC: {e}")
            results["metrics"]["lrac"] = {"error": str(e)}
        
        # Calculate overall score (average of all metrics)
        valid_scores = [m["score"] for m in results["metrics"].values() 
                       if isinstance(m, dict) and "score" in m]
        
        if valid_scores:
            results["overall_score"] = sum(valid_scores) / len(valid_scores)
            logger.info(f"Overall Score: {results['overall_score']:.2f}%")
        
        return results
    
    def save_results(self, results: Dict, output_path: Optional[str] = None) -> str:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            output_path: Path to save results (optional)
            
        Returns:
            Path to saved results
        """
        if not output_path:
            # Generate filename based on document name and timestamp
            document_name = os.path.basename(results["document_path"])
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"evaluation_{document_name}_{timestamp}.json")
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        return output_path
    
    def generate_report(self, results: Dict, output_path: Optional[str] = None) -> str:
        """
        Generate a visual report of evaluation results.
        
        Args:
            results: Evaluation results
            output_path: Path to save report (optional)
            
        Returns:
            Path to saved report
        """
        if not output_path:
            # Generate filename based on document name and timestamp
            document_name = os.path.basename(results["document_path"])
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"report_{document_name}_{timestamp}.pdf")
        
        # Extract scores
        metrics = {}
        for metric, data in results["metrics"].items():
            if isinstance(data, dict) and "score" in data:
                metrics[metric.upper()] = data["score"]
        
        if not metrics:
            logger.warning("No valid metrics found for report generation")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Bar chart
        ax = sns.barplot(x="Metric", y="Score", data=df, palette="viridis")
        
        # Add labels and title
        plt.title("BERTje Evaluation Results", fontsize=16)
        plt.xlabel("Metrics", fontsize=12)
        plt.ylabel("Score (%)", fontsize=12)
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for i, v in enumerate(df["Score"]):
            ax.text(i, v + 2, f"{v:.1f}%", ha="center")
        
        # Add overall score
        if "overall_score" in results:
            plt.axhline(y=results["overall_score"], color='r', linestyle='--', 
                       label=f"Overall: {results['overall_score']:.1f}%")
            plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Report saved to {output_path}")
        return output_path


def evaluate_batch(evaluator: BERTjeEvaluator, 
                  document_dir: str, 
                  output_path: Optional[str] = None) -> Dict:
    """
    Evaluate a batch of documents.
    
    Args:
        evaluator: BERTjeEvaluator instance
        document_dir: Directory containing documents to evaluate
        output_path: Path to save batch results (optional)
        
    Returns:
        Dictionary with batch evaluation results
    """
    # Find all text files in directory
    document_paths = []
    for ext in [".txt", ".md", ".json"]:
        document_paths.extend(list(Path(document_dir).glob(f"*{ext}")))
    
    if not document_paths:
        logger.warning(f"No documents found in {document_dir}")
        return {"error": "No documents found"}
    
    # Initialize batch results
    batch_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "document_dir": document_dir,
        "num_documents": len(document_paths),
        "documents": {}
    }
    
    # Evaluate each document
    for doc_path in document_paths:
        logger.info(f"Evaluating {doc_path}...")
        try:
            results = evaluator.evaluate_document(str(doc_path))
            batch_results["documents"][str(doc_path)] = results
        except Exception as e:
            logger.error(f"Error evaluating {doc_path}: {e}")
            batch_results["documents"][str(doc_path)] = {"error": str(e)}
    
    # Calculate average scores
    all_metrics = {}
    for doc_results in batch_results["documents"].values():
        if "metrics" in doc_results:
            for metric, data in doc_results["metrics"].items():
                if isinstance(data, dict) and "score" in data:
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(data["score"])
    
    # Compute averages
    batch_results["average_scores"] = {}
    for metric, scores in all_metrics.items():
        if scores:
            batch_results["average_scores"][metric] = sum(scores) / len(scores)
    
    # Calculate overall average
    if batch_results["average_scores"]:
        batch_results["overall_average"] = sum(batch_results["average_scores"].values()) / len(batch_results["average_scores"])
    
    # Save results
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(evaluator.output_dir, f"batch_results_{timestamp}.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=2)
    
    logger.info(f"Batch results saved to {output_path}")
    return batch_results


def main():
    """Main function to run evaluation from command line."""
    parser = argparse.ArgumentParser(description="BERTje Evaluation Framework")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--document", help="Path to document file")
    input_group.add_argument("--batch", help="Path to directory containing documents")
    
    # Additional inputs
    parser.add_argument("--questions-risks", help="Path to questions/risks JSON file (for RCC)")
    
    # Resource paths
    parser.add_argument("--civil-code", help="Path to Dutch Civil Code (PDF or JSON)")
    parser.add_argument("--evidence", help="Path to evidence database (directory or JSON)")
    parser.add_argument("--nli-model", help="Path to NLI model for contradiction detection")
    
    # Output options
    parser.add_argument("--output", help="Path to save results")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--report", action="store_true", help="Generate visual report")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BERTjeEvaluator(
        civil_code_path=args.civil_code,
        evidence_path=args.evidence,
        nli_model_path=args.nli_model,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    if args.document:
        # Single document evaluation
        results = evaluator.evaluate_document(
            document_path=args.document,
            questions_risks_path=args.questions_risks
        )
        
        # Save results
        output_path = evaluator.save_results(results, args.output)
        
        # Generate report if requested
        if args.report:
            report_path = evaluator.generate_report(results)
            print(f"Report saved to: {report_path}")
        
        print(f"Results saved to: {output_path}")
        print(f"Overall Score: {results.get('overall_score', 0):.2f}%")
    
    elif args.batch:
        # Batch evaluation
        batch_results = evaluate_batch(evaluator, args.batch, args.output)
        
        if "overall_average" in batch_results:
            print(f"Batch evaluation complete. Overall average: {batch_results['overall_average']:.2f}%")
        else:
            print("Batch evaluation complete, but no valid results were obtained.")


if __name__ == "__main__":
    main()