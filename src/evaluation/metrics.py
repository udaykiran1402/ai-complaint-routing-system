"""Evaluation metrics for the complaint routing system."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from typing import Dict, List


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None) -> Dict:
    """
    Evaluate classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
    
    Returns:
        Dictionary with metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'classification_report': classification_report(y_true, y_pred, target_names=labels),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    return metrics


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Evaluate regression model.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2
    }
    
    return metrics


def evaluate_retrieval(relevant_items: List[int], retrieved_items: List[int], k: int = 5) -> Dict:
    """
    Evaluate retrieval/similarity search.
    
    Args:
        relevant_items: List of relevant item IDs
        retrieved_items: List of retrieved item IDs (top-k)
        k: Number of items to consider
    
    Returns:
        Dictionary with metrics
    """
    retrieved_k = retrieved_items[:k]
    relevant_set = set(relevant_items)
    retrieved_set = set(retrieved_k)
    
    # Recall@K
    recall_at_k = len(relevant_set & retrieved_set) / len(relevant_set) if relevant_set else 0
    
    # Precision@K
    precision_at_k = len(relevant_set & retrieved_set) / k if k > 0 else 0
    
    # Mean Reciprocal Rank (MRR)
    mrr = 0
    for i, item in enumerate(retrieved_k, 1):
        if item in relevant_set:
            mrr = 1 / i
            break
    
    metrics = {
        f'recall@{k}': recall_at_k,
        f'precision@{k}': precision_at_k,
        'mrr': mrr
    }
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Evaluation Metrics"):
    """Pretty print metrics."""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
        elif isinstance(value, str):
            print(f"\n{key}:")
            print(value)
        else:
            print(f"{key}: {value}")
    
    print("="*60)


if __name__ == "__main__":
    # Test metrics
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])
    
    metrics = evaluate_classification(y_true, y_pred, labels=['Low', 'Medium', 'High'])
    print_metrics(metrics, "Classification Metrics")
    
    y_true_reg = np.array([3.5, 7.2, 2.1, 5.8])
    y_pred_reg = np.array([3.2, 7.5, 2.3, 5.5])
    
    metrics_reg = evaluate_regression(y_true_reg, y_pred_reg)
    print_metrics(metrics_reg, "Regression Metrics")
