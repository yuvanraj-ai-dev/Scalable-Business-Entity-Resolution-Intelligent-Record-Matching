#!/usr/bin/env python3
"""
Matching Model: LightGBM Classifier

Trains a binary classifier on (S1, S2/S3) pairs from training data.
Outputs match probability; threshold tuned on validation split for F_0.5.
"""

import lightgbm as lgb
import numpy as np
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score
import joblib
import os


def f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F_beta score: weights precision (beta^2 × recall vs. precision)."""
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


def compute_threshold_for_f_beta(y_true: np.ndarray, y_scores: np.ndarray, beta: float = 0.5) -> Tuple[float, float]:
    """
    Find threshold that maximizes F_beta score on validation data.
    
    Args:
        y_true: Binary labels (0/1)
        y_scores: Predicted probabilities
        beta: Beta for F_beta score (0.5 weights precision 2x over recall)
    
    Returns:
        (best_threshold, best_f_beta_score)
    """
    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0.5
    best_score = 0.0
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f_score = f_beta_score(precision, recall, beta)
        
        if f_score > best_score:
            best_score = f_score
            best_threshold = thresh
    
    return best_threshold, best_score


class MatchingModel:
    """LightGBM-based matching classifier."""
    
    def __init__(self, params: Dict = None):
        default_params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None
        self.threshold = 0.5
        self.feature_names = []
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        val_split: float = 0.2,
    ) -> Dict:
        """
        Train the model and tune threshold on validation split.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (0 = no match, 1 = match)
            feature_names: List of feature names
            val_split: Fraction for validation (threshold tuning)
        
        Returns:
            Dict with training metrics
        """
        self.feature_names = feature_names
        
        # Split for training and validation (threshold tuning)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y
        )
        
        # Train
        train_data = lgb.Dataset(X_train, label=y_train, feature_names=feature_names)
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params.pop("n_estimators", 200),
        )
        
        # Predict on validation
        y_val_pred = self.model.predict(X_val)
        
        # Tune threshold for F_0.5
        self.threshold, f_score = compute_threshold_for_f_beta(y_val, y_val_pred, beta=0.5)
        
        # Metrics
        y_val_binary = (y_val_pred >= self.threshold).astype(int)
        tp = np.sum((y_val_binary == 1) & (y_val == 1))
        fp = np.sum((y_val_binary == 1) & (y_val == 0))
        fn = np.sum((y_val_binary == 0) & (y_val == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            "threshold": self.threshold,
            "f_0.5": f_score,
            "precision": precision,
            "recall": recall,
            "val_samples": len(X_val),
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict match probability for samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Probabilities (0–1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using tuned threshold."""
        probs = self.predict(X)
        return (probs >= self.threshold).astype(int)
    
    def save(self, path: str):
        """Save model and threshold to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path + ".model")
        with open(path + ".threshold", "w") as f:
            f.write(str(self.threshold))
        with open(path + ".features", "w") as f:
            f.write("\n".join(self.feature_names))
    
    def load(self, path: str):
        """Load model and threshold from disk."""
        self.model = lgb.Booster(model_file=path + ".model")
        with open(path + ".threshold", "r") as f:
            self.threshold = float(f.read().strip())
        with open(path + ".features", "r") as f:
            self.feature_names = [line.strip() for line in f]
