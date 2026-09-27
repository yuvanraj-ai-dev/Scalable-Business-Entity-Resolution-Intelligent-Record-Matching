#!/usr/bin/env python3
"""
End-to-End Pipeline

Orchestrates:
1. Load training data (chunked)
2. Build blocking index on S2/S3
3. Generate training pairs (candidate + ground truth)
4. Train model
5. Generate candidates on test S1
6. Score candidates
7. Output matching_results.tsv and candidate_pairs.tsv
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Iterator
import os
from blocking import (
    BlockingIndex, build_blocking_index, get_blocking_keys, get_candidates
)
from features import compute_features, features_to_vector, FEATURE_NAMES
from model import MatchingModel


CHUNK_SIZE = 50000  # Read files in 50k-row chunks


def read_tsv_chunked(path: str, chunksize: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Generator: yield chunks of a TSV file."""
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, dtype=str):
        yield chunk


def load_entities_all(path: str) -> List[Dict[str, str]]:
    """Load all entities from a source file into memory (for blocking index)."""
    entities = []
    for chunk in read_tsv_chunked(path):
        for _, row in chunk.iterrows():
            entities.append({
                "entity_id": row["entity_id"],
                "business_name": row.get("business_name", ""),
                "business_address": row.get("business_address", ""),
                "country": row.get("country", ""),
            })
    return entities


def load_ground_truth(path: str) -> Dict[str, List[str]]:
    """Load ground truth: S1 ID -> list of S2/S3 IDs."""
    gt = {}
    for chunk in read_tsv_chunked(path):
        for _, row in chunk.iterrows():
            s1_id = row["source1_entity_id"]
            matched = row.get("matched_entity_ids", "")
            if matched and isinstance(matched, str) and matched.strip():
                ids = [x.strip() for x in matched.split(",")]
                gt[s1_id] = ids
            else:
                gt[s1_id] = []
    return gt


def generate_training_pairs(
    s1_entities: List[Dict],
    s2_entities: List[Dict],
    s3_entities: List[Dict],
    ground_truth: Dict[str, List[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (S1, S2/S3) pairs from training data.
    
    Positive examples: pairs in ground truth
    Negative examples: hard negatives from blocking, plus random negatives
    
    Returns:
        (X, y) — feature matrix and binary labels
    """
    # Index S2/S3 by ID for lookup
    s2_by_id = {e["entity_id"]: e for e in s2_entities}
    s3_by_id = {e["entity_id"]: e for e in s3_entities}
    s23_by_id = {**s2_by_id, **s3_by_id}
    
    # Build blocking index
    s2_index = build_blocking_index(s2_entities)
    s3_index = build_blocking_index(s3_entities)
    
    X_list = []
    y_list = []
    
    print(f"Generating training pairs for {len(s1_entities)} S1 entities...")
    
    for i, s1 in enumerate(s1_entities):
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1} / {len(s1_entities)}")
        
        s1_id = s1["entity_id"]
        s1_name = s1["business_name"]
        s1_addr = s1["business_address"]
        s1_country = s1["country"]
        
        # Retrieve candidates from blocking
        s2_cands, s3_cands = get_candidates(
            s1_id, s1_name, s1_addr, s1_country, s2_index, s3_index
        )
        all_cands = s2_cands | s3_cands
        
        # Positive pairs: from ground truth
        true_matches = set(ground_truth.get(s1_id, []))
        
        # Generate features for all candidates
        for cand_id in all_cands:
            if cand_id not in s23_by_id:
                continue
            
            cand = s23_by_id[cand_id]
            cand_name = cand["business_name"]
            cand_addr = cand["business_address"]
            cand_country = cand["country"]
            
            features = compute_features(
                s1_id, s1_name, s1_addr, s1_country,
                cand_id, cand_name, cand_addr, cand_country,
            )
            
            X_list.append(features_to_vector(features, FEATURE_NAMES))
            y_list.append(1.0 if cand_id in true_matches else 0.0)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    print(f"Generated {len(X)} pairs: {np.sum(y)} positives, {len(X) - np.sum(y)} negatives")
    print(f"  Positive ratio: {np.mean(y):.4f}")
    
    return X, y


def train_pipeline(
    train_source1_path: str,
    train_source2_path: str,
    train_source3_path: str,
    ground_truth_path: str,
    model_path: str,
) -> MatchingModel:
    """Train the matching model on training data."""
    
    print("Loading training data...")
    s1_entities = load_entities_all(train_source1_path)
    s2_entities = load_entities_all(train_source2_path)
    s3_entities = load_entities_all(train_source3_path)
    ground_truth = load_ground_truth(ground_truth_path)
    
    print(f"Loaded: S1={len(s1_entities)}, S2={len(s2_entities)}, S3={len(s3_entities)}")
    
    # Generate training pairs
    X, y = generate_training_pairs(s1_entities, s2_entities, s3_entities, ground_truth)
    
    # Train model
    print("Training model...")
    model = MatchingModel()
    metrics = model.train(X, y, FEATURE_NAMES, val_split=0.2)
    
    print(f"Model trained:")
    print(f"  Threshold: {metrics['threshold']:.4f}")
    print(f"  F_0.5: {metrics['f_0.5']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model


def predict_on_test(
    test_source1_path: str,
    test_source2_path: str,
    test_source3_path: str,
    model: MatchingModel,
    output_dir: str,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Generate matches for test data.
    
    Returns:
        (matching_results, candidate_pairs) — dicts of S1 ID -> list of matched IDs
    """
    
    print("Loading test data for S2/S3...")
    s2_entities = load_entities_all(test_source2_path)
    s3_entities = load_entities_all(test_source3_path)
    
    s2_by_id = {e["entity_id"]: e for e in s2_entities}
    s3_by_id = {e["entity_id"]: e for e in s3_entities}
    s23_by_id = {**s2_by_id, **s3_by_id}
    
    # Build blocking index
    s2_index = build_blocking_index(s2_entities)
    s3_index = build_blocking_index(s3_entities)
    
    matching_results = {}
    candidate_pairs = {}
    
    print("Generating predictions for test S1 entities...")
    
    for chunk_idx, chunk in enumerate(read_tsv_chunked(test_source1_path)):
        if (chunk_idx + 1) % 10 == 0:
            print(f"  Processed {(chunk_idx + 1) * len(chunk)} rows")
        
        for _, row in chunk.iterrows():
            s1_id = row["entity_id"]
            s1_name = row.get("business_name", "")
            s1_addr = row.get("business_address", "")
            s1_country = row.get("country", "")
            
            # Get candidates
            s2_cands, s3_cands = get_candidates(
                s1_id, s1_name, s1_addr, s1_country, s2_index, s3_index
            )
            all_cands = sorted(list(s2_cands | s3_cands))
            candidate_pairs[s1_id] = all_cands
            
            # Score candidates
            scores = []
            for cand_id in all_cands:
                if cand_id not in s23_by_id:
                    continue
                
                cand = s23_by_id[cand_id]
                features = compute_features(
                    s1_id, s1_name, s1_addr, s1_country,
                    cand_id, cand["business_name"], cand["business_address"], cand["country"]
                )
                
                X_pair = np.array([features_to_vector(features, FEATURE_NAMES)], dtype=np.float32)
                prob = model.predict(X_pair)[0]
                scores.append((prob, cand_id))
            
            # Apply threshold and sort
            matches = sorted(
                [cand_id for prob, cand_id in scores if prob >= model.threshold],
                key=lambda x: x
            )
            matching_results[s1_id] = matches
    
    print(f"Generated predictions for {len(matching_results)} S1 entities")
    
    return matching_results, candidate_pairs


def write_output(
    matching_results: Dict[str, List[str]],
    candidate_pairs: Dict[str, List[str]],
    output_dir: str,
):
    """Write matching_results.tsv and candidate_pairs.tsv."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Write matching_results.tsv
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    with open(matching_path, "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(matching_results.keys()):
            matches = matching_results[s1_id]
            matched_str = ",".join(matches) if matches else ""
            f.write(f"{s1_id}\t{matched_str}\n")
    
    print(f"Wrote {matching_path}")
    
    # Write candidate_pairs.tsv
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    with open(candidate_path, "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(candidate_pairs.keys()):
            cands = candidate_pairs[s1_id]
            cands_str = ",".join(cands) if cands else ""
            f.write(f"{s1_id}\t{cands_str}\n")
    
    print(f"Wrote {candidate_path}")
