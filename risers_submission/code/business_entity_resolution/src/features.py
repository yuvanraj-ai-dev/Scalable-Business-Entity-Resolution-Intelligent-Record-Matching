#!/usr/bin/env python3
"""
Feature Engineering for Pairwise Entity Matching

Computes similarity features between S1 and S2/S3 pairs:
- Name similarity: Jaccard, token-sort Levenshtein ratio
- Address similarity: component-wise overlap, edit distance
- Metadata: country match, length ratios
"""

from difflib import SequenceMatcher
from collections import Counter
from typing import Dict, Tuple
import math
from blocking import tokenize, normalize_text, extract_postal_code, extract_state_city


def jaccard_similarity(s1: str, s2: str) -> float:
    """Jaccard index on token sets."""
    tokens1 = set(tokenize(s1))
    tokens2 = set(tokenize(s2))
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Token-sort Levenshtein ratio (order-invariant name comparison)."""
    # Normalize and tokenize
    t1 = tokenize(s1)
    t2 = tokenize(s2)
    
    # Sort tokens to handle reordering
    sorted1 = " ".join(sorted(t1))
    sorted2 = " ".join(sorted(t2))
    
    return SequenceMatcher(None, sorted1, sorted2).ratio()


def token_overlap_ratio(s1: str, s2: str) -> float:
    """Proportion of tokens in s1 that appear in s2."""
    tokens1 = tokenize(s1)
    tokens2 = set(tokenize(s2))
    
    if not tokens1:
        return 1.0 if not tokens2 else 0.0
    
    overlap = sum(1 for t in tokens1 if t in tokens2)
    return overlap / len(tokens1)


def address_component_similarity(addr1: str, addr2: str, country: str) -> float:
    """
    Compare address components (postal code, state, city).
    Strong match on postal/PIN is highly indicative.
    """
    postal1 = extract_postal_code(addr1)
    postal2 = extract_postal_code(addr2)
    
    if postal1 and postal2:
        if postal1 == postal2:
            return 1.0  # Exact postal match is very strong
        else:
            return 0.0  # Different postal codes → different addresses
    
    # Fall back to state/city match
    state1, city1 = extract_state_city(addr1, country)
    state2, city2 = extract_state_city(addr2, country)
    
    score = 0.0
    if state1 and state2 and state1 == state2:
        score += 0.5
    if city1 and city2 and city1 == city2:
        score += 0.5
    
    return score


def compute_features(
    s1_id: str, s1_name: str, s1_address: str, s1_country: str,
    s2_id: str, s2_name: str, s2_address: str, s2_country: str,
) -> Dict[str, float]:
    """
    Compute all features for an (S1, S2/S3) pair.
    
    Returns a dict of feature_name -> value, ready for a classifier.
    """
    features = {}
    
    # Country match
    features["country_match"] = 1.0 if normalize_text(s1_country) == normalize_text(s2_country) else 0.0
    
    # Name similarity
    features["name_jaccard"] = jaccard_similarity(s1_name, s2_name)
    features["name_levenshtein_ratio"] = levenshtein_ratio(s1_name, s2_name)
    features["name_token_overlap"] = token_overlap_ratio(s1_name, s2_name)
    
    # Address similarity
    features["address_jaccard"] = jaccard_similarity(s1_address, s2_address)
    features["address_levenshtein_ratio"] = levenshtein_ratio(s1_address, s2_address)
    features["address_token_overlap"] = token_overlap_ratio(s1_address, s2_address)
    features["address_component_sim"] = address_component_similarity(
        s1_address, s2_address, s1_country
    )
    
    # Length ratios (guard against wildly different lengths)
    s1_name_len = len(tokenize(s1_name))
    s2_name_len = len(tokenize(s2_name))
    if s1_name_len > 0 and s2_name_len > 0:
        features["name_length_ratio"] = min(s1_name_len, s2_name_len) / max(s1_name_len, s2_name_len)
    else:
        features["name_length_ratio"] = 0.0 if (s1_name_len > 0) != (s2_name_len > 0) else 1.0
    
    s1_addr_len = len(tokenize(s1_address))
    s2_addr_len = len(tokenize(s2_address))
    if s1_addr_len > 0 and s2_addr_len > 0:
        features["address_length_ratio"] = min(s1_addr_len, s2_addr_len) / max(s1_addr_len, s2_addr_len)
    else:
        features["address_length_ratio"] = 0.0 if (s1_addr_len > 0) != (s2_addr_len > 0) else 1.0
    
    # Combined signal
    features["combined_name_address"] = (
        0.6 * features["name_jaccard"] +
        0.4 * features["address_component_sim"]
    )
    
    return features


def features_to_vector(features: Dict[str, float], feature_names: list) -> list:
    """Convert feature dict to ordered vector for model inference."""
    return [features.get(name, 0.0) for name in feature_names]


# Standard feature order (used consistently in training and inference)
FEATURE_NAMES = [
    "country_match",
    "name_jaccard",
    "name_levenshtein_ratio",
    "name_token_overlap",
    "address_jaccard",
    "address_levenshtein_ratio",
    "address_token_overlap",
    "address_component_sim",
    "name_length_ratio",
    "address_length_ratio",
    "combined_name_address",
]
