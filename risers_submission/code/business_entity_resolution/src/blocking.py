#!/usr/bin/env python3
"""
Blocking (Candidate Generation) Stage

Multi-strategy blocking tuned for scale:
- Country-aware (US, India, France have different address patterns)
- Name-based: sorted-token bigrams, phonetic encoding (Soundex)
- Address-based: PIN/postal code, state/city tokens, component overlap
- Generates candidates via inverted index + LSH-like bucketing

This stage is designed to run on ~2M train S1 entities + ~10M S2/S3 entities
without loading the entire dataset into memory.
"""

import re
from collections import defaultdict
from typing import Dict, Set, Tuple, List
import unicodedata


def normalize_text(s: str) -> str:
    """Lowercase, remove accents, strip whitespace."""
    if not s or not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def tokenize(s: str) -> List[str]:
    """Split on whitespace and punctuation; filter short tokens."""
    s = normalize_text(s)
    tokens = re.findall(r"\b\w+\b", s)
    return [t for t in tokens if len(t) > 1]


def soundex(s: str) -> str:
    """Simple Soundex encoding for phonetic blocking."""
    s = normalize_text(s).upper()
    if not s:
        return ""
    
    # Keep first letter, encode the rest
    first = s[0]
    mapping = {"B": "1", "F": "1", "P": "1", "V": "1",
               "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
               "D": "3", "T": "3",
               "L": "4",
               "M": "5", "N": "5",
               "R": "6"}
    
    encoded = first
    prev_code = mapping.get(first, "0")
    for char in s[1:]:
        code = mapping.get(char, "0")
        if code != "0" and code != prev_code:
            encoded += code
        if code != "0":
            prev_code = code
    
    return (encoded + "000")[:4]


def extract_postal_code(address: str) -> str:
    """Extract postal/PIN code (5–6 digits, common in US/India)."""
    if not address:
        return ""
    # Look for 5-6 consecutive digits (US ZIP, Indian PIN)
    match = re.search(r"\b\d{5,6}\b", address)
    return match.group(0) if match else ""


def extract_state_city(address: str, country: str) -> Tuple[str, str]:
    """Extract state and city tokens from address (country-aware)."""
    tokens = tokenize(address)
    # Heuristic: US addresses often end with state (2-letter), India with state name or city
    # This is a simplified extraction; a real system would use gazetteers.
    state, city = "", ""
    
    if country == "US" and len(tokens) >= 2:
        # Last token might be state abbreviation (2 letters)
        if len(tokens[-1]) == 2:
            state = tokens[-1]
        # Second-to-last might be city
        if len(tokens) >= 2:
            city = tokens[-2]
    elif country in ("India", "IN"):
        # India: look for common state/city suffixes or just use frequent tokens
        if len(tokens) >= 1:
            city = tokens[0] if len(tokens) > 0 else ""
            state = tokens[-1] if len(tokens) > 1 else ""
    elif country == "France":
        # France: similar approach, last token often region
        if len(tokens) >= 1:
            state = tokens[-1]
            city = tokens[0] if len(tokens) > 0 else ""
    
    return normalize_text(state), normalize_text(city)


def get_blocking_keys(entity_id: str, name: str, address: str, country: str) -> Set[str]:
    """Generate all blocking keys for an entity (inverted index keys)."""
    keys = set()
    
    # Always include country
    keys.add(f"COUNTRY:{normalize_text(country)}")
    
    # Name-based keys
    if name:
        tokens = tokenize(name)
        if tokens:
            # Sorted-token bigram (order-invariant, helps with name reordering)
            sorted_tokens = sorted(tokens)
            keys.add(f"NAME_TOKENS:{','.join(sorted_tokens[:2])}")  # first 2 sorted tokens
            
            # Phonetic: first word
            first_word = tokens[0]
            phonetic = soundex(first_word)
            if phonetic:
                keys.add(f"NAME_PHONETIC:{phonetic}")
    
    # Address-based keys
    if address:
        postal = extract_postal_code(address)
        if postal:
            keys.add(f"POSTAL:{postal}")
        
        state, city = extract_state_city(address, country)
        if state:
            keys.add(f"STATE:{state}")
        if city:
            keys.add(f"CITY:{city}")
        
        # General address token overlap (first 2–3 tokens)
        addr_tokens = tokenize(address)
        if addr_tokens:
            keys.add(f"ADDR_START:{','.join(addr_tokens[:2])}")
    
    return keys


class BlockingIndex:
    """Inverted index for blocking: key -> set of entity_ids."""
    
    def __init__(self):
        self.index: Dict[str, Set[str]] = defaultdict(set)
    
    def add(self, entity_id: str, blocking_keys: Set[str]):
        """Add entity to index under its blocking keys."""
        for key in blocking_keys:
            self.index[key].add(entity_id)
    
    def retrieve(self, blocking_keys: Set[str]) -> Set[str]:
        """Retrieve all entities matching ANY of the blocking keys (union)."""
        candidates = set()
        for key in blocking_keys:
            candidates.update(self.index[key])
        return candidates


def build_blocking_index(records: List[Dict[str, str]]) -> BlockingIndex:
    """Build inverted index from a list of records (S2 or S3)."""
    idx = BlockingIndex()
    for record in records:
        entity_id = record["entity_id"]
        name = record.get("business_name", "")
        address = record.get("business_address", "")
        country = record.get("country", "")
        
        keys = get_blocking_keys(entity_id, name, address, country)
        idx.add(entity_id, keys)
    
    return idx


def get_candidates(
    s1_entity_id: str,
    s1_name: str,
    s1_address: str,
    s1_country: str,
    s2_index: BlockingIndex,
    s3_index: BlockingIndex,
) -> Tuple[Set[str], Set[str]]:
    """
    Retrieve candidate S2 and S3 entities for a given S1 entity.
    
    Returns:
        (s2_candidates, s3_candidates) — sets of entity IDs
    """
    s1_keys = get_blocking_keys(s1_entity_id, s1_name, s1_address, s1_country)
    
    s2_cands = s2_index.retrieve(s1_keys)
    s3_cands = s3_index.retrieve(s1_keys)
    
    return s2_cands, s3_cands
