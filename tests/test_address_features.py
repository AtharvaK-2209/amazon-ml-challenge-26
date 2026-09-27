"""
test_address_features.py — Phase 4 Part 2 Unit Tests for Address Features

Tests scenarios A through T as required by Phase 4 specifications:
A. Exact same address
B. Minor formatting differences
C. Different house numbers
D. Same house number but different street
E. Same postal code
F. Different postal code
G. Same city
H. Different city
I. Same state
J. Different state
K. Missing address
L. Empty address
M. Missing postal code
N. Missing city
O. Missing state
P. Addresses with no numbers
Q. Addresses containing multiple numbers
R. Minor spelling errors
S. Different token ordering
T. Unicode/non-English addresses
"""

import pytest
import math
import numpy as np

from src.features.address_features import (
    AddressFeatureExtractor,
    extract_address_features,
    jaccard_similarity,
    levenshtein_distance,
    numeric_match,
    numeric_token_overlap,
    street_type_match,
    city_match,
    state_match,
    postal_match,
    house_number_match
)


class TestAddressFeaturesPhase4:

    @pytest.fixture
    def extractor(self):
        return AddressFeatureExtractor()

    # A. Exact same address
    def test_exact_same_address(self, extractor):
        addr = "1795 Westchester Drive, High Point, NC 27262"
        feats = extractor.extract_features(addr, addr, "US", "US")
        assert feats['address_fuzz_ratio'] == 1.0
        assert feats['address_wratio'] == 1.0
        assert feats['address_token_sort_ratio'] == 1.0
        assert feats['address_token_set_ratio'] == 1.0
        assert feats['address_jaccard'] == 1.0
        assert feats['address_tfidf_cosine'] == pytest.approx(1.0, abs=1e-3)
        assert feats['address_char_cosine'] == pytest.approx(1.0, abs=1e-3)
        assert feats['house_number_match'] == 1
        assert feats['postal_match'] == 1
        assert feats['city_match'] == 1
        assert feats['state_match'] == 1

    # B. Minor formatting differences (1795 Westchester Drive High Point NC vs 1795 Westchester Dr High Point NC)
    def test_minor_formatting_differences(self, extractor):
        addr1 = "1795 Westchester Drive High Point NC 27262"
        addr2 = "1795 Westchester Dr High Point NC 27262"
        feats = extractor.extract_features(addr1, addr2, "US", "US")
        assert feats['address_wratio'] > 0.85
        assert feats['address_char_cosine'] > 0.80
        assert feats['house_number_match'] == 1
        assert feats['postal_match'] == 1
        assert feats['state_match'] == 1

    # C. Different house numbers
    def test_different_house_numbers(self, extractor):
        addr1 = "1795 Westchester Drive, High Point, NC"
        addr2 = "1800 Westchester Drive, High Point, NC"
        feats = extractor.extract_features(addr1, addr2, "US", "US")
        assert feats['house_number_match'] == 0
        assert feats['address_numeric_overlap'] == 0.0

    # D. Same house number but different street
    def test_same_house_different_street(self, extractor):
        addr1 = "1795 Westchester Drive, High Point, NC"
        addr2 = "1795 Market Street, High Point, NC"
        feats = extractor.extract_features(addr1, addr2, "US", "US")
        assert feats['house_number_match'] == 1
        assert feats['address_token_set_ratio'] < 0.90
        assert feats['address_jaccard'] < 0.80

    # E & F. Same vs Different postal code
    def test_postal_code_match_and_mismatch(self, extractor):
        addr1 = "100 Main St, Seattle, WA 98101"
        addr2 = "200 Pine St, Seattle, WA 98101"
        addr3 = "100 Main St, Seattle, WA 98109"
        
        feats_match = extractor.extract_features(addr1, addr2, "US", "US")
        feats_mismatch = extractor.extract_features(addr1, addr3, "US", "US")
        
        assert feats_match['postal_match'] == 1
        assert feats_mismatch['postal_match'] == 0

    # G & H. Same vs Different city
    def test_city_match_and_mismatch(self, extractor):
        addr_fr1 = "5 Rue de Rivoli, 75001 Paris"
        addr_fr2 = "10 Rue de Rivoli, 75001 Paris"
        addr_fr3 = "5 Rue de la Paix, 69001 Lyon"
        
        feats_match = extractor.extract_features(addr_fr1, addr_fr2, "FRANCE", "FRANCE")
        feats_mismatch = extractor.extract_features(addr_fr1, addr_fr3, "FRANCE", "FRANCE")
        
        assert feats_match['city_match'] == 1
        assert feats_mismatch['city_match'] == 0

    # I & J. Same vs Different state
    def test_state_match_and_mismatch(self, extractor):
        addr1 = "100 Main St, Los Angeles, CA 90001"
        addr2 = "200 Market St, San Francisco, CA 94105"
        addr3 = "300 Pine St, Seattle, WA 98101"
        
        feats_match = extractor.extract_features(addr1, addr2, "US", "US")
        feats_mismatch = extractor.extract_features(addr1, addr3, "US", "US")
        
        assert feats_match['state_match'] == 1
        assert feats_mismatch['state_match'] == 0

    # K & L. Missing / Empty address
    def test_missing_and_empty_address(self, extractor):
        addr1 = "123 Main St, New York, NY"
        addr2 = None
        addr3 = ""
        
        feats_none = extractor.extract_features(addr1, addr2, "US", "US")
        feats_empty = extractor.extract_features(addr3, addr1, "US", "US")
        
        assert feats_none['address_fuzz_ratio'] == 0.0
        assert feats_none['address_wratio'] == 0.0
        assert feats_none['postal_match'] == -1
        assert feats_none['city_match'] == -1
        assert feats_none['state_match'] == -1
        assert feats_none['house_number_match'] == -1
        
        assert feats_empty['address_fuzz_ratio'] == 0.0
        assert feats_empty['postal_match'] == -1

    # M, N, O. Missing postal code, city, state
    def test_missing_address_components(self, extractor):
        addr1 = "Westchester Drive"
        addr2 = "Westchester Drive"
        
        feats = extractor.extract_features(addr1, addr2, "US", "US")
        assert feats['postal_match'] == -1
        assert feats['city_match'] == -1
        assert feats['state_match'] == -1
        assert feats['house_number_match'] == -1

    # P. Addresses with no numbers
    def test_addresses_with_no_numbers(self, extractor):
        addr1 = "Westchester Drive High Point"
        addr2 = "Westchester Dr High Point"
        feats = extractor.extract_features(addr1, addr2)
        assert feats['address_numeric_overlap'] == 0.0
        assert feats['house_number_match'] == -1

    # Q. Addresses containing multiple numbers
    def test_addresses_multiple_numbers(self, extractor):
        addr1 = "Flat 4B Building 12 Plot 99 MG Road 560001"
        addr2 = "Flat 4B Building 12 Plot 99 MG Road 560001"
        feats = extractor.extract_features(addr1, addr2, "INDIA", "INDIA")
        assert feats['address_numeric_overlap'] == 1.0
        assert feats['postal_match'] == 1

    # R. Minor spelling errors
    def test_minor_spelling_errors(self, extractor):
        addr1 = "1795 Westchestr Drive High Point"
        addr2 = "1795 Westchester Drive High Point"
        feats = extractor.extract_features(addr1, addr2)
        assert feats['address_wratio'] > 0.90
        assert feats['address_char_cosine'] > 0.85

    # S. Different token ordering
    def test_different_token_ordering(self, extractor):
        addr1 = "Suite 400 500 Fifth Avenue New York"
        addr2 = "500 Fifth Avenue Suite 400 New York"
        feats = extractor.extract_features(addr1, addr2)
        assert feats['address_token_sort_ratio'] == 1.0
        assert feats['address_token_set_ratio'] == 1.0

    # T. Unicode / non-English text
    def test_unicode_non_english(self, extractor):
        addr1 = "Rue de l'Église, 75001 Paris"
        addr2 = "Rue de l Eglise, 75001 Paris"
        feats = extractor.extract_features(addr1, addr2, "FRANCE", "FRANCE")
        assert feats['address_wratio'] > 0.85
        assert feats['postal_match'] == 1
        assert feats['city_match'] == 1

    # Quality and boundary assertions
    def test_no_nan_or_inf_and_correct_ranges(self, extractor):
        test_pairs = [
            ("123 Main St", "456 Oak Ave"),
            (None, "123 Main St"),
            ("", ""),
            ("Plot 42, Road #10, Apt 3B", "Plot 42, Road 10"),
        ]
        for a1, a2 in test_pairs:
            feats = extractor.extract_features(a1, a2)
            for k, v in feats.items():
                assert not math.isnan(v), f"NaN found in feature {k} for ({a1}, {a2})"
                assert not math.isinf(v), f"Inf found in feature {k} for ({a1}, {a2})"
                
                # Check range conventions
                if k in ['house_number_match', 'postal_match', 'city_match', 'state_match']:
                    assert v in [-1.0, 0.0, 1.0], f"Invalid flag value {v} for {k}"
                elif 'levenshtein' not in k and 'length' not in k:
                    assert 0.0 <= v <= 1.0, f"Out-of-range value {v} for {k}"
