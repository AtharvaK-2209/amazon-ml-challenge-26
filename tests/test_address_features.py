"""
test_address_features.py — Unit tests for Address Features (Member 3)

Tests scenarios A through M as specified in the prompt:
A. Identical addresses
B. Slightly different addresses
C. Same business name but different cities
D. Same city but different businesses
E. Same postal/PIN code
F. Same house number
G. Missing address
H. Missing name
I. Both name and address missing
J. Different address word ordering
K. Common abbreviations (Road ↔ Rd, Street ↔ St)
L. Addresses containing punctuation/noise
M. Unicode/non-English text
"""

import pytest
import math
from src.features.address_features import (
    AddressFeatureExtractor,
    extract_address_features,
    jaccard_similarity,
    levenshtein_distance,
    numeric_match,
    numeric_token_overlap,
    street_type_match,
    city_match
)
from src.preprocessing.address_parser import parse_address


class TestAddressFeatures:

    @pytest.fixture
    def extractor(self):
        return AddressFeatureExtractor()

    # A. Identical addresses
    def test_identical_addresses(self, extractor):
        addr = "123 Main Street, Suite 100, New York, NY 10001"
        feats = extractor.extract_features(addr, addr, "US", "US")
        assert feats['address_similarity'] == 1.0
        assert feats['address_ratio'] == 1.0
        assert feats['address_wratio'] == 1.0
        assert feats['address_token_sort_ratio'] == 1.0
        assert feats['address_token_set_ratio'] == 1.0
        assert feats['same_house_number'] == 1.0
        assert feats['same_postal_code'] == 1.0
        assert feats['same_state'] == 1.0

    # B. Slightly different addresses
    def test_slightly_different_addresses(self, extractor):
        addr1 = "123 Main Street, New York, NY 10001"
        addr2 = "123 Main St, New York, NY 10001"
        feats = extractor.extract_features(addr1, addr2, "US", "US")
        assert feats['address_similarity'] > 0.8
        assert feats['same_house_number'] == 1.0
        assert feats['same_postal_code'] == 1.0

    # C. Same business name but different cities
    def test_same_name_different_cities(self, extractor):
        addr1 = "456 Market St, San Francisco, CA 94105"
        addr2 = "456 Market St, Seattle, WA 98101"
        feats = extractor.extract_features(addr1, addr2, "US", "US")
        assert feats['same_city'] == 0.0 or feats['same_postal_code'] == 0.0
        assert feats['same_state'] == 0.0

    # D. Same city but different businesses
    def test_same_city_different_businesses(self, extractor):
        addr1 = "100 Pine Street, Seattle, WA 98101"
        addr2 = "999 Oak Avenue, Seattle, WA 98101"
        feats = extractor.extract_features(addr1, addr2, "US", "US")
        assert feats['same_postal_code'] == 1.0
        assert feats['same_state'] == 1.0
        assert feats['same_house_number'] == 0.0

    # E. Same postal/PIN code
    def test_same_postal_code(self, extractor):
        addr1 = "Plot 42 MG Road, Bangalore 560001"
        addr2 = "Flat 10 Brigade Rd, Bangalore 560001"
        feats = extractor.extract_features(addr1, addr2, "INDIA", "INDIA")
        assert feats['same_postal_code'] == 1.0

    # F. Same house number
    def test_same_house_number(self, extractor):
        addr1 = "No 42 M.G. Road, Bangalore"
        addr2 = "House 42 Residency Road, Bangalore"
        feats = extractor.extract_features(addr1, addr2, "INDIA", "INDIA")
        assert feats['same_house_number'] == 1.0

    # G. Missing address
    def test_missing_address(self, extractor):
        addr1 = "123 Main Street"
        addr2 = ""
        feats = extractor.extract_features(addr1, addr2, "US", "US")
        assert feats['address_similarity'] == 0.0
        assert feats['address_ratio'] == 0.0
        assert feats['same_house_number'] == 0.0
        assert feats['same_postal_code'] == 0.0

    # H & I. None / empty / both missing
    def test_both_missing_address(self, extractor):
        feats = extractor.extract_features(None, "", "US", "US")
        assert feats['address_similarity'] == 0.0
        assert feats['numeric_token_overlap'] == 0.0

    # J. Different address word ordering
    def test_different_word_ordering(self, extractor):
        addr1 = "Suite 400 500 Fifth Avenue New York"
        addr2 = "500 Fifth Avenue Suite 400 New York"
        feats = extractor.extract_features(addr1, addr2, "US", "US")
        assert feats['address_token_sort_ratio'] == 1.0
        assert feats['address_token_set_ratio'] == 1.0

    # K. Common abbreviations (Road ↔ Rd, Street ↔ St)
    def test_common_abbreviations(self, extractor):
        addr1 = "100 Grand Road"
        addr2 = "100 Grand Rd"
        assert street_type_match(addr1, addr2) == True
        feats = extractor.extract_features(addr1, addr2)
        assert feats['address_street_type_match'] == 1.0

    # L. Addresses containing punctuation / noise
    def test_punctuation_and_noise(self, extractor):
        addr1 = "100-A, M.G. Road (Opp. Bank of India), City!!"
        addr2 = "100A MG Road Opp Bank of India City"
        feats = extractor.extract_features(addr1, addr2)
        assert feats['address_wratio'] > 0.8
        assert feats['numeric_token_overlap'] > 0.0

    # M. Unicode / non-English text
    def test_unicode_and_non_english(self, extractor):
        addr1 = "Rue de l'Église, 75001 Paris"
        addr2 = "Rue de l Eglise, 75001 Paris"
        feats = extractor.extract_features(addr1, addr2, "FRANCE", "FRANCE")
        assert feats['address_similarity'] > 0.85
        assert feats['same_postal_code'] == 1.0

    # Helper sanity checks
    def test_no_nan_or_inf_in_output(self, extractor):
        feats = extractor.extract_features("123 Test St", "456 Other Ave", "US", "US")
        for k, v in feats.items():
            assert not math.isnan(v), f"NaN found in feature {k}"
            assert not math.isinf(v), f"Inf found in feature {k}"
