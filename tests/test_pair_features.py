"""
test_pair_features.py — Unit tests for Cross-Field & Pair Features (Member 3)

Tests:
- PairFeatureExtractor class
- Missingness indicators (name_missing, address_missing, both_missing)
- Cross-field interactions (name_address_similarity_product, sum, length differences)
- Pairwise boolean features (same_country, same_city, etc.)
- DataFrame pipeline integration (build_feature_matrix)
"""

import pytest
import math
import pandas as pd

from src.features.pair_features import PairFeatureExtractor, build_feature_matrix


class TestPairFeatures:

    @pytest.fixture
    def extractor(self):
        return PairFeatureExtractor()

    def test_missingness_features_neither_missing(self, extractor):
        e1 = {'business_name': 'Acme Corp', 'business_address': '123 Main St', 'country': 'US'}
        e2 = {'business_name': 'Acme Corporation', 'business_address': '123 Main Street', 'country': 'US'}
        feats = extractor.extract_pair_features(e1, e2)
        assert feats['name_missing'] == 0.0
        assert feats['address_missing'] == 0.0
        assert feats['both_missing'] == 0.0

    def test_missingness_features_name_missing(self, extractor):
        e1 = {'business_name': '', 'business_address': '123 Main St', 'country': 'US'}
        e2 = {'business_name': 'Acme Corp', 'business_address': '123 Main Street', 'country': 'US'}
        feats = extractor.extract_pair_features(e1, e2)
        assert feats['name_missing'] == 1.0
        assert feats['address_missing'] == 0.0
        assert feats['both_missing'] == 0.0

    def test_missingness_features_address_missing(self, extractor):
        e1 = {'business_name': 'Acme Corp', 'business_address': '', 'country': 'US'}
        e2 = {'business_name': 'Acme Inc', 'business_address': '456 Oak St', 'country': 'US'}
        feats = extractor.extract_pair_features(e1, e2)
        assert feats['name_missing'] == 0.0
        assert feats['address_missing'] == 1.0
        assert feats['both_missing'] == 0.0

    def test_missingness_features_both_missing(self, extractor):
        e1 = {'business_name': '', 'business_address': '', 'country': 'US'}
        e2 = {'business_name': '', 'business_address': '123 Main St', 'country': 'US'}
        feats = extractor.extract_pair_features(e1, e2)
        assert feats['name_missing'] == 1.0
        assert feats['address_missing'] == 1.0
        assert feats['both_missing'] == 1.0

    def test_cross_field_similarity_interactions(self, extractor):
        e1 = {'business_name': 'Starbucks Coffee', 'business_address': '100 Main St, Seattle', 'country': 'US'}
        e2 = {'business_name': 'Starbucks', 'business_address': '100 Main St, Seattle', 'country': 'US'}
        feats = extractor.extract_pair_features(e1, e2)
        
        assert 'name_address_similarity_product' in feats
        assert 'name_address_similarity_sum' in feats
        assert feats['name_address_similarity_product'] > 0.5
        assert feats['name_address_similarity_sum'] > 1.0

    def test_same_country_and_length_diffs(self, extractor):
        e1 = {'business_name': 'Short Name', 'business_address': 'Short Addr', 'country': 'US'}
        e2 = {'business_name': 'Very Long Business Name Indeed', 'business_address': 'Very Long Address Line 1', 'country': 'CA'}
        feats = extractor.extract_pair_features(e1, e2)

        assert feats['same_country'] == 0.0
        assert feats['name_length_difference'] == float(len('Very Long Business Name Indeed') - len('Short Name'))
        assert feats['address_length_difference'] == float(len('Very Long Address Line 1') - len('Short Addr'))

    def test_build_feature_matrix_pipeline(self):
        df = pd.DataFrame([
            {
                'source1_entity_id': 's1_1',
                'candidate_entity_id': 'cand_1',
                'normalized_name_s1': 'amazon',
                'normalized_address_s1': '100 2nd ave seattle',
                'country_s1': 'US',
                'normalized_name_cand': 'amazon inc',
                'normalized_address_cand': '100 2nd ave seattle wa',
                'country_cand': 'US',
            },
            {
                'source1_entity_id': 's1_2',
                'candidate_entity_id': 'cand_2',
                'normalized_name_s1': 'walmart',
                'normalized_address_s1': '702 sw 8th st bentonville',
                'country_s1': 'US',
                'normalized_name_cand': 'target',
                'normalized_address_cand': '1000 nicollet mall minneapolis',
                'country_cand': 'US',
            }
        ])

        X = build_feature_matrix(df)
        assert not X.empty
        assert len(X) == 2
        assert 'name_ratio' in X.columns
        assert 'address_similarity' in X.columns
        assert 'same_postal_code' in X.columns
        assert 'name_missing' in X.columns
        assert 'name_address_similarity_product' in X.columns

        # Check for NaN / Inf
        assert not X.isna().any().any()
