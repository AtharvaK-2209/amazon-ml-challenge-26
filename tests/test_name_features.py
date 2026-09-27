import pytest
import math
from src.features.name_features import generate_name_features
from sklearn.feature_extraction.text import TfidfVectorizer

@pytest.fixture
def word_vec():
    # Fit a simple dummy vocabulary
    v = TfidfVectorizer(analyzer="word")
    v.fit(["abc retail limited private inc sons technologies"])
    return v

@pytest.fixture
def char_vec():
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3))
    v.fit(["abc retail limited private inc sons technologies"])
    return v

def test_identical_names(word_vec, char_vec):
    feats = generate_name_features("abc retail", "abc retail", word_vec, char_vec)
    assert feats["name_levenshtein"] == 0
    assert feats["name_fuzz_ratio"] == 100.0
    assert feats["name_wratio"] == 100.0
    assert feats["name_token_sort_ratio"] == 100.0
    assert feats["name_token_set_ratio"] == 100.0
    assert feats["name_jaccard"] == 1.0
    assert feats["name_length_diff"] == 0.0
    assert feats["name_length_ratio"] == 1.0
    assert feats["name_token_count_diff"] == 0.0
    assert feats["name_tfidf_cosine"] > 0.99
    assert feats["name_char_cosine"] > 0.99

def test_ltd_limited():
    feats = generate_name_features("abc retail ltd", "abc retail limited")
    # Jaccard: intersection {"abc", "retail"}, union {"abc", "retail", "ltd", "limited"} => 2 / 4 = 0.5
    assert feats["name_jaccard"] == 0.5
    assert feats["name_token_count_diff"] == 0
    assert feats["name_length_diff"] > 0

def test_pvt_private():
    feats = generate_name_features("abc pvt ltd", "abc private limited")
    assert feats["name_jaccard"] == 0.2 # {"abc"} out of 5 unique tokens

def test_ampersand():
    feats = generate_name_features("abc & sons", "abc and sons")
    assert feats["name_jaccard"] == 0.5 # {"abc", "sons"} out of 4 unique
    assert feats["name_fuzz_ratio"] < 100.0

def test_typos():
    feats = generate_name_features("abc technologies", "abc technolgies")
    assert feats["name_levenshtein"] == 1
    assert feats["name_fuzz_ratio"] > 90.0

def test_word_order():
    feats = generate_name_features("abc retail services", "retail services abc")
    assert feats["name_fuzz_ratio"] < 100.0
    assert feats["name_token_sort_ratio"] == 100.0
    assert feats["name_token_set_ratio"] == 100.0
    assert feats["name_jaccard"] == 1.0

def test_empty_strings():
    feats = generate_name_features("", "")
    assert feats["name_levenshtein"] == 0
    assert feats["name_fuzz_ratio"] == 100.0
    assert feats["name_jaccard"] == 1.0
    assert feats["name_length_ratio"] == 1.0

def test_none_values():
    feats = generate_name_features(None, "abc")
    assert feats["name_levenshtein"] == 3
    assert feats["name_fuzz_ratio"] == 0.0
    assert feats["name_jaccard"] == 0.0
    assert feats["name_length_ratio"] == 0.0
    assert feats["name_length_diff"] == 3

def test_nan_values():
    feats = generate_name_features(float('nan'), "abc")
    assert feats["name_fuzz_ratio"] == 0.0
    assert feats["name_jaccard"] == 0.0

def test_names_with_numbers():
    feats = generate_name_features("3m company", "3m co")
    assert feats["name_levenshtein"] > 0
    assert feats["name_jaccard"] == 1/3 # {"3m"} intersection

def test_completely_different():
    feats = generate_name_features("apple inc", "banana corp")
    assert feats["name_fuzz_ratio"] < 40.0
    assert feats["name_jaccard"] == 0.0
    assert feats["name_token_sort_ratio"] < 40.0
