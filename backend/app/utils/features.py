"""
Feature engineering for priority prediction.

This module extracts features from tweet text to be used by the ML classifier.
The same functions will be imported and used in the backend to ensure
feature consistency between training and inference.

Features extracted:
- Basic counts: text length, word count, average word length
- Urgency keyword flags (binary)
- Punctuation signals: exclamation count, question count, multiple exclamation flag
- ALL CAPS ratio
- Sentiment polarity and subjectivity (using TextBlob, optional)
"""

import re
from typing import Dict, List, Union
import pandas as pd

# Optional sentiment analysis
try:
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("Warning: TextBlob not installed. Sentiment features will be zeros.")


# Urgency keywords (same as in labeling)
URGENT_KEYWORDS = [
    'refund', 'cancel', 'broken', 'help', 'urgent',
    'asap', 'stuck', 'issue', 'error', 'fail', 'wrong'
]


def extract_features_single(text: str) -> Dict[str, Union[int, float]]:
    """
    Extract features from a single text string.

    Args:
        text (str): Tweet text.

    Returns:
        Dict: Feature name to value mapping.
    """
    if not isinstance(text, str):
        text = ""

    text_lower = text.lower()
    words = text.split()
    num_words = len(words)

    # Basic length features
    features = {
        'text_length': len(text),
        'word_count': num_words,
        'avg_word_length': sum(len(w) for w in words) / max(num_words, 1),
    }

    # Keyword presence (binary)
    for kw in URGENT_KEYWORDS:
        features[f'has_{kw}'] = 1 if kw in text_lower else 0

    # Punctuation features
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['has_multiple_exclamations'] = 1 if features['exclamation_count'] >= 3 else 0

    # ALL CAPS ratio
    if num_words > 0:
        caps_words = sum(1 for w in words if w.isupper())
        features['caps_ratio'] = caps_words / num_words
    else:
        features['caps_ratio'] = 0.0

    # Sentiment features
    if SENTIMENT_AVAILABLE:
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        features['is_negative'] = 1 if features['sentiment_polarity'] < -0.2 else 0
    else:
        features['sentiment_polarity'] = 0.0
        features['sentiment_subjectivity'] = 0.0
        features['is_negative'] = 0

    return features


def extract_features(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Extract features for an entire DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing text column.
        text_column (str): Name of column with text.

    Returns:
        pd.DataFrame: Feature matrix.
    """
    features_list = df[text_column].apply(extract_features_single).tolist()
    return pd.DataFrame(features_list)


def get_feature_names() -> List[str]:
    """Return list of all feature names in correct order."""
    sample = extract_features_single("sample text")
    return list(sample.keys())


def get_numeric_feature_names() -> List[str]:
    """
    Return names of numeric features that should be imputed and scaled.
    
    Binary/categorical features (like keyword flags) are excluded because
    they don't benefit from scaling and imputation.
    """
    # Features that are binary (0/1) or counts that don't need scaling
    # (though you could scale counts – it's a choice)
    binary_features = [f'has_{kw}' for kw in URGENT_KEYWORDS]
    binary_features += ['has_multiple_exclamations', 'is_negative']
    
    all_features = get_feature_names()
    numeric_features = [
        f for f in all_features 
        if f not in binary_features
    ]
    return numeric_features