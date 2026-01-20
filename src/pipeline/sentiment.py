"""
Sentiment Analysis Module using NLTK

This module provides functions for analyzing sentiment in financial news headlines
using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner).
"""

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

# Optional heavy imports (transformers / textblob) are lazy-initialized below
_transformer_pipeline = None
_transformer_model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
try:
    from transformers import pipeline as _hf_pipeline
except Exception:
    _hf_pipeline = None

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None


def setup_nltk_resources():
    """Download required NLTK data packages."""
    required_packages = ['vader_lexicon', 'punkt', 'stopwords']
    for package in required_packages:
        try:
            nltk.data.find(f'sentiment/{package}' if package == 'vader_lexicon' else f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)


def analyze_headline_sentiment(headline: str) -> Dict[str, float]:
    """
    Analyze sentiment of a single headline using NLTK VADER.
    
    Args:
        headline: News headline text
        
    Returns:
        Dictionary with sentiment scores:
        - neg: Negative sentiment (0-1)
        - neu: Neutral sentiment (0-1)
        - pos: Positive sentiment (0-1)
        - compound: Overall sentiment (-1 to 1)
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(headline)
    return scores


def batch_sentiment_analysis(headlines: pd.Series) -> pd.DataFrame:
    """
    Process multiple headlines efficiently.
    
    Args:
        headlines: Pandas Series of headline texts
        
    Returns:
        DataFrame with columns: neg, neu, pos, compound, sentiment_label
    """
    sia = SentimentIntensityAnalyzer()
    
    results = []
    for headline in headlines:
        if pd.isna(headline):
            results.append({'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0})
        else:
            results.append(sia.polarity_scores(str(headline)))
    
    df = pd.DataFrame(results)
    
    # Add sentiment label based on compound score
    df['sentiment_label'] = df['compound'].apply(
        lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
    )
    
    return df


def aggregate_daily_sentiment(df: pd.DataFrame, date_col: str = 'date', 
                              ticker_col: str = 'stock') -> pd.DataFrame:
    """
    Aggregate sentiment scores by date and ticker.
    
    Args:
        df: DataFrame with sentiment scores
        date_col: Name of date column
        ticker_col: Name of ticker column
        
    Returns:
        DataFrame with aggregated daily sentiment per ticker
    """
    # Group by date and ticker, calculate mean sentiment
    agg_dict = {
        'compound': ['mean', 'std', 'count'],
        'pos': 'mean',
        'neg': 'mean',
        'neu': 'mean'
    }
    
    aggregated = df.groupby([date_col, ticker_col]).agg(agg_dict).reset_index()
    
    # Flatten column names
    aggregated.columns = [
        f'{col[0]}_{col[1]}' if col[1] else col[0] 
        for col in aggregated.columns
    ]
    
    # Rename for clarity
    aggregated.rename(columns={
        'compound_mean': 'avg_sentiment',
        'compound_std': 'sentiment_std',
        'compound_count': 'news_count'
    }, inplace=True)
    
    return aggregated


def extract_sentiment_features(headline: str) -> Dict[str, any]:
    """
    Extract comprehensive features for ML models.
    
    Args:
        headline: News headline text
        
    Returns:
        Dictionary with sentiment scores and text features
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(headline)
    
    # Add text-based features
    features = sentiment.copy()
    features['headline_length'] = len(headline)
    features['word_count'] = len(headline.split())
    features['has_exclamation'] = 1 if '!' in headline else 0
    features['has_question'] = 1 if '?' in headline else 0
    features['is_uppercase'] = 1 if headline.isupper() else 0
    
    return features


def _init_transformer_pipeline(model_name: str = None):
    """Lazily initialize a Hugging Face sentiment pipeline.

    Returns None if transformers is not installed.
    """
    global _transformer_pipeline, _transformer_model_name
    if model_name:
        _transformer_model_name = model_name
    if _hf_pipeline is None:
        return None
    if _transformer_pipeline is None:
        try:
            _transformer_pipeline = _hf_pipeline('sentiment-analysis', model=_transformer_model_name)
        except Exception:
            _transformer_pipeline = None
    return _transformer_pipeline


def compute_sentiment(text: str, method: str = 'vader', transformer_model: str = None) -> Dict[str, any]:
    """Compute sentiment for a single text using the specified method.

    Methods supported: 'vader' (default), 'textblob', 'transformer'.
    Returns a dict with keys `score` (float) and `label` (str) plus raw details when available.
    """
    if text is None:
        return {'score': 0.0, 'label': 'neutral'}

    method = (method or 'vader').lower()
    text = str(text)

    if method == 'textblob' and TextBlob is not None:
        tb = TextBlob(text)
        polarity = tb.sentiment.polarity
        label = 'positive' if polarity > 0.05 else ('negative' if polarity < -0.05 else 'neutral')
        return {'score': float(polarity), 'label': label, 'method': 'textblob'}

    if method == 'transformer':
        pipe = _init_transformer_pipeline(transformer_model)
        if pipe is None:
            # fallback to vader
            return compute_sentiment(text, method='vader')
        try:
            out = pipe(text[:512])[0]
            # HF returns labels like 'POSITIVE' or 'NEGATIVE' and a score
            score_raw = float(out.get('score', 0.0))
            label_raw = out.get('label', '').lower()
            # Normalize label to positive/negative/neutral (no neutral in many models)
            label = 'positive' if 'pos' in label_raw else ('negative' if 'neg' in label_raw else 'neutral')
            # Map score to signed score where positive is >0
            score = score_raw if label == 'positive' else -score_raw if label == 'negative' else 0.0
            return {'score': score, 'label': label, 'method': 'transformer', 'raw': out}
        except Exception:
            return compute_sentiment(text, method='vader')

    # default: vader
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound = float(scores.get('compound', 0.0))
    label = 'positive' if compound >= 0.05 else ('negative' if compound <= -0.05 else 'neutral')
    return {'score': compound, 'label': label, 'method': 'vader', 'details': scores}


def batch_sentiment(df: pd.DataFrame, text_col: str = 'review_text', out_score_col: str = 'sentiment_score', out_label_col: str = 'sentiment_label', method: str = 'vader', transformer_model: str = None) -> pd.DataFrame:
    """Compute sentiment for a DataFrame column and attach score/label columns.

    Returns a copy of the DataFrame with new columns added.
    """
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in DataFrame")

    texts = df[text_col].fillna('').astype(str)

    results = [compute_sentiment(t, method=method, transformer_model=transformer_model) for t in texts]
    scores = [r.get('score', 0.0) for r in results]
    labels = [r.get('label', 'neutral') for r in results]

    out = df.copy()
    out[out_score_col] = scores
    out[out_label_col] = labels
    return out
