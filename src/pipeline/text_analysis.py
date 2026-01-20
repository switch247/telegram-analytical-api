import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from collections import Counter
import re
from typing import List, Tuple, Dict

def get_common_phrases(df: pd.DataFrame, column: str, n: int = 10, ngram_range: Tuple[int, int] = (1, 1)) -> List[Tuple[str, int]]:
    """
    Get the most common words or phrases (n-grams) in a text column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Text column name.
        n (int): Number of top phrases to return.
        ngram_range (Tuple[int, int]): Range of n-grams (e.g., (1, 1) for unigrams, (2, 2) for bigrams).

    Returns:
        List[Tuple[str, int]]: List of (phrase, count) tuples.
    """
    if column not in df.columns:
        return []

    # Drop NaNs and convert to string
    text_data = df[column].dropna().astype(str)
    
    if text_data.empty:
        return []

    # Use CountVectorizer for efficient n-gram counting
    # stop_words='english' removes common English stop words
    vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range, max_features=10000)
    
    try:
        X = vectorizer.fit_transform(text_data)
        counts = X.sum(axis=0).A1
        vocab = vectorizer.get_feature_names_out()
        
        # Combine counts and vocab
        freq_dist = list(zip(vocab, counts))
        # Sort by count descending
        freq_dist.sort(key=lambda x: x[1], reverse=True)
        
        return freq_dist[:n]
    except ValueError:
        # Handle case with empty vocabulary (e.g., all stop words)
        return []

def perform_topic_modeling(df: pd.DataFrame, column: str, n_topics: int = 5, n_top_words: int = 10) -> Dict[int, List[str]]:
    """
    Perform topic modeling using NMF (Non-Negative Matrix Factorization).

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Text column name.
        n_topics (int): Number of topics to extract.
        n_top_words (int): Number of top words to return per topic.

    Returns:
        Dict[int, List[str]]: Dictionary mapping topic index to list of top words.
    """
    if column not in df.columns:
        return {}

    text_data = df[column].dropna().astype(str)
    
    if text_data.empty:
        return {}

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(text_data)

    # NMF
    nmf = NMF(n_components=n_topics, random_state=42, init='nndsvd').fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    topics = {}
    for topic_idx, topic in enumerate(nmf.components_):
        top_words = [tfidf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics[topic_idx] = top_words

    return topics


class TextAnalyzer:
    """Convenience wrapper for keyword extraction and topic/theme extraction.

    Uses TF-IDF + NMF for topic modeling and CountVectorizer for keyword extraction.
    """
    def __init__(self, stop_words: str = 'english'):
        self.stop_words = stop_words
        self.tfidf_vectorizer = None
        self.nmf_model = None

    def extract_keywords(self, texts: List[str], top_n: int = 10, ngram_range: Tuple[int, int] = (1, 2)) -> List[Tuple[str, int]]:
        """Return top `top_n` keywords across a list of texts."""
        if not texts:
            return []
        vec = CountVectorizer(stop_words=self.stop_words, ngram_range=ngram_range, max_features=10000)
        X = vec.fit_transform([str(t) for t in texts if t])
        counts = X.sum(axis=0).A1
        vocab = vec.get_feature_names_out()
        freq = list(zip(vocab, counts))
        freq.sort(key=lambda x: x[1], reverse=True)
        return freq[:top_n]

    def fit_topic_model(self, texts: List[str], n_topics: int = 5, n_top_words: int = 10) -> Dict[int, List[str]]:
        """Fit an NMF topic model on texts and return top words per topic."""
        texts = [str(t) for t in texts if t]
        if not texts:
            return {}
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=self.stop_words)
        tfidf = self.tfidf_vectorizer.fit_transform(texts)
        self.nmf_model = NMF(n_components=n_topics, random_state=42, init='nndsvd')
        nmf = self.nmf_model.fit(tfidf)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(nmf.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics[topic_idx] = top_words
        return topics

    def get_themes_by_bank(self, df: pd.DataFrame, text_col: str = 'review_text', bank_col: str = 'bank_name', n_themes: int = 3, n_top_words: int = 8) -> Dict[str, List[List[str]]]:
        """Return top `n_themes` topics per bank as lists of keywords.

        Output: {bank_name: [[topic1_terms], [topic2_terms], ...]}
        """
        if bank_col not in df.columns or text_col not in df.columns:
            return {}
        themes = {}
        for bank, grp in df.groupby(bank_col):
            texts = grp[text_col].dropna().astype(str).tolist()
            if not texts:
                themes[bank] = []
                continue
            # Fit a small topic model per bank (n_themes topics)
            try:
                topics = self.fit_topic_model(texts, n_topics=n_themes, n_top_words=n_top_words)
                # convert to list of lists
                themes[bank] = [topics[t] for t in sorted(topics.keys())]
            except Exception:
                # Fallback to keywords if topic modeling fails
                kws = self.extract_keywords(texts, top_n=n_themes, ngram_range=(1,2))
                themes[bank] = [[k for k,_ in kws[i:i+1]] for i in range(min(n_themes, len(kws)))]
        return themes
