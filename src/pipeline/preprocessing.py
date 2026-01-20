"""NLP preprocessing helpers: tokenization, stop-word removal, lemmatization.

Provides functions to preprocess single texts or a DataFrame column and
produce a cleaned, lemmatized string suitable for vectorizers or sentiment
analysis. Uses NLTK and performs lazy resource downloads when needed.
"""
from typing import List, Callable, Optional
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag

_nlp_setup_done = False


def setup_nlp_resources():
    global _nlp_setup_done
    if _nlp_setup_done:
        return
    required = ['punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'stopwords']
    for pkg in required:
        try:
            nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)
    _nlp_setup_done = True


def _pos_tag_to_wordnet(tag: str) -> str:
    """Map POS tag to WordNet lemmatizer POS."""
    if tag.startswith('J'):
        return 'a'
    if tag.startswith('V'):
        return 'v'
    if tag.startswith('N'):
        return 'n'
    if tag.startswith('R'):
        return 'r'
    return 'n'


def preprocess_text(text: str, min_token_len: int = 2, extra_stopwords: Optional[List[str]] = None) -> str:
    """Tokenize, remove stopwords/punctuation, lemmatize and return cleaned string.

    Args:
        text: input string
        min_token_len: minimum token length to keep
        extra_stopwords: list of additional stopwords to remove

    Returns:
        Cleaned string of space-separated lemmas (lowercased).
    """
    if text is None:
        return ''
    setup_nlp_resources()

    text = str(text)
    # Basic cleanup
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\w\s'-]", ' ', text)
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.isalpha()]

    stop_words = set(stopwords.words('english'))
    if extra_stopwords:
        stop_words.update(extra_stopwords)

    tokens = [t for t in tokens if t not in stop_words and len(t) >= min_token_len]

    # Lemmatize with POS
    lemmatizer = WordNetLemmatizer()
    tagged = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(tok, _pos_tag_to_wordnet(tag)) for tok, tag in tagged]

    return ' '.join(lemmas)


def preprocess_dataframe(df, text_col: str = 'review_text', out_col: str = 'review_text_preprocessed', inplace: bool = False, **kwargs):
    """Preprocess a DataFrame column and add an output column with cleaned text.

    Returns the modified DataFrame (copy unless inplace=True).
    """
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in DataFrame")
    target_df = df if inplace else df.copy()
    target_df[out_col] = target_df[text_col].fillna('').astype(str).apply(lambda t: preprocess_text(t, **kwargs))
    return target_df
