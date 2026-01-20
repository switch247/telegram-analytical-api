import pytest
from src.pipeline import sentiment as sent


def test_compute_sentiment_vader_basic():
    res = sent.compute_sentiment("I love this app. It's great!", method='vader')
    assert 'score' in res
    assert 'label' in res


def test_batch_sentiment_adds_columns():
    import pandas as pd
    df = pd.DataFrame({'review_text': ['Good', 'Bad']})
    out = sent.batch_sentiment(df, text_col='review_text')
    assert 'sentiment_score' in out.columns
    assert 'sentiment_label' in out.columns
