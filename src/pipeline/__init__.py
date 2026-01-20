# Pipeline module exports
# Import what's actually available in the modules

try:
    from .sentiment import (
        setup_nltk_resources,
        analyze_headline_sentiment,
        batch_sentiment_analysis,
        aggregate_daily_sentiment,
        extract_sentiment_features
    )
except ImportError:
    pass

try:
    from .text_analysis import TextAnalyzer
except ImportError:
    TextAnalyzer = None

try:
    from .stock_metrics import calculate_stock_metrics
except ImportError:
    calculate_stock_metrics = None
