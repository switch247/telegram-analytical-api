"""Utilities package - imports are optional to avoid heavy import-time requirements.

This file prefers lazy/guarded imports so that importing the package in CI
or tests does not require heavy optional dependencies (e.g., google-play-scraper,
transformers, or large language models).
"""

try:
	from .scraper import PlayStoreScraper
except Exception:
	PlayStoreScraper = None

try:
	from ..preprocessing.preprocessor import DatasetPreprocessor
except Exception:
	DatasetPreprocessor = None

# Backwards compatibility alias
ReviewPreprocessor = DatasetPreprocessor

try:
	from .data_base_loader import DatabaseLoader
except Exception:
	DatabaseLoader = None

try:
	from ..preprocessing.data_cleaning import convert_to_datetime, handle_missing_values, remove_duplicates
except Exception:
	convert_to_datetime = handle_missing_values = remove_duplicates = None

try:
	from .alignment import normalize_dates, merge_news_stock_data, prepare_ml_features, validate_date_alignment
except Exception:
	normalize_dates = merge_news_stock_data = prepare_ml_features = validate_date_alignment = None

try:
	from ..config.constants import ROOT, DATA_DIR, RAW_DIR, PROCESSED_DIR, OUTPUT_DIR
except Exception:
	ROOT = DATA_DIR = RAW_DIR = PROCESSED_DIR = OUTPUT_DIR = None

try:
	from ..config.logger import get_logger
except Exception:
	def get_logger(name=None):
		import logging
		return logging.getLogger(name)

try:
	from .seed import seed_everything
except Exception:
	def seed_everything(seed: int = 42):
		try:
			import random, numpy as _np
			random.seed(seed)
			_np.random.seed(seed)
		except Exception:
			pass
