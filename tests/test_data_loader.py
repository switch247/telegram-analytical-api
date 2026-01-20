"""
Test basic imports and structure
"""
import pytest


def test_imports():
    """Test that basic modules can be imported"""
    from src import config
    from src import utils
    from src import analysis
    from src import visualisation
    from src import pipeline
    
    assert config is not None
    assert utils is not None
    assert analysis is not None
    assert visualisation is not None
    assert pipeline is not None


def test_utils_constants():
    """Test utils constants"""
    from src.utils import ROOT, DATA_DIR
    assert ROOT is not None
    assert DATA_DIR is not None


def test_database_loader_init():
    """Test DatabaseLoader initialization"""
    from src.utils import DatabaseLoader
    
    loader = DatabaseLoader()
    assert loader.db_params is not None
    assert loader.conn is None
    assert loader.cur is None


def test_database_loader_custom_params():
    """Test DatabaseLoader with custom parameters"""
    from src.utils import DatabaseLoader
    
    custom_params = {
        "host": "testhost",
        "database": "testdb",
        "user": "testuser",
        "password": "testpass"
    }
    loader = DatabaseLoader(db_params=custom_params)
    assert loader.db_params == custom_params
