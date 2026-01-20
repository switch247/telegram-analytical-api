import pytest
import pandas as pd
import numpy as np
from src.features.xente_features import TemporalExtractor, CustomerAggregates

def test_temporal_extractor():
    """Test that TemporalExtractor correctly extracts time components."""
    df = pd.DataFrame({'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-02 11:00:00']})
    extractor = TemporalExtractor(time_col='TransactionStartTime')
    df_transformed = extractor.transform(df)
    
    assert 'transaction_hour' in df_transformed.columns
    assert 'transaction_day' in df_transformed.columns
    assert 'transaction_month' in df_transformed.columns
    assert 'transaction_year' in df_transformed.columns
    assert df_transformed['transaction_hour'].iloc[0] == 10
    assert df_transformed['transaction_day'].iloc[1] == 2

def test_customer_aggregates():
    """Test that CustomerAggregates correctly computes aggregated features."""
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, 200, 50]
    })
    aggregator = CustomerAggregates(customer_id_col='CustomerId', amount_col='Amount')
    aggregator.fit(df)
    df_transformed = aggregator.transform(df)
    
    assert 'total_amount' in df_transformed.columns
    assert 'avg_amount' in df_transformed.columns
    assert 'txn_count' in df_transformed.columns
    
    # Check values for C1
    c1_data = df_transformed[df_transformed['CustomerId'] == 'C1']
    assert c1_data['total_amount'].iloc[0] == 300
    assert c1_data['txn_count'].iloc[0] == 2
    assert c1_data['avg_amount'].iloc[0] == 150
    
    # Check values for C2
    c2_data = df_transformed[df_transformed['CustomerId'] == 'C2']
    assert c2_data['total_amount'].iloc[0] == 50
    assert c2_data['txn_count'].iloc[0] == 1
