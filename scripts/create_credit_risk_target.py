import sys
import os
import pandas as pd
import logging

# Add the project root to the python path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.rfm_proxy import RFMProxyLabeler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    input_path = os.path.join('data', 'processed', 'xente_processed.csv')
    output_path = os.path.join('data', 'processed', 'xente_processed_with_risk.csv')

    logger.info(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
        return

    logger.info(f"Data loaded. Shape: {df.shape}")

    # Initialize the labeler
    # Using 'CustomerId', 'TransactionStartTime', 'Amount'
    labeler = RFMProxyLabeler(
        customer_id_col='CustomerId',
        transaction_time_col='TransactionStartTime',
        amount_col='Amount',
        n_clusters=3,
        random_state=42
    )

    logger.info("Calculating RFM and assigning risk labels...")
    df_labeled = labeler.fit_transform(df)

    # Log some stats about the clusters
    rfm_stats = labeler.rfm_df_.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    logger.info("Cluster Centroids (Mean RFM):")
    logger.info("\n" + str(rfm_stats))
    
    high_risk_cluster = labeler.high_risk_cluster_
    logger.info(f"Identified High-Risk Cluster: {high_risk_cluster}")
    
    risk_counts = df_labeled['is_high_risk'].value_counts()
    logger.info("Risk Label Distribution:")
    logger.info("\n" + str(risk_counts))

    logger.info(f"Saving labeled data to {output_path}...")
    df_labeled.to_csv(output_path, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    main()
