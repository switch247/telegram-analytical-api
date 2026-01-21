# Pipeline Orchestration (Dagster)

This project uses Dagster to orchestrate the end-to-end data pipeline for Telegram medical business analytics. The pipeline automates:
- Scraping Telegram data
- Loading raw data to PostgreSQL
- Running dbt transformations and tests
- Enriching data with YOLOv8 image detection

## How to Run
1. Install Dagster:
   ```
   pip install dagster dagster-webserver
   ```
2. Launch Dagster UI:
   ```
   dagster dev -f scripts/pipeline.py
   ```
3. Trigger the pipeline from the Dagster UI or CLI.

## Scheduling & Monitoring
- The pipeline can be scheduled to run daily and monitored for failures.
- See Dagster documentation for advanced scheduling and alerting.

## Integration
- Each pipeline step is modular and can be extended or monitored independently.
- The pipeline ensures data reliability, reproducibility, and observability for production use.

For more details, see the main project README and reports.
