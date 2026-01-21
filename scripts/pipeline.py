from dagster import job, op, ScheduleDefinition, RunRequest, sensor, RunConfig
import subprocess
import logging

@op
def scrape_telegram_data():
    logging.info("Scraping Telegram data...")
    subprocess.run(["python", "src/scraper.py"], check=True)

@op
def load_raw_to_postgres():
    logging.info("Loading raw data to PostgreSQL...")
    subprocess.run(["python", "scripts/load_telegram_messages.py"], check=True)

@op
def run_dbt_transformations():
    logging.info("Running dbt transformations and tests...")
    subprocess.run([".venv/Scripts/dbt", "run", "--project-dir", "medical_warehouse"], check=True)
    subprocess.run([".venv/Scripts/dbt", "test", "--project-dir", "medical_warehouse"], check=True)

@op
def run_yolo_enrichment():
    logging.info("Running YOLO enrichment...")
    subprocess.run(["python", "src/yolo_detect.py"], check=True)

@job(tags={"env": "prod", "team": "data-eng"})
def telegram_medical_pipeline():
    scrape_telegram_data()
    load_raw_to_postgres()
    run_dbt_transformations()
    run_yolo_enrichment()

# Daily schedule at 2am UTC
telegram_medical_schedule = ScheduleDefinition(
    job=telegram_medical_pipeline,
    cron_schedule="0 2 * * *",
    run_config={},
    tags={"schedule": "daily"}
)

# Example sensor for manual triggering or monitoring
@sensor(job=telegram_medical_pipeline)
def manual_trigger_sensor(context):
    # Could add logic to check for new files, etc.
    if False:  # Replace with real condition
        yield RunRequest(run_key=None, run_config=RunConfig(tags={"triggered_by": "sensor"}))
