from dagster import job, op
import subprocess

@op
def scrape_telegram_data():
    subprocess.run(["python", "src/scraper.py"], check=True)

@op
def load_raw_to_postgres():
    subprocess.run(["python", "scripts/load_telegram_messages.py"], check=True)

@op
def run_dbt_transformations():
    subprocess.run([".venv/Scripts/dbt", "run", "--project-dir", "medical_warehouse"], check=True)
    subprocess.run([".venv/Scripts/dbt", "test", "--project-dir", "medical_warehouse"], check=True)

@op
def run_yolo_enrichment():
    subprocess.run(["python", "src/yolo_detect.py"], check=True)

@job
def telegram_medical_pipeline():
    scrape_telegram_data()
    load_raw_to_postgres()
    run_dbt_transformations()
    run_yolo_enrichment()
