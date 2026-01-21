# Telegram Medical Business Analytical Pipeline

This project builds a robust, end-to-end data platform for analyzing Ethiopian medical business activity on Telegram. It covers scraping, data warehousing, enrichment, API exposure, and orchestration.

## Project Overview
- **Source Data:** Public Telegram channels selling medical, pharmaceutical, and cosmetic products in Ethiopia.
- **Pipeline:**
  1. **Extract & Load:** Scrape messages/images, store as JSON/images, load into PostgreSQL (raw schema).
  2. **Transform (dbt):** Clean, standardize, and model data into a star schema using dbt. Includes staging, dimension, and fact tables, plus data quality tests.
  3. **Enrich (YOLOv8):** Run object detection on images, classify content, and join results to warehouse.
  4. **Expose (FastAPI):** Serve analytics via a REST API with endpoints for top products, channel activity, message search, and visual content stats.
  5. **Orchestrate (Dagster):** Automate and schedule the pipeline for production use.

## Key Features
- Modular, reproducible, and production-ready pipeline
- Dimensional modeling (star schema) for analytics
- Data quality enforced with dbt tests
- Image enrichment with YOLOv8
- REST API for business insights
- Orchestration and scheduling with Dagster

## How to Run
1. Set up Python environment and install dependencies from `requirements.txt`
2. Configure PostgreSQL and dbt profiles
3. Run pipeline steps individually or orchestrate with Dagster:
   ```
   dagster dev -f scripts/pipeline.py
   ```
4. Run FastAPI server for analytics:
   ```
   uvicorn api.main:app --reload
   ```
5. Access dbt docs and API docs for schema and endpoint details

## Documentation
- See `medical_warehouse/README.md` for dbt models and schema
- See `api/README.md` for API usage
- See `scripts/README.md` for pipeline orchestration
- See `reports/` for interim and final project reports

## Technologies
- Python, dbt, FastAPI, SQLAlchemy, YOLOv8, Dagster, PostgreSQL

---
For business context, requirements, and schema diagrams, see `experiments/todo.md` and `reports/`.

