# Medical Telegram Data Warehouse (dbt)

This project is part of an end-to-end data pipeline for analyzing Ethiopian medical business activity on Telegram. It uses dbt to transform raw, messy Telegram data into a clean, trusted data warehouse, enabling advanced analytics and reporting.

## Project Overview
- **Source Data:** Public Telegram channels selling medical, pharmaceutical, and cosmetic products in Ethiopia (e.g., CheMed, Lobelia Cosmetics, Tikvah Pharma).
- **Pipeline:**
  1. **Extract & Load:** Scrape messages and images from Telegram, store as JSON and images in a data lake, then load into PostgreSQL (raw schema).
  2. **Transform (dbt):** Clean, standardize, and model the data into a star schema using dbt. This includes staging, dimension, and fact tables, plus data quality tests.
  3. **Enrich:** (Next steps) Use YOLO object detection to analyze images and join results to the warehouse.
  4. **Expose:** (Next steps) Serve analytics via a FastAPI-powered API.

## Star Schema Design
- **Fact Table:** `fct_messages` (one row per Telegram message, with metrics and foreign keys)
- **Dimension Tables:**
  - `dim_channels` (channel info, type, post stats)
  - `dim_dates` (date attributes for time-based analysis)

This design enables efficient analysis of trends, channel activity, and content types.

## Key Features
- Cleans and standardizes raw Telegram data
- Implements dimensional modeling (star schema)
- Data quality enforced with dbt tests (unique, not_null, relationships, custom business rules)
- Modular, scalable, and ready for enrichment and API exposure

## How to Use
1. Ensure raw data is loaded into PostgreSQL (see pipeline scripts)
2. Configure your dbt profile for PostgreSQL connection
3. Run dbt transformations:
   ```
   dbt run --project-dir medical_warehouse
   dbt test --project-dir medical_warehouse
   dbt docs generate --project-dir medical_warehouse
   dbt docs serve --project-dir medical_warehouse
   ```
4. Explore the generated documentation and marts for analytics

## Documentation
- See the [project report](../experiments/todo.md) for business context, requirements, and schema diagrams.
- dbt auto-generates detailed documentation for all models and tests.

---
For more, see the main project README and the challenge document in `experiments/todo.md`.
