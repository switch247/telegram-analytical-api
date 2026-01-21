# Final Report: Telegram Medical Business Analytical Pipeline

## Overview
This project delivers a production-ready data platform for analyzing Ethiopian medical business activity on Telegram. It covers scraping, data warehousing, enrichment, API exposure, and orchestration.

## Pipeline Architecture
- **Extract & Load:** Scrape Telegram messages/images, store as JSON/images, load into PostgreSQL.
- **Transform (dbt):** Clean, standardize, and model data into a star schema. Data quality enforced with dbt tests.
- **Enrich (YOLOv8):** Object detection on images, classification, and integration with warehouse.
- **Expose (FastAPI):** REST API for analytics (top products, channel activity, message search, visual content stats).
- **Orchestrate (Dagster):** Automated, scheduled pipeline for reliability and observability.

## Star Schema Design
- **Fact Table:** `fct_messages` (one row per message, metrics, FK to dimensions)
- **Dimension Tables:**
  - `dim_channels` (channel info, type, post stats)
  - `dim_dates` (date attributes)

## Technical Choices & Justifications
- **dbt:** Modular, testable transformations and dimensional modeling
- **YOLOv8:** Efficient image enrichment for visual analytics
- **FastAPI:** Modern, documented REST API for business insights
- **Dagster:** Robust orchestration and scheduling

## Screenshots
- dbt documentation (models, tests)
- FastAPI endpoints and OpenAPI docs
- Dagster UI showing successful pipeline runs

## Challenges & Learnings
- Handling messy, unstructured Telegram data
- Ensuring data quality and trust with dbt tests
- Integrating image analytics for richer insights
- Building a modular, scalable pipeline

## Improvements
- Expand enrichment to more image classes
- Add more analytical endpoints
- Enhance pipeline monitoring and alerting

---
For full details, see the main README, dbt/ and api/ documentation, and challenge context in `experiments/todo.md`.
