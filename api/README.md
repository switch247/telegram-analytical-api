# Telegram Medical Data Analytical API

This FastAPI application exposes analytical endpoints for the Telegram Medical Data Warehouse. It enables business users and analysts to query insights about Ethiopian medical business channels scraped from Telegram.

## Features
- **Top Products Endpoint:** Returns the most frequently mentioned products/terms across all channels.
- **Channel Activity Endpoint:** Returns posting activity and trends for a specific channel.
- **Message Search Endpoint:** Searches for messages containing a specific keyword.
- **Visual Content Stats Endpoint:** Returns statistics about image usage and content types across channels (integrated with YOLOv8 detection results).

## Endpoints
- `GET /api/reports/top-products?limit=10` — Top mentioned products/terms
- `GET /api/channels/{channel_name}/activity` — Channel posting stats
- `GET /api/search/messages?query=paracetamol&limit=20` — Search messages by keyword
- `GET /api/reports/visual-content` — Visual content statistics by channel

## Usage
1. Ensure the PostgreSQL database is populated and dbt models are built.
2. Install dependencies:
   ```
   pip install fastapi uvicorn sqlalchemy psycopg2-binary
   ```
3. Run the API:
   ```
   uvicorn api.main:app --reload
   ```
4. Access interactive docs at `/docs` (OpenAPI UI).

## Data Sources
- Data is sourced from the dbt models and YOLOv8 image detection pipeline.
- See the main project README and dbt documentation for schema details.

## Testing
- Basic endpoint test included in `api/test_api.py`.
- Use pytest to run tests:
   ```
   pytest api/test_api.py
   ```

## Project Context
This API is part of a larger pipeline for analyzing Ethiopian medical business activity on Telegram, including scraping, data warehousing (dbt), image enrichment (YOLOv8), and orchestration (Dagster).

For more details, see the main project README and challenge documentation.
