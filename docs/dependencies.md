# Dependencies & Environment

This project uses standard Python data tooling and selected ML libraries. The exact versions are managed via `requirements.txt` and `pyproject.toml`.

## Installation (High Level)


## Typical Libraries (Indicative)

## Notes
## Project Dependencies

### Python Packages
- telethon (Telegram scraping)
- dbt-postgres (data transformation)
- ultralytics (YOLOv8 object detection)
- fastapi, uvicorn (API)
- dagster, dagster-webserver (orchestration)
- SQLAlchemy (database connection)
- pydantic (data validation)

### System/Service Dependencies
- Docker & Docker Compose
- PostgreSQL

### Other Tools
- dbt (Data Build Tool)
- YOLOv8 (pre-trained model: yolov8n.pt)

See `requirements.txt` for the full list of Python dependencies.
- Prefer pinned versions for reproducibility.
- Use CI to validate installation and tests on push.
