# Interim Report: Telegram Medical Business Analytical Pipeline
 
## Data Lake Structure
- Raw Telegram messages stored as partitioned JSON files: `data/raw/telegram_messages/YYYY-MM-DD/channel_name.json`
- Images downloaded to: `data/raw/images/{channel_name}/{message_id}.jpg`
- Logs captured in `logs/` for scraping activity and errors
 
## Star Schema Diagram
- Fact Table: `fct_messages` (message_id, channel_key, date_key, metrics)
- Dimension Tables: `dim_channels` (channel info), `dim_dates` (date attributes)
 
## Data Quality Issues & Solutions
- **Missing/empty messages:** Filtered in staging model
- **Inconsistent data types:** Cast in dbt staging
- **Duplicate message IDs:** Enforced with unique tests
- **Future dates:** Custom dbt test
- **Negative view counts:** Custom dbt test
 
## Summary
The pipeline reliably ingests, cleans, and models Telegram data for medical business analytics. Data quality is enforced at every stage, and the star schema supports efficient analysis.
 
---
See `experiments/todo.md` for full challenge context and requirements.


