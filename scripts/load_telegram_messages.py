import os
import json
import psycopg2
from psycopg2.extras import execute_values
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Database connection settings
DB_HOST = os.getenv('PGHOST', 'localhost')
DB_PORT = os.getenv('PGPORT', '5432')
DB_NAME = os.getenv('PGDATABASE', 'medical_warehouse')
DB_USER = os.getenv('PGUSER', 'postgres')
DB_PASS = os.getenv('PGPASSWORD', 'password')

RAW_DATA_DIR = os.path.join('data', 'raw', 'telegram_messages')

CREATE_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS raw.telegram_messages (
    message_id BIGINT PRIMARY KEY,
    channel_name TEXT,
    channel_type TEXT,
    message_text TEXT,
    post_date TIMESTAMP,
    view_count INTEGER,
    forward_count INTEGER,
    image_url TEXT
);
'''

INSERT_SQL = '''
INSERT INTO raw.telegram_messages (
    message_id, channel_name, channel_type, message_text, post_date, view_count, forward_count, image_url
) VALUES %s
ON CONFLICT (message_id) DO NOTHING;
'''

def load_json_files():
    records = []
    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for msg in data:
                        records.append((
                            msg.get('message_id'),
                            msg.get('channel_name'),
                            msg.get('channel_type'),
                            msg.get('message_text'),
                            msg.get('post_date'),
                            msg.get('view_count'),
                            msg.get('forward_count'),
                            msg.get('image_url')
                        ))
    return records

def main():
    logging.info('Connecting to PostgreSQL...')
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    cur = conn.cursor()
    cur.execute('CREATE SCHEMA IF NOT EXISTS raw;')
    cur.execute(CREATE_TABLE_SQL)
    records = load_json_files()
    if records:
        execute_values(cur, INSERT_SQL, records)
        logging.info(f'Inserted {len(records)} records into raw.telegram_messages.')
    else:
        logging.warning('No records found to insert.')
    conn.commit()
    cur.close()
    conn.close()
    logging.info('Loader finished.')

if __name__ == '__main__':
    main()
