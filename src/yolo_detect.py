import os
import csv
from ultralytics import YOLO
import psycopg2
from psycopg2.extras import execute_values

IMAGE_DIR = 'data/raw/images'
OUTPUT_CSV = 'data/processed/yolo_detections.csv'
MODEL_PATH = 'yolov8n.pt'  # Use YOLOv8 nano for efficiency

# Classes of interest for medical business context
PRODUCT_CLASSES = {'bottle', 'container'}
PERSON_CLASS = 'person'

model = YOLO(MODEL_PATH)

def classify_image(detections):
    has_person = any(det['name'] == PERSON_CLASS for det in detections)
    has_product = any(det['name'] in PRODUCT_CLASSES for det in detections)
    if has_person and has_product:
        return 'promotional'
    elif has_product:
        return 'product_display'
    elif has_person:
        return 'lifestyle'
    else:
        return 'other'

def scan_images():
    results = []
    for channel in os.listdir(IMAGE_DIR):
        channel_path = os.path.join(IMAGE_DIR, channel)
        if not os.path.isdir(channel_path):
            continue
        for img_file in os.listdir(channel_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(channel_path, img_file)
            yolo_result = model(img_path)
            detections = [
                {'name': model.names[cls], 'confidence': float(conf)}
                for cls, conf in zip(yolo_result[0].boxes.cls.tolist(), yolo_result[0].boxes.conf.tolist())
            ]
            image_category = classify_image(detections)
            for det in detections:
                results.append({
                    'channel_name': channel,
                    'message_id': os.path.splitext(img_file)[0],
                    'detected_class': det['name'],
                    'confidence_score': det['confidence'],
                    'image_category': image_category
                })
    return results

def save_results(results):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['channel_name', 'message_id', 'detected_class', 'confidence_score', 'image_category'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

def load_to_postgres(results):
    DB_HOST = os.getenv('PGHOST', 'localhost')
    DB_PORT = os.getenv('PGPORT', '5432')
    DB_NAME = os.getenv('PGDATABASE', 'medical_warehouse')
    DB_USER = os.getenv('PGUSER', 'postgres')
    DB_PASS = os.getenv('PGPASSWORD', 'password')
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    cur = conn.cursor()
    cur.execute('CREATE SCHEMA IF NOT EXISTS processed;')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS processed.yolo_detections (
            channel_name TEXT,
            message_id TEXT,
            detected_class TEXT,
            confidence_score FLOAT,
            image_category TEXT
        );
    ''')
    values = [(
        r['channel_name'], r['message_id'], r['detected_class'], r['confidence_score'], r['image_category']
    ) for r in results]
    execute_values(cur, '''
        INSERT INTO processed.yolo_detections (channel_name, message_id, detected_class, confidence_score, image_category)
        VALUES %s
        ON CONFLICT DO NOTHING;
    ''', values)
    conn.commit()
    cur.close()
    conn.close()

def main():
    results = scan_images()
    save_results(results)
    load_to_postgres(results)
    print(f"Saved {len(results)} detections to {OUTPUT_CSV} and loaded to PostgreSQL.")

if __name__ == '__main__':
    main()
