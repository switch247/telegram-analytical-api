"""
Data Loader Script
Task 1: Data Loading

This script loads processed data into a PostgreSQL database.
"""

import os
import psycopg2
import pandas as pd
from src.config.settings import DATA_PATHS

class DatabaseLoader:
    """Loader class for PostgreSQL database"""

    def __init__(self, db_params=None):
        """
        Initialize loader
        
        Args:
            db_params (dict): Database connection parameters
        """
        self.db_params = db_params or {
            "host": "localhost",
            "database": "restaurant_reviews", # Default from notebook, should probably be configurable
            "user": "admin",
            "password": "admin123"
        }
        self.conn = None
        self.cur = None

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_params)
            self.cur = self.conn.cursor()
            print("Connected to database successfully")
            return True
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            return False

    def create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.cur:
            return False
        
        try:
            # Create restaurants table
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS restaurants (
                    restaurant_id SERIAL PRIMARY KEY,
                    restaurant_name VARCHAR(255) UNIQUE
                );
            """)
            
            # Create reviews table
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id SERIAL PRIMARY KEY,
                    restaurant_id INT REFERENCES restaurants(restaurant_id),
                    review_text TEXT,
                    rating INT,
                    review_date DATE,
                    source VARCHAR(50)
                );
            """)
            
            self.conn.commit()
            print("Tables created successfully")
            return True
        except Exception as e:
            print(f"Error creating tables: {str(e)}")
            self.conn.rollback()
            return False

    def insert_data(self, restaurants_df, reviews_df):
        """Insert data into tables"""
        if not self.cur:
            return False

        try:
            # Insert restaurants
            print("Inserting restaurants...")
            for _, row in restaurants_df.iterrows():
                self.cur.execute("""
                    INSERT INTO restaurants (restaurant_id, restaurant_name)
                    VALUES (%s, %s)
                    ON CONFLICT (restaurant_name) DO NOTHING;
                """, (int(row["restaurant_id"]), row["restaurant_name"]))

            # Insert reviews
            print("Inserting reviews...")
            for _, row in reviews_df.iterrows():
                self.cur.execute("""
                    INSERT INTO reviews 
                        (review_id, restaurant_id, review_text, rating, review_date, source)
                    VALUES 
                        (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (review_id) DO NOTHING;
                """, (
                    int(row["review_id"]),
                    int(row["restaurant_id"]),
                    row["review_text"],
                    int(row["rating"]),
                    row["review_date"],
                    row["source"]
                ))
            
            self.conn.commit()
            print("Data inserted successfully")
            return True
        except Exception as e:
            print(f"Error inserting data: {str(e)}")
            self.conn.rollback()
            return False

    def close(self):
        """Close database connection"""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        print("Database connection closed")

if __name__ == "__main__":
    # Example usage
    loader = DatabaseLoader()
    if loader.connect():
        loader.create_tables()
        # Load dataframes here if needed for testing
        # df_restaurants = pd.read_csv(...)
        # df_reviews = pd.read_csv(...)
        # loader.insert_data(df_restaurants, df_reviews)
        loader.close()
