"""
Postgres DB helper using connection pooling.

Provides a `PostgresDB` class that manages a connection pool and common
operations: creating tables, inserting banks and reviews, and running
verification queries.

Usage:
    from src.utils.db_helper import PostgresDB
    db = PostgresDB()
    db.init_pool()
    db.create_tables()
    db.insert_reviews_from_df(df)
    db.close_pool()

This module follows PEP8 class and naming conventions.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse
from contextlib import contextmanager
from typing import Dict, Iterable, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import SimpleConnectionPool


class PostgresDB:
    """Manage PostgreSQL connection pool and provide helper methods.

    Reads connection parameters from environment variables with sensible
    defaults. Use `init_pool` to create the connection pool.
    """

    def __init__(self, db_params: Optional[Dict[str, str]] = None, minconn: int = 1, maxconn: int = 5):
        # Allow DATABASE_URL to be used as a single connection string.
        database_url = os.getenv("DATABASE_URL")
        if database_url and not db_params:
            # Expect format: postgresql://user:pass@host:port/dbname
            parsed = urlparse(database_url)
            self.db_params = {
                "host": parsed.hostname or "localhost",
                "port": int(parsed.port) if parsed.port else 5432,
                "database": parsed.path.lstrip("/") or "bank_reviews",
                "user": parsed.username or "postgres",
                "password": parsed.password or "",
            }
        else:
            self.db_params = db_params or {
                "host": os.getenv("PGHOST", "localhost"),
                "port": int(os.getenv("PGPORT", 5432)),
                "database": os.getenv("PGDATABASE", "bank_reviews"),
                "user": os.getenv("PGUSER", "postgres"),
                "password": os.getenv("PGPASSWORD", "")
            }
        self.minconn = minconn
        self.maxconn = maxconn
        self.pool: Optional[SimpleConnectionPool] = None

    def init_pool(self) -> None:
        """Initialize the psycopg2 connection pool."""
        if self.pool:
            return
        d = self.db_params.copy()
        # psycopg2 expects strings for everything but port
        d["port"] = str(d["port"]) if "port" in d else None
        self.pool = SimpleConnectionPool(self.minconn, self.maxconn, **self.db_params)

    def close_pool(self) -> None:
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
            self.pool = None

    @contextmanager
    def get_conn(self):
        """Context manager to get a connection from the pool."""
        if not self.pool:
            self.init_pool()
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    def create_tables(self) -> None:
        """Create `banks` and `reviews` tables if they don't exist."""
        create_banks = (
            """
            CREATE TABLE IF NOT EXISTS banks (
                bank_id SERIAL PRIMARY KEY,
                bank_name TEXT UNIQUE NOT NULL,
                app_name TEXT
            );
            """
        )

        create_reviews = (
            """
            CREATE TABLE IF NOT EXISTS reviews (
                review_id SERIAL PRIMARY KEY,
                orig_review_id TEXT UNIQUE,
                bank_id INTEGER REFERENCES banks(bank_id) ON DELETE SET NULL,
                review_text TEXT,
                rating INTEGER,
                review_date DATE,
                sentiment_label TEXT,
                sentiment_score REAL,
                source TEXT
            );
            """
        )

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(create_banks)
                cur.execute(create_reviews)
            conn.commit()

    def _upsert_banks(self, banks: Iterable[str]) -> Dict[str, int]:
        """Insert banks and return mapping bank_name -> bank_id.

        Uses ON CONFLICT DO NOTHING then selects ids.
        """
        bank_list = [(b,) for b in set(banks) if b]
        mapping: Dict[str, int] = {}
        if not bank_list:
            return mapping

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    "INSERT INTO banks (bank_name) VALUES %s ON CONFLICT (bank_name) DO NOTHING",
                    bank_list,
                )
                # Fetch ids for all banks provided
                cur.execute(
                    "SELECT bank_id, bank_name FROM banks WHERE bank_name = ANY(%s)",
                    (list(set(banks)),),
                )
                rows = cur.fetchall()
                mapping = {row[1]: int(row[0]) for row in rows}
            conn.commit()

        return mapping

    def insert_reviews_from_df(self, df: pd.DataFrame, batch_size: int = 1000) -> int:
        """Insert reviews from a DataFrame into the database.

        The DataFrame is expected to include columns:
        - 'bank_name', 'review_text', 'rating', 'review_date', 'sentiment_label', 'sentiment_score', 'source'

        Returns the number of rows inserted (attempted).
        """
        expected_cols = {
            "bank_name",
            "review_text",
            "rating",
            "review_date",
            "sentiment_label",
            "sentiment_score",
            "source",
            # optional original id coming from CSV (UUID/string)
            "review_id",
            # support notebooks and CSVs that already provide `orig_review_id`
            "orig_review_id",
        }
        if df is None or df.empty:
            return 0

        # Ensure expected columns exist; fill missing with None
        for c in expected_cols:
            if c not in df.columns:
                df[c] = None

        banks = df["bank_name"].astype(str).tolist()
        bank_map = self._upsert_banks(banks)

        records = []
        for _, row in df.iterrows():
            bank_name = row.get("bank_name")
            bank_id = bank_map.get(bank_name)
            review_date = None
            if pd.notna(row.get("review_date")):
                try:
                    review_date = pd.to_datetime(row.get("review_date")).date()
                except Exception:
                    review_date = None

            # Prefer any existing `orig_review_id` column (notebook/csv may provide this),
            # otherwise fall back to `review_id` if present.
            orig_id = None
            if "orig_review_id" in df.columns and pd.notna(row.get("orig_review_id")):
                orig_id = str(row.get("orig_review_id"))
            elif "review_id" in df.columns and pd.notna(row.get("review_id")):
                orig_id = str(row.get("review_id"))

            records.append(
                (
                    orig_id,
                    bank_id,
                    row.get("review_text"),
                    int(row.get("rating")) if pd.notna(row.get("rating")) else None,
                    review_date,
                    row.get("sentiment_label"),
                    float(row.get("sentiment_score")) if pd.notna(row.get("sentiment_score")) else None,
                    row.get("source"),
                )
            )

        insert_sql = (
            "INSERT INTO reviews (orig_review_id, bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, source) "
            "VALUES %s ON CONFLICT (orig_review_id) DO UPDATE SET "
            "bank_id = EXCLUDED.bank_id, review_text = EXCLUDED.review_text, rating = EXCLUDED.rating, review_date = EXCLUDED.review_date, "
            "sentiment_label = EXCLUDED.sentiment_label, sentiment_score = EXCLUDED.sentiment_score, source = EXCLUDED.source"
        )

        inserted = 0
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(records), batch_size):
                    batch = records[i : i + batch_size]
                    execute_values(cur, insert_sql, batch)
                    inserted += len(batch)
            conn.commit()

        return inserted

    def query_review_count_by_bank(self) -> Dict[str, int]:
        """Return a mapping of bank_name -> review count."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT b.bank_name, COUNT(r.review_id) "
                    "FROM banks b LEFT JOIN reviews r ON b.bank_id = r.bank_id "
                    "GROUP BY b.bank_name ORDER BY COUNT(r.review_id) DESC"
                )
                rows = cur.fetchall()
        return {row[0]: int(row[1]) for row in rows}

    def avg_rating_by_bank(self) -> Dict[str, float]:
        """Return average rating per bank (None if no ratings)."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT b.bank_name, AVG(r.rating) "
                    "FROM banks b LEFT JOIN reviews r ON b.bank_id = r.bank_id "
                    "GROUP BY b.bank_name ORDER BY AVG(r.rating) DESC NULLS LAST"
                )
                rows = cur.fetchall()
        return {row[0]: (float(row[1]) if row[1] is not None else None) for row in rows}

    def avg_sentiment_by_bank(self) -> Dict[str, Optional[float]]:
        """Return average sentiment_score per bank."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT b.bank_name, AVG(r.sentiment_score) "
                    "FROM banks b LEFT JOIN reviews r ON b.bank_id = r.bank_id "
                    "GROUP BY b.bank_name ORDER BY AVG(r.sentiment_score) DESC NULLS LAST"
                )
                rows = cur.fetchall()
        return {row[0]: (float(row[1]) if row[1] is not None else None) for row in rows}

    def sentiment_counts_by_bank(self) -> Dict[str, Dict[str, int]]:
        """Return counts of sentiment_label values per bank.

        Example return: { 'CBE': {'positive': 10, 'negative': 2, 'neutral': 3}, ... }
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT b.bank_name, r.sentiment_label, COUNT(r.review_id) "
                    "FROM banks b LEFT JOIN reviews r ON b.bank_id = r.bank_id "
                    "GROUP BY b.bank_name, r.sentiment_label"
                )
                rows = cur.fetchall()

        out: Dict[str, Dict[str, int]] = {}
        for bank_name, label, cnt in rows:
            out.setdefault(bank_name, {})
            out[bank_name][label if label is not None else 'unknown'] = int(cnt)
        return out

    def verify_sentiment_storage(self) -> Dict[str, int]:
        """Return overall sentiment_label counts and null count for quick verification.

        Returns a dict: {'positive': n, 'neutral': n, 'negative': n, 'unknown': n, 'nulls': n}
        """
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT sentiment_label, COUNT(*) FROM reviews GROUP BY sentiment_label")
                rows = cur.fetchall()
                cur.execute("SELECT COUNT(*) FROM reviews WHERE sentiment_label IS NULL")
                nulls = cur.fetchone()[0]

        out: Dict[str, int] = {row[0] if row[0] is not None else 'unknown': int(row[1]) for row in rows}
        out['nulls'] = int(nulls)
        return out


__all__ = ["PostgresDB"]
