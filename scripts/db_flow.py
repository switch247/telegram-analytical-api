"""Database management flow script.

Combines functionality for creating the database, testing connections, 
running migrations, and dumping the database.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import psycopg2

# Try to import project specific helper if available
try:
    # Add project root to path to allow imports from src
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.utils.db_helper import PostgresDB
except ImportError:
    PostgresDB = None


def get_conn_params_from_env():
    """Extract connection parameters from DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError("DATABASE_URL not set")
    parsed = urlparse(database_url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/") or "",
        "user": parsed.username or "",
        "password": parsed.password or "",
    }


def create_database():
    """Create the target Postgres database if it doesn't exist."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError("DATABASE_URL not set")

    parsed = urlparse(database_url)
    target_db = parsed.path.lstrip("/")
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    user = parsed.username or "postgres"
    password = parsed.password or ""

    print(f"Checking if database '{target_db}' exists...")
    # Connect to the maintenance DB 'postgres' to run CREATE DATABASE
    conn = None
    try:
        conn = psycopg2.connect(host=host, port=port, dbname="postgres", user=user, password=password)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
        exists = cur.fetchone() is not None
        if exists:
            print(f"Database '{target_db}' already exists.")
        else:
            print(f"Creating database '{target_db}'...")
            cur.execute(f"CREATE DATABASE \"{target_db}\"")
            print("Database created.")
        cur.close()
    except Exception as e:
        print(f"Failed to create database: {e}")
    finally:
        if conn:
            conn.close()


def test_connection():
    """Test database connection."""
    print("Testing DB connection using DATABASE_URL...")
    
    # Test with psycopg2
    try:
        params = get_conn_params_from_env()
        conn = psycopg2.connect(
            host=params["host"],
            port=params["port"],
            dbname=params["database"],
            user=params["user"],
            password=params["password"],
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        val = cur.fetchone()
        cur.close()
        conn.close()
        print(f"psycopg2: SELECT 1 -> {val}")
    except Exception as e:
        print(f"psycopg2 test failed: {e}")
        return False

    # Test with PostgresDB helper if available
    if PostgresDB:
        try:
            db = PostgresDB()
            db.init_pool()
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    val = cur.fetchone()
            db.close_pool()
            print(f"PostgresDB helper: SELECT 1 -> {val}")
        except Exception as e:
            print(f"PostgresDB helper test failed: {e}")
    
    return True


def run_migrations(sql_path: str = "sql/schema.sql"):
    """Run SQL migrations."""
    p = Path(sql_path)
    if not p.exists():
        # Try relative to project root if not found
        project_root = Path(__file__).resolve().parents[1]
        p = project_root / sql_path
        if not p.exists():
            raise FileNotFoundError(f"Schema file not found: {sql_path}")

    print(f"Running migrations from {p}...")
    params = get_conn_params_from_env()

    conn = None
    try:
        conn = psycopg2.connect(
            host=params["host"],
            port=params["port"],
            dbname=params["database"],
            user=params["user"],
            password=params["password"],
        )
        cur = conn.cursor()
        sql = p.read_text(encoding='utf-8')
        cur.execute(sql)
        conn.commit()
        cur.close()
        print("Migrations applied successfully.")
    except Exception as e:
        print(f"Migration failed: {e}")
        raise
    finally:
        if conn:
            conn.close()


def run_pg_dump(out_path: str) -> bool:
    """Try to run pg_dump."""
    pg_dump = shutil.which("pg_dump")
    if not pg_dump:
        return False
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError("DATABASE_URL not set")
    
    print(f"Running pg_dump to {out_path}...")
    # Use the URL as the DSN for pg_dump
    cmd = [pg_dump, "--dbname", database_url, "-f", out_path]
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"pg_dump failed: {e}")
        return False


def fallback_export(out_path: str):
    """Fallback export using SELECT statements."""
    print("Running fallback export...")
    params = get_conn_params_from_env()
    conn = psycopg2.connect(
        host=params["host"],
        port=params["port"],
        dbname=params["database"],
        user=params["user"],
        password=params["password"],
    )
    cur = conn.cursor()
    
    with open(out_path, 'w', encoding='utf8') as f:
        f.write('-- Dump generated by scripts/db_flow.py (fallback)\n')
        
        # Get all tables
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = cur.fetchall()
        
        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            f.write(f"\n-- Data for table {table_name}\n")
            
            cur.execute(f"SELECT * FROM {table_name}")
            rows = cur.fetchall()
            if rows:
                for row in rows:
                    vals = []
                    for x in row:
                        if x is None:
                            vals.append('NULL')
                        elif isinstance(x, (int, float)):
                            vals.append(str(x))
                        else:
                            # Escape single quotes
                            vals.append(f"'{str(x).replace("'", "''")}'")
                    
                    vals_str = ', '.join(vals)
                    f.write(f"INSERT INTO {table_name} VALUES ({vals_str});\n")
    
    cur.close()
    conn.close()


def dump_db(out_path: str = "customer_fintec_dump.sql"):
    """Dump database to file."""
    try:
        if run_pg_dump(out_path):
            print(f"pg_dump succeeded -> {out_path}")
            return
    except Exception as e:
        print(f"pg_dump attempt failed: {e}")

    print("Falling back to simple export...")
    try:
        fallback_export(out_path)
        print(f"Fallback export written to {out_path}")
    except Exception as e:
        print(f"Fallback export failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Database management flow")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create command
    subparsers.add_parser("create", help="Create database if not exists")

    # Test command
    subparsers.add_parser("test", help="Test database connection")

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run SQL migrations")
    migrate_parser.add_argument("--sql", default="sql/schema.sql", help="Path to SQL schema file")

    # Dump command
    dump_parser = subparsers.add_parser("dump", help="Dump database")
    dump_parser.add_argument("--out", default="customer_fintec_dump.sql", help="Output file path")

    args = parser.parse_args()

    if args.command == "create":
        create_database()
    elif args.command == "test":
        test_connection()
    elif args.command == "migrate":
        run_migrations(args.sql)
    elif args.command == "dump":
        dump_db(args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
