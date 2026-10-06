"""Test utilities for cleaning up between test runs to avoid stale data."""

import psycopg2

from rsmq import RedisSMQ

from config import POSTGRES_DSN, REDIS_HOST, REDIS_PORT

TABLES = [
    "labeled_data",
    "prediction_data",
    "suggestions",
    "paragraph_extraction_data",
    "paragraphs_from_languages",
]


def drain_queue(qname: str) -> None:
    """Remove all pending messages from a Redis queue."""
    queue = RedisSMQ(host=REDIS_HOST, port=REDIS_PORT, qname=qname, quiet=False)
    while True:
        message = queue.receiveMessage().exceptions(False).execute()
        if not message:
            break
        queue.deleteMessage(id=message["id"]).execute()


def truncate_all_data() -> None:
    """Remove all records from all persistence tables."""
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for table in TABLES:
                cur.execute(f"TRUNCATE TABLE {table}")
    finally:
        conn.close()


def delete_tenant_data(run_name: str) -> None:
    """Delete all PostgreSQL records for a given run_name across all relevant tables."""
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for table in TABLES:
                cur.execute(f"DELETE FROM {table} WHERE run_name = %s", (run_name,))
    finally:
        conn.close()
