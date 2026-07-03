"""Test utilities for cleaning up between test runs to avoid stale data."""

from pymongo import MongoClient
from rsmq import RedisSMQ

from config import MONGO_HOST, MONGO_PORT, REDIS_HOST, REDIS_PORT


def drain_queue(qname: str) -> None:
    """Remove all pending messages from a Redis queue."""
    queue = RedisSMQ(host=REDIS_HOST, port=REDIS_PORT, qname=qname, quiet=False)
    while True:
        message = queue.receiveMessage().exceptions(False).execute()
        if not message:
            break
        queue.deleteMessage(id=message["id"]).execute()


def delete_tenant_data(run_name: str) -> None:
    """Delete all MongoDB records for a given run_name across all relevant collections."""
    client = MongoClient(f"{MONGO_HOST}:{MONGO_PORT}")
    db = client["pdf_metadata_extraction"]
    for collection in [
        "labeled_data",
        "prediction_data",
        "suggestions",
        "paragraph_extraction_data",
        "paragraphs_from_languages",
    ]:
        db[collection].delete_many({"run_name": run_name})
    client.close()
