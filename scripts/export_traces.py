"""
Export Langfuse traces to CSV with flattened chunk and episode fields.

Usage:
    python scripts/export_traces.py [--days 7] [--output traces.csv]
"""

import os
import csv
import argparse
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()


def get_langfuse_client():
    """Initialize Langfuse client."""
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            timeout=60,  # us-region read API is slow; default timeout was too short
        )
    except Exception as e:
        print(f"Error initializing Langfuse: {e}")
        return None


def fetch_traces(client, days: int = 7):
    """Fetch traces from Langfuse API."""
    # Use the Langfuse API to fetch traces
    # The client provides access to the underlying API
    try:
        from langfuse.api import FetchTracesResponse
    except ImportError:
        pass

    traces = []
    page = 1

    while True:
        response = client.api.trace.list(
            name="rag_query",
            page=page,
            limit=100
        )

        if not response.data:
            break

        traces.extend(response.data)

        if not response.meta or page >= (response.meta.total_pages or 1):
            break
        page += 1

    return traces


def flatten_trace(trace) -> dict:
    """Flatten a trace into a CSV-friendly dict."""
    row = {
        "trace_id": trace.id,
        "timestamp": trace.timestamp.isoformat() if trace.timestamp else "",
        "session_id": trace.session_id or "",
        "input": trace.input or "",
        "output": trace.output or "",
        "latency_ms": "",
    }

    metadata = trace.metadata or {}

    # Extract latency
    row["latency_ms"] = metadata.get("latency_ms", "")

    # Extract chunk fields (up to 10 chunks)
    for i in range(1, 11):
        row[f"chunk_{i}_text"] = metadata.get(f"chunk_{i}_text", "")
        row[f"chunk_{i}_source"] = metadata.get(f"chunk_{i}_source", "")
        row[f"chunk_{i}_score"] = metadata.get(f"chunk_{i}_score", "")

    # Extract suggested episode fields (up to 5 episodes)
    for i in range(1, 6):
        row[f"suggested_{i}_title"] = metadata.get(f"suggested_{i}_title", "")
        row[f"suggested_{i}_guest"] = metadata.get(f"suggested_{i}_guest", "")
        row[f"suggested_{i}_url"] = metadata.get(f"suggested_{i}_url", "")

    return row


def export_to_csv(traces, output_file: str):
    """Export traces to CSV file."""
    if not traces:
        print("No traces to export")
        return

    # Flatten all traces
    rows = [flatten_trace(t) for t in traces]

    # Get fieldnames from first row
    fieldnames = list(rows[0].keys())

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} traces to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Export Langfuse traces to CSV")
    parser.add_argument("--days", type=int, default=7, help="Number of days to fetch (default: 7)")
    parser.add_argument("--output", type=str, default="traces_export.csv", help="Output CSV file")
    args = parser.parse_args()

    client = get_langfuse_client()
    if not client:
        print("Failed to initialize Langfuse client. Check your API keys.")
        return

    print(f"Fetching traces from the last {args.days} days...")
    traces = fetch_traces(client, args.days)
    print(f"Found {len(traces)} traces")

    export_to_csv(traces, args.output)


if __name__ == "__main__":
    main()
