"""
OpenSearch backend helpers.

This module handles:
1. OpenSearch client creation
2. Index creation
3. CSV snapshot ingestion into a single document index
"""

import json
import os
from typing import Any

import pandas as pd
from opensearchpy import OpenSearch

from hasher import detokenize_value

DEFAULT_OS_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
DEFAULT_INDEX = os.getenv("OPENSEARCH_INDEX", "freshmart_records")


def get_client() -> OpenSearch:
    """Create an OpenSearch client for local Docker development."""
    return OpenSearch(
        hosts=[DEFAULT_OS_URL],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )


def _load_registry() -> dict[str, Any]:
    with open("schema_registry.json", "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_index(client: OpenSearch, index_name: str = DEFAULT_INDEX) -> None:
    """Create an index with a flexible mapping for mixed snapshot documents."""
    if client.indices.exists(index=index_name):
        return

    body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "dynamic": True,
            "properties": {
                "branch": {"type": "keyword"},
                "resource": {"type": "keyword"},
                "snapshot": {"type": "date", "format": "strict_date_optional_time||yyyy-MM-dd"},
                "file_path": {"type": "keyword"},
                "product_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "category": {"type": "keyword"},
                "product_id": {"type": "keyword"},
                "sale_date": {"type": "date", "format": "strict_date_optional_time||yyyy-MM-dd"},
            },
        },
    }
    client.indices.create(index=index_name, body=body)


def _coerce_value(value: Any) -> Any:
    """Normalize dataframe values before indexing."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    return str(value)


def index_snapshot_data(index_name: str = DEFAULT_INDEX, reset: bool = False) -> int:
    """
    Load all CSV snapshots from schema_registry.json into one OpenSearch index.

    Returns the number of indexed documents.
    """
    client = get_client()
    if reset and client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    ensure_index(client, index_name=index_name)
    registry = _load_registry()

    total = 0
    for branch, bdata in registry["branches"].items():
        for resource, rdata in bdata["resources"].items():
            sensitive_cols = [
                col for col, meta in rdata["columns"].items() if meta.get("is_sensitive")
            ]

            for fname in rdata["files"]:
                fpath = os.path.join(rdata["path"], fname)
                snapshot = fname.rsplit("_", 1)[-1].replace(".csv", "")

                df = pd.read_csv(fpath)
                for col in sensitive_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(str).map(detokenize_value)
                        numeric = pd.to_numeric(df[col], errors="coerce")
                        df[col] = numeric.where(numeric.notna(), df[col])

                for row in df.to_dict(orient="records"):
                    doc = {
                        "branch": branch,
                        "resource": resource,
                        "snapshot": snapshot,
                        "file_path": fpath.replace("\\", "/"),
                    }
                    doc.update({k: _coerce_value(v) for k, v in row.items()})
                    client.index(index=index_name, body=doc)
                    total += 1

    client.indices.refresh(index=index_name)
    return total
