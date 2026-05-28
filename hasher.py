"""
hasher.py

Token-based masking for sensitive CSV columns.

Instead of encryption, each sensitive value is replaced with a
human-readable token when written to disk, e.g.:

    price  4.50  →  PRICE_HASH_a3f9b2c1

All mappings are stored in hash_lookup.json as:

    { "PRICE_HASH_a3f9b2c1": "4.50", ... }

At query time sql_executor swaps tokens back to real values before
handing the DataFrame to DuckDB, so SQL arithmetic still works.
After the query, detokenize_results() cleans up any tokens that
make it into the final result set before displaying to the user.
"""

import json
import os
import uuid

LOOKUP_FILE = "hash_lookup.json"


def _load_lookup() -> dict:
    if os.path.exists(LOOKUP_FILE):
        with open(LOOKUP_FILE) as f:
            return json.load(f)
    return {}


def _save_lookup(lookup: dict) -> None:
    with open(LOOKUP_FILE, "w") as f:
        json.dump(lookup, f, indent=2)


# ── single-value helpers ──────────────────────────────────────────────────────

def tokenize_value(col_name: str, value: str) -> str:
    """
    Replace a sensitive value with a token like  PRICE_HASH_a3f9b2c1.
    The same (col_name, value) pair always gets the same token so
    re-running generate_data.py produces consistent tokens.
    The mapping is persisted in hash_lookup.json.
    """
    lookup = _load_lookup()
    reverse_key = f"__rev__{col_name}__{value}"
    if reverse_key in lookup:
        return lookup[reverse_key]            # already tokenized
    token = f"{col_name.upper()}_HASH_{uuid.uuid4().hex[:8]}"
    lookup[token]       = value               # token → original
    lookup[reverse_key] = token               # reverse index (dedup)
    _save_lookup(lookup)
    return token


def detokenize_value(token: str) -> str:
    """
    Look up a token and return the original value.
    Returns the token itself if not found (safe fallback).
    """
    return _load_lookup().get(token, token)


# ── column-level helpers (for pandas DataFrames) ─────────────────────────────

def tokenize_column(col_name: str, values: list[str]) -> list[str]:
    """Tokenize every item in a list — loads/saves the lookup once per call."""
    lookup  = _load_lookup()
    result  = []
    changed = False
    for value in values:
        reverse_key = f"__rev__{col_name}__{value}"
        if reverse_key in lookup:
            result.append(lookup[reverse_key])
        else:
            token = f"{col_name.upper()}_HASH_{uuid.uuid4().hex[:8]}"
            lookup[token]       = value
            lookup[reverse_key] = token
            result.append(token)
            changed = True
    if changed:
        _save_lookup(lookup)
    return result


def detokenize_column(tokens: list[str]) -> list[str]:
    """Detokenize every item in a list — loads the lookup once per call."""
    lookup = _load_lookup()
    return [lookup.get(t, t) for t in tokens]


# ── result-level helper (call after sql_executor) ─────────────────────────────

def detokenize_results(records: list[dict]) -> list[dict]:
    """
    Scan every value in a list of result dicts and replace any
    token string with its original value before displaying to the user.
    """
    lookup = _load_lookup()
    return [
        {
            k: lookup.get(str(v), v) if isinstance(v, str) else v
            for k, v in row.items()
        }
        for row in records
    ]
