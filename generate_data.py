"""
generate_data.py

Run this once to:
  1. Create sample branch/resource CSV files under data/
  2. Write schema_registry.json

Directory layout produced:
  data/
    branch_a/
      products/   products_2025-01-15.csv  products_2025-03-01.csv  ...
      sales/      sales_2025-01-15.csv     ...
      inventory/  inventory_2025-01-15.csv ...
    branch_b/ ...
    branch_c/ ...

Sensitive columns (price, cost, total_amount) are encrypted at rest
using the Fernet key in secret.key.

Usage:
  uv run generate_data.py
"""

import os
import json
import random
import pandas as pd
from datetime import date, timedelta
from hasher import tokenize_column


# ── static config ─────────────────────────────────────────────────────────────

BRANCHES = ["branch_a", "branch_b", "branch_c"]

# Snapshot dates – these become the CSV file-name timestamps
SNAPSHOTS = ["2025-01-15", "2025-03-01", "2025-05-01"]

# (product_id, name, category)
PRODUCTS = [
    ("P001", "Almond Milk",        "Dairy-Free"),
    ("P002", "Oat Milk",           "Dairy-Free"),
    ("P003", "Whole Milk",         "Dairy"),
    ("P004", "Greek Yogurt",       "Dairy"),
    ("P005", "Cheddar Cheese",     "Dairy"),
    ("P006", "Sourdough Bread",    "Bakery"),
    ("P007", "Whole Wheat Bread",  "Bakery"),
    ("P008", "Organic Eggs",       "Protein"),
    ("P009", "Chicken Breast",     "Protein"),
    ("P010", "Salmon Fillet",      "Protein"),
]

BASE_PRICES = {"Dairy-Free": 4.50, "Dairy": 3.50, "Bakery": 2.50, "Protein": 7.00}

# Columns to encrypt before writing to disk
SENSITIVE = {
    "products":  ["price", "cost"],
    "sales":     ["total_amount"],
    "inventory": [],
}


# ── row generators ────────────────────────────────────────────────────────────

def _rand_price(base: float, rng: random.Random) -> float:
    return round(base + rng.uniform(-0.50, 0.50), 2)


def make_products(branch: str, snapshot: str) -> pd.DataFrame:
    rng = random.Random(f"{branch}-{snapshot}-products")
    rows = []
    for pid, name, cat in PRODUCTS:
        price = _rand_price(BASE_PRICES[cat], rng)
        rows.append({
            "product_id":   pid,
            "product_name": name,
            "category":     cat,
            "price":        price,
            "cost":         round(price * 0.60, 2),
            "in_stock":     1 if rng.choice([True, True, True, False]) else 0,
        })
    df = pd.DataFrame(rows)
    for col in SENSITIVE["products"]:
        df[col] = tokenize_column(col, df[col].astype(str).tolist())
    return df


def make_sales(branch: str, snapshot: str) -> pd.DataFrame:
    rng = random.Random(f"{branch}-{snapshot}-sales")
    snap_date = date.fromisoformat(snapshot)
    rows = []
    for i in range(20):
        pid, _, cat = rng.choice(PRODUCTS)
        qty   = rng.randint(1, 10)
        total = round(_rand_price(BASE_PRICES[cat], rng) * qty, 2)
        rows.append({
            "sale_id":      f"S{i + 1:04d}",
            "product_id":   pid,
            "quantity":     qty,
            "total_amount": total,
            "sale_date":    (snap_date - timedelta(days=rng.randint(0, 14))).isoformat(),
        })
    df = pd.DataFrame(rows)
    for col in SENSITIVE["sales"]:
        df[col] = tokenize_column(col, df[col].astype(str).tolist())
    return df


def make_inventory(branch: str, snapshot: str) -> pd.DataFrame:
    rng = random.Random(f"{branch}-{snapshot}-inventory")
    rows = []
    for pid, _, _ in PRODUCTS:
        rows.append({
            "product_id":       pid,
            "quantity_on_hand": rng.randint(0, 200),
            "reorder_level":    20,
            "last_updated":     snapshot,
        })
    return pd.DataFrame(rows)


MAKERS = {
    "products":  make_products,
    "sales":     make_sales,
    "inventory": make_inventory,
}


# ── schema registry ───────────────────────────────────────────────────────────

# Column-level schema shared across all branches
COLUMN_SCHEMAS = {
    "products": {
        "product_id":   {"is_sensitive": False, "description": "Unique product identifier"},
        "product_name": {"is_sensitive": False, "description": "Name of the product"},
        "category":     {"is_sensitive": False, "description": "Product category"},
        "price":        {"is_sensitive": True,  "description": "Selling price (encrypted at rest)"},
        "cost":         {"is_sensitive": True,  "description": "Branch cost price (encrypted at rest)"},
        "in_stock":     {"is_sensitive": False, "description": "1 if in stock, 0 if out of stock"},
    },
    "sales": {
        "sale_id":      {"is_sensitive": False, "description": "Unique sale identifier"},
        "product_id":   {"is_sensitive": False, "description": "Product sold"},
        "quantity":     {"is_sensitive": False, "description": "Units sold"},
        "total_amount": {"is_sensitive": True,  "description": "Total transaction amount (encrypted at rest)"},
        "sale_date":    {"is_sensitive": False, "description": "Date of the sale (YYYY-MM-DD)"},
    },
    "inventory": {
        "product_id":       {"is_sensitive": False, "description": "Product identifier"},
        "quantity_on_hand": {"is_sensitive": False, "description": "Current stock quantity"},
        "reorder_level":    {"is_sensitive": False, "description": "Minimum stock level before reorder"},
        "last_updated":     {"is_sensitive": False, "description": "Last inventory update date"},
    },
}

BRANCH_NAMES = {
    "branch_a": "FreshMart Downtown",
    "branch_b": "FreshMart Westside",
    "branch_c": "FreshMart Northpark",
}


def build_schema_registry() -> dict:
    registry = {
        "general_metadata": {
            "company_name": "FreshMart Co.",
            "about": (
                "FreshMart Co. is a regional grocery chain with three branches. "
                "We focus on fresh, locally sourced produce and pantry staples."
            ),
            "announcements": [
                "Opening a new East Side branch in Q4 2025.",
                "Loyalty programme now gives 2× points on all Dairy-Free products.",
                "Online ordering is live for Branch A and Branch B.",
            ],
            "contact": "support@freshmart.com",
        },
        "branches": {},
    }

    for branch in BRANCHES:
        branch_entry = {
            "name": BRANCH_NAMES[branch],
            "path": f"data/{branch}",
            "resources": {},
        }
        for resource in MAKERS:
            resource_path = os.path.join("data", branch, resource)
            files = sorted(
                f for f in os.listdir(resource_path) if f.endswith(".csv")
            ) if os.path.isdir(resource_path) else []

            branch_entry["resources"][resource] = {
                "path":    resource_path,
                "files":   files,
                "columns": COLUMN_SCHEMAS[resource],
            }
        registry["branches"][branch] = branch_entry

    return registry


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    for branch in BRANCHES:
        for resource, maker in MAKERS.items():
            folder = os.path.join("data", branch, resource)
            os.makedirs(folder, exist_ok=True)

            for snapshot in SNAPSHOTS:
                df   = maker(branch, snapshot)
                path = os.path.join(folder, f"{resource}_{snapshot}.csv")
                df.to_csv(path, index=False)
                print(f"  wrote {path}")

    registry = build_schema_registry()
    with open("schema_registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    print("\nschema_registry.json written.")


if __name__ == "__main__":
    main()
