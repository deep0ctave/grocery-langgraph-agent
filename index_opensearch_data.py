"""
Index FreshMart CSV snapshots into OpenSearch.

Usage:
  c:/Users/avinash/Documents/My_Projects/basic_agent/.venv/Scripts/python.exe index_opensearch_data.py
  c:/Users/avinash/Documents/My_Projects/basic_agent/.venv/Scripts/python.exe index_opensearch_data.py --reset
"""

import sys

from opensearch_backend import DEFAULT_INDEX, index_snapshot_data


def main() -> None:
    reset = "--reset" in sys.argv
    count = index_snapshot_data(index_name=DEFAULT_INDEX, reset=reset)
    print(f"Indexed {count} documents into index '{DEFAULT_INDEX}'.")


if __name__ == "__main__":
    main()
