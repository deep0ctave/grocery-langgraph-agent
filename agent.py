"""
agent.py

Multi-node LangGraph agent for querying FreshMart branch data.

Flow (see diagram in README):

  user_query
    └─► intent_classifier  ──(off-topic)──► END  "cannot respond"
              │ relevant
              ▼
           router  ──(company)──► metadata_responder ──► END
              │ dataset
              ▼
        dataset_resolver        (which branches / resources / columns?)
              │
              ▼
        temporal_resolver       (which snapshot date? if none → latest)
              │
              ▼
        sql_generator           (NL → DuckDB SQL)
              │
              ▼
        schema_validator        (hallucinated table names?)
              │ valid          │ invalid
              ▼                ▼
        sql_sanitizer    response_synthesizer ──► END
      (read-only check)
              │ clean          │ rejected
              ▼                ▼
        sql_executor     response_synthesizer ──► END
     (load CSV, decrypt,
      run DuckDB query)
              │
              ▼
       response_synthesizer ──► END

Usage:
  from agent import run_agent
  answer = run_agent("How do almond milk prices compare in branch_a vs branch_b?")
"""

import json
import os
import re
from typing import TypedDict, Optional

import duckdb
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from hasher import detokenize_column, detokenize_results

# ── LLM (swap model name here if needed) ────────────────────────────────────────
llm = ChatOllama(model="qwen2.5", temperature=0)


# ── helpers ───────────────────────────────────────────────────────────────────

def _llm(system: str, user: str) -> str:
    """Minimal LLM wrapper — returns plain text."""
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content.strip()


def _registry() -> dict:
    with open("schema_registry.json") as f:
        return json.load(f)


# ── state ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    user_query:        str
    is_relevant:       bool
    query_type:        str           # "company" | "dataset"
    relevant_datasets: list          # list of dataset dicts resolved by dataset_resolver
    target_snapshot:   Optional[str] # ISO date or None (means latest)
    sql_query:         str
    sql_is_valid:      bool
    query_results:     list
    final_response:    str
    error:             Optional[str]


# ── nodes ─────────────────────────────────────────────────────────────────────

def intent_classifier(state: AgentState) -> dict:
    """Gate: is the question relevant to FreshMart at all?"""
    meta    = _registry()["general_metadata"]
    company = meta["company_name"]
    about   = meta["about"]
    answer = _llm(
        system=(
            f"You are the relevance gate for {company}.\n"
            f"Company context: {about}\n\n"
            "Decide if the question is relevant to this company or its data.\n"
            "Be generous — if in doubt, say YES.\n\n"
            "Say YES for any of these:\n"
            "  - Company info: name, about, contact, announcements, policies, locations\n"
            "  - Product data: prices, availability, categories, names\n"
            "  - Sales, inventory, or branch data\n"
            "  - Any comparison or question that could relate to a grocery chain\n\n"
            "Say NO only when the question is clearly unrelated to any grocery "
            "business (e.g. sports scores, coding help, medical advice, recipes).\n\n"            "Examples:\n"
            "  Q: What are your latest announcements?  \u2192 YES\n"
            "  Q: How much does almond milk cost?       \u2192 YES\n"
            "  Q: What is your contact email?           \u2192 YES\n"
            "  Q: Who won the football World Cup?       \u2192 NO\n"
            "  Q: How do I fix a Python bug?            \u2192 NO\n\n"            "Output ONLY YES or NO."
        ),
        user=state["user_query"],
    )
    is_relevant = answer.upper().startswith("YES")
    result: dict = {"is_relevant": is_relevant}
    if not is_relevant:
        result["final_response"] = (
            "I can only answer questions about FreshMart Co. and its data. "
            "Please ask something related to our products, branches, or sales."
        )
    return result


def router(state: AgentState) -> dict:
    """Route to 'company' (general info) or 'dataset' (needs a DB query)."""
    answer = _llm(
        system=(
            "Classify a grocery-store question.\n"
            "Reply COMPANY if it's about general info, policies, announcements, "
            "or contact details — anything that does NOT need a database lookup.\n"
            "Reply DATASET if it requires looking up actual data "
            "(prices, sales figures, inventory levels, comparisons).\n\n"
            "Examples:\n"
            "  Q: What is your contact email?                  \u2192 COMPANY\n"
            "  Q: Do you have a loyalty programme?             \u2192 COMPANY\n"
            "  Q: What are this week's announcements?          \u2192 COMPANY\n"
            "  Q: How much does almond milk cost in branch_a?  \u2192 DATASET\n"
            "  Q: Which branch sold the most last month?       \u2192 DATASET\n"
            "  Q: Are any products out of stock?               \u2192 DATASET\n\n"
            "Output ONLY COMPANY or DATASET."
        ),
        user=state["user_query"],
    )
    return {"query_type": "dataset" if "DATASET" in answer.upper() else "company"}


def metadata_responder(state: AgentState) -> dict:
    """Answer company-level questions using only general_metadata."""
    meta = json.dumps(_registry()["general_metadata"], indent=2)
    response = _llm(
        system=(
            "You are a helpful assistant for FreshMart Co.\n"
            "Answer using ONLY the metadata provided. Do not invent information.\n\n"
            f"METADATA:\n{meta}"
        ),
        user=state["user_query"],
    )
    return {"final_response": response}


def dataset_resolver(state: AgentState) -> dict:
    """
    Identify which branches, resources, and columns are needed.
    Returns a trimmed list of dataset descriptors.
    """
    reg = _registry()

    # Build a compact schema summary with resource descriptions
    _res_desc = {
        "products":  "product catalogue — name, category, price, cost, availability",
        "sales":     "sales transactions — what was sold, quantity, amount, date",
        "inventory": "stock levels — quantity on hand, reorder level, last updated",
    }
    schema_summary: dict = {}
    for branch, bdata in reg["branches"].items():
        schema_summary[branch] = {
            resource: {
                "about":   _res_desc.get(resource, resource),
                "columns": list(rdata["columns"].keys()),
            }
            for resource, rdata in bdata["resources"].items()
        }

    raw = _llm(
        system=(
            "You are a data resolver for a grocery chain.\n"
            "Given the schema below and the user question, decide which tables are needed.\n\n"
            "Output a JSON array. Each item must have:\n"
            '  "branch"   : branch name (string)\n'
            '  "resource" : table name — one of: products, sales, inventory (string)\n'
            '  "columns"  : column names needed to answer the question (array of strings)\n\n'
            "Rules:\n"
            "- If the question names a specific branch (e.g. branch_a), include only that branch.\n"
            "- If the question asks to compare branches or asks \'which branch\' without naming one,"
            " include ALL three branches (branch_a, branch_b, branch_c) for each required resource.\n"
            "- If the question asks for product names alongside sales or inventory data,"
            " also include the \'products\' resource for the same branch so a JOIN is possible.\n"
            "Output ONLY valid JSON — no explanation, no markdown fences.\n\n"
            f"SCHEMA:\n{json.dumps(schema_summary, indent=2)}"
        ),
        user=state["user_query"],
    )

    # Strip markdown fences if the LLM adds them
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        datasets = json.loads(clean)
    except json.JSONDecodeError:
        datasets = []

    # Enrich each entry with file list and column metadata from the registry
    enriched = []
    for entry in datasets:
        branch   = entry.get("branch", "")
        resource = entry.get("resource", "")
        if branch not in reg["branches"]:
            continue
        if resource not in reg["branches"][branch]["resources"]:
            continue

        rdata = reg["branches"][branch]["resources"][resource]
        enriched.append({
            "branch":            branch,
            "resource":          resource,
            "path":              rdata["path"],
            "files":             rdata["files"],           # sorted snapshot filenames
            "requested_columns": entry.get("columns", []),
            "all_columns":       list(rdata["columns"].keys()),
            "sensitive_columns": [
                col for col, meta in rdata["columns"].items() if meta["is_sensitive"]
            ],
        })

    # Ensure JOIN-compatible context: if sales/inventory is selected for a branch,
    # include that branch's products table so SQL generation can safely join for names/categories.
    existing = {(d["branch"], d["resource"]) for d in enriched}
    auto_added = []
    for ds in enriched:
        if ds["resource"] not in {"sales", "inventory"}:
            continue
        key = (ds["branch"], "products")
        if key in existing:
            continue
        pdata = reg["branches"][ds["branch"]]["resources"]["products"]
        auto_added.append({
            "branch":            ds["branch"],
            "resource":          "products",
            "path":              pdata["path"],
            "files":             pdata["files"],
            "requested_columns": ["product_id", "product_name", "category"],
            "all_columns":       list(pdata["columns"].keys()),
            "sensitive_columns": [
                col for col, meta in pdata["columns"].items() if meta["is_sensitive"]
            ],
        })
        existing.add(key)

    enriched.extend(auto_added)

    return {"relevant_datasets": enriched}


def temporal_resolver(state: AgentState) -> dict:
    """
    Detect any date / time-period mention in the query.
    If found, pick the closest snapshot on or before that date.
    If not found, use the latest snapshot.
    """
    answer = _llm(
        system=(
            "Extract any date or time period from the user question.\n"
            "If a specific date is mentioned output it as YYYY-MM-DD.\n"
            "If only a month/year is mentioned output the first day of that month.\n"
            "If no date is mentioned output exactly: LATEST\n\n"
            "Examples:\n"
            "  Q: How do prices compare between branches?         \u2192 LATEST\n"
            "  Q: What were sales in January 2025?                \u2192 2025-01-01\n"
            "  Q: Show me inventory as of March 2025              \u2192 2025-03-01\n"
            "  Q: What happened on 15 February 2025?             \u2192 2025-02-15\n\n"
            "Output ONLY the date string or LATEST — nothing else."
        ),
        user=state["user_query"],
    )
    target = answer.strip()

    resolved = []
    for ds in state["relevant_datasets"]:
        files = sorted(ds["files"])   # e.g. ["products_2025-01-15.csv", ...]
        if not files:
            continue

        if target == "LATEST":
            chosen = files[-1]
        else:
            # Pick the latest snapshot whose date is <= target
            chosen = files[0]
            for f in files:
                snap_date = f.rsplit("_", 1)[-1].replace(".csv", "")
                if snap_date <= target:
                    chosen = f
                # If we've gone past the target date we stop
                elif snap_date > target:
                    break

        resolved.append({
            **ds,
            "file_path": os.path.join(ds["path"], chosen),
            "snapshot":  chosen.rsplit("_", 1)[-1].replace(".csv", ""),
        })

    return {
        "target_snapshot":   None if target == "LATEST" else target,
        "relevant_datasets": resolved,
    }


def sql_generator(state: AgentState) -> dict:
    """Translate the user question into a DuckDB SQL query."""
    q_lower = state["user_query"].lower()

    # Deterministic fallback for a known high-value pattern that small local models often miss.
    if "most expensive" in q_lower and "each branch" in q_lower:
        branches = sorted(
            {ds["branch"] for ds in state["relevant_datasets"] if ds["resource"] == "products"}
        )
        if branches:
            parts = []
            for branch in branches:
                parts.append(
                    f"SELECT '{branch}' AS branch, p.product_name, p.price\n"
                    f"FROM {branch}_products p\n"
                    f"JOIN (\n"
                    f"  SELECT MAX(price) AS max_price\n"
                    f"  FROM {branch}_products\n"
                    f") m ON p.price = m.max_price"
                )
            return {"sql_query": "\nUNION ALL\n".join(parts)}

    schema_lines = []
    table_names  = []

    for ds in state["relevant_datasets"]:
        table = f"{ds['branch']}_{ds['resource']}"
        cols  = ", ".join(ds["all_columns"])
        schema_lines.append(f"  Table '{table}': columns → {cols}")
        table_names.append(table)

    sql = _llm(
        system=(
            "You write DuckDB SQL SELECT queries.\n"
            "Available tables (already loaded — reference by name only):\n"
            + "\n".join(schema_lines) + "\n\n"
            "Rules:\n"
            "- Output exactly ONE SELECT statement (never two separate SELECTs).\n"
            "- To compare two tables side-by-side, use UNION ALL or a JOIN — not two SELECTs.\n"
            "- When labelling which branch in a UNION ALL, use a literal string "
            "(e.g. SELECT 'branch_a' AS branch ...) — never a CASE expression.\n"
            "- When you need GROUP BY or ORDER BY after a UNION ALL, wrap the whole "
            "UNION ALL in a subquery: SELECT ... FROM (...UNION ALL...) t GROUP BY ... ORDER BY ...\n"
            "- Use ILIKE for case-insensitive string matching "
            "(e.g. product_name ILIKE '%almond milk%').\n"
            "- The snapshot files (products, inventory) do NOT have a date column. "
            "The correct snapshot has already been pre-selected — do NOT add any date filter "
            "to queries on products or inventory tables.\n"
            "- Only the 'sales' table has a sale_date column you may filter on.\n"
            "- When the result should show product names, JOIN the sales or inventory table "
            "with the corresponding products table on product_id.\n"
            "- For per-branch extrema (for example: most expensive product in each branch), "
            "compute the max inside each branch table and join back to that same branch table.\n"
            "- Only SELECT — no INSERT / UPDATE / DELETE / DROP / CREATE / ALTER.\n"
            "- Reference tables by name (e.g. FROM branch_a_products).\n"
            "- Output ONLY the SQL — no explanation, no markdown fences.\n\n"
            "Examples:\n"
            "  Q: Compare almond milk prices across branch_a and branch_b\n"
            "  A:\n"
            "    SELECT branch, product_name, price FROM (\n"
            "      SELECT 'branch_a' AS branch, product_name, price\n"
            "        FROM branch_a_products WHERE product_name ILIKE '%almond milk%'\n"
            "      UNION ALL\n"
            "      SELECT 'branch_b', product_name, price\n"
            "        FROM branch_b_products WHERE product_name ILIKE '%almond milk%'\n"
            "    ) t\n\n"
            "  Q: Which branch has the highest total dairy inventory?\n"
            "  A:\n"
            "    SELECT branch, SUM(quantity_on_hand) AS total FROM (\n"
            "      SELECT 'branch_a' AS branch, i.quantity_on_hand\n"
            "        FROM branch_a_inventory i\n"
            "        JOIN branch_a_products p ON i.product_id = p.product_id\n"
            "        WHERE p.category ILIKE '%dairy%'\n"
            "      UNION ALL\n"
            "      SELECT 'branch_b', i.quantity_on_hand\n"
            "        FROM branch_b_inventory i\n"
            "        JOIN branch_b_products p ON i.product_id = p.product_id\n"
            "        WHERE p.category ILIKE '%dairy%'\n"
            "    ) t GROUP BY branch ORDER BY total DESC LIMIT 1\n\n"
            "  Q: Top 3 best-selling products in branch_b by quantity\n"
            "  A:\n"
            "    SELECT p.product_name, SUM(s.quantity) AS total_qty\n"
            "      FROM branch_b_sales s\n"
            "      JOIN branch_b_products p ON s.product_id = p.product_id\n"
            "      GROUP BY p.product_name\n"
            "      ORDER BY total_qty DESC LIMIT 3\n\n"
            "  Q: What is the price of salmon fillet in branch_a?\n"
            "  A:\n"
            "    SELECT product_name, price\n"
            "      FROM branch_a_products\n"
            "      WHERE product_name ILIKE '%salmon fillet%'\n\n"
            "  Q: What is the price of Greek yogurt in branch_c as of March 2025?\n"
            "  A:\n"
            "    SELECT product_name, price\n"
            "      FROM branch_c_products\n"
            "      WHERE product_name ILIKE '%greek yogurt%'"
        ),
        user=state["user_query"],
    )
    sql = re.sub(r"```(?:sql)?|```", "", sql).strip()
    return {"sql_query": sql}


def schema_validator(state: AgentState) -> dict:
    """
    Lightweight validation:
    - Query must start with SELECT
    - Every table name in FROM / JOIN must be in our resolved dataset list
    """
    sql = state["sql_query"].strip()

    if not sql.lower().startswith("select"):
        return {
            "sql_is_valid":   False,
            "error":          "Generated SQL is not a SELECT statement.",
            "final_response": "I couldn't generate a valid query. Please try rephrasing.",
        }

    known_tables = {
        f"{ds['branch']}_{ds['resource']}"
        for ds in state["relevant_datasets"]
    }

    # Extract table references after FROM and JOIN keywords
    referenced = re.findall(
        r'(?:from|join)\s+([a-z_][a-z0-9_]*)',
        sql.lower()
    )
    unknown = [t for t in referenced if t not in known_tables]

    if unknown:
        return {
            "sql_is_valid":   False,
            "error":          f"SQL references unknown table(s): {unknown}",
            "final_response": "I couldn't generate a valid query (unknown table). Please try rephrasing.",
        }

    return {"sql_is_valid": True}


def sql_sanitizer(state: AgentState) -> dict:
    """
    Safety layer:
    - Block any DDL or write keywords
    - Block multiple statements (semicolon-chained)
    """
    sql = state["sql_query"]
    sql_lower = sql.lower()

    FORBIDDEN = [
        "insert", "update", "delete", "drop", "create", "alter",
        "truncate", "grant", "revoke", "exec", "execute",
    ]
    for kw in FORBIDDEN:
        # Match as a whole word
        if re.search(rf'\b{kw}\b', sql_lower):
            return {
                "sql_query":    "",
                "sql_is_valid": False,
                "error":        f"Rejected: forbidden keyword '{kw}' in SQL.",
                "final_response": f"That query was rejected for safety reasons ('{kw}').",
            }

    # Only one statement allowed
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    if len(statements) > 1:
        return {
            "sql_query":    "",
            "sql_is_valid": False,
            "error":        "Rejected: multiple SQL statements detected.",
            "final_response": "That query was rejected (multiple statements).",
        }

    return {"sql_query": sql}


def sql_executor(state: AgentState) -> dict:
    """
    1. Load each resolved CSV into a pandas DataFrame
    2. Decrypt sensitive columns in memory
    3. Register DataFrames as DuckDB tables
    4. Execute the SQL and return results
    """
    if not state.get("sql_query"):
        return {"query_results": [], "error": "No valid SQL to execute."}

    con = duckdb.connect()

    try:
        for ds in state["relevant_datasets"]:
            table_name = f"{ds['branch']}_{ds['resource']}"
            df = pd.read_csv(ds["file_path"])

            # Detokenize sensitive columns in memory so DuckDB can do arithmetic
            for col in ds["sensitive_columns"]:
                if col in df.columns:
                    detokenized = detokenize_column(df[col].astype(str).tolist())
                    df[col] = pd.to_numeric(detokenized, errors="coerce")

            con.register(table_name, df)

        result_df = con.execute(state["sql_query"]).df()
        # Detokenize any tokens that surfaced in the result set before returning
        records = detokenize_results(result_df.to_dict(orient="records"))
        return {"query_results": records}

    except Exception as e:
        return {"query_results": [], "error": f"SQL execution error: {e}"}

    finally:
        con.close()


def response_synthesizer(state: AgentState) -> dict:
    """Turn raw query results (or an error) into a friendly natural-language answer."""
    # If a pre-built response already exists (e.g. from sanitizer / validator), keep it
    if state.get("final_response"):
        return {}

    results_json = json.dumps(state.get("query_results", []), indent=2, default=str)
    response = _llm(
        system=(
            "You are a helpful data analyst for FreshMart Co.\n"
            "Summarize the query results in clear, conversational language.\n"
            "Keep it concise. Format currency values with a $ sign and 2 decimal places."
        ),
        user=(
            f"Original question: {state['user_query']}\n\n"
            f"Query results:\n{results_json}"
        ),
    )
    return {"final_response": response}


# ── routing functions ─────────────────────────────────────────────────────────

def _route_intent(state: AgentState) -> str:
    return "router" if state["is_relevant"] else "end"


def _route_router(state: AgentState) -> str:
    return "metadata_responder" if state["query_type"] == "company" else "dataset_resolver"


def _route_validator(state: AgentState) -> str:
    return "sql_sanitizer" if state["sql_is_valid"] else "response_synthesizer"


def _route_sanitizer(state: AgentState) -> str:
    return "sql_executor" if state.get("sql_query") else "response_synthesizer"


# ── graph ─────────────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("intent_classifier",    intent_classifier)
    g.add_node("router",               router)
    g.add_node("metadata_responder",   metadata_responder)
    g.add_node("dataset_resolver",     dataset_resolver)
    g.add_node("temporal_resolver",    temporal_resolver)
    g.add_node("sql_generator",        sql_generator)
    g.add_node("schema_validator",     schema_validator)
    g.add_node("sql_sanitizer",        sql_sanitizer)
    g.add_node("sql_executor",         sql_executor)
    g.add_node("response_synthesizer", response_synthesizer)

    # Entry point
    g.set_entry_point("intent_classifier")

    # Edges
    g.add_conditional_edges("intent_classifier", _route_intent, {
        "router": "router",
        "end":    END,
    })
    g.add_conditional_edges("router", _route_router, {
        "metadata_responder": "metadata_responder",
        "dataset_resolver":   "dataset_resolver",
    })
    g.add_edge("metadata_responder",   END)
    g.add_edge("dataset_resolver",     "temporal_resolver")
    g.add_edge("temporal_resolver",    "sql_generator")
    g.add_edge("sql_generator",        "schema_validator")
    g.add_conditional_edges("schema_validator", _route_validator, {
        "sql_sanitizer":       "sql_sanitizer",
        "response_synthesizer": "response_synthesizer",
    })
    g.add_conditional_edges("sql_sanitizer", _route_sanitizer, {
        "sql_executor":         "sql_executor",
        "response_synthesizer": "response_synthesizer",
    })
    g.add_edge("sql_executor",         "response_synthesizer")
    g.add_edge("response_synthesizer", END)

    return g.compile()


# Compiled graph (lazy singleton)
_graph = None


# ── public API ────────────────────────────────────────────────────────────────

def run_agent(query: str, verbose: bool = False) -> str:
    """
    Run the agent on a natural-language query.
    Returns the final answer as a string.
    """
    global _graph
    if _graph is None:
        _graph = _build_graph()

    initial: AgentState = {
        "user_query":        query,
        "is_relevant":       False,
        "query_type":        "",
        "relevant_datasets": [],
        "target_snapshot":   None,
        "sql_query":         "",
        "sql_is_valid":      False,
        "query_results":     [],
        "final_response":    "",
        "error":             None,
    }

    final = _graph.invoke(initial)

    if verbose:
        print(f"\n[debug] sql_query      : {final.get('sql_query')}")
        print(f"[debug] sql_is_valid   : {final.get('sql_is_valid')}")
        print(f"[debug] target_snapshot: {final.get('target_snapshot')}")
        print(f"[debug] error          : {final.get('error')}\n")

    return final.get("final_response") or "No response generated."
