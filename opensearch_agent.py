"""
OpenSearch-based agent path.

This is an experimental alternative to the SQL/DuckDB pipeline:
- Classify relevance
- Route company vs dataset questions
- Resolve datasets/time scope
- Generate OpenSearch DSL JSON
- Execute against indexed documents
- Synthesize final answer
"""

import json
import re
from typing import Any, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from opensearch_backend import DEFAULT_INDEX, get_client

llm = ChatOllama(model="qwen2.5", temperature=0)


class OSAgentState(TypedDict):
    user_query: str
    is_relevant: bool
    query_type: str
    relevant_datasets: list
    target_snapshot: Optional[str]
    os_query: dict[str, Any]
    query_results: dict[str, Any]
    final_response: str
    error: Optional[str]


def _llm(system: str, user: str) -> str:
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content.strip()


def _registry() -> dict[str, Any]:
    with open("schema_registry.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _intent_classifier(query: str) -> bool:
    meta = _registry()["general_metadata"]
    answer = _llm(
        system=(
            f"You are the relevance gate for {meta['company_name']}.\n"
            f"Company context: {meta['about']}\n\n"
            "Decide if the question is relevant to this company or its data.\n"
            "Be generous — if in doubt, say YES.\n\n"
            "Say YES for any of these:\n"
            "  - Company info: name, about, contact, announcements, policies, locations\n"
            "  - Product data: prices, availability, categories, names\n"
            "  - Sales, inventory, or branch data\n"
            "  - Any comparison or question that could relate to a grocery chain\n\n"
            "Say NO only when the question is clearly unrelated to any grocery "
            "business (e.g. sports scores, coding help, medical advice, recipes).\n\n"
            "Examples:\n"
            "  Q: What are your latest announcements?  \u2192 YES\n"
            "  Q: How much does almond milk cost?       \u2192 YES\n"
            "  Q: What is your contact email?           \u2192 YES\n"
            "  Q: Who won the football World Cup?       \u2192 NO\n"
            "  Q: How do I fix a Python bug?            \u2192 NO\n\n"
            "Output ONLY YES or NO."
        ),
        user=query,
    )
    return answer.upper().startswith("YES")


def _route(query: str) -> str:
    answer = _llm(
        system=(
            "Classify as COMPANY or DATASET.\n"
            "COMPANY: announcements, contact, general information.\n"
            "DATASET: questions requiring data lookup or comparison.\n"
            "Output only COMPANY or DATASET."
        ),
        user=query,
    )
    return "dataset" if "DATASET" in answer.upper() else "company"


def _metadata_response(query: str) -> str:
    meta = json.dumps(_registry()["general_metadata"], indent=2)
    return _llm(
        system=(
            "Answer using only metadata. Be concise and factual.\n"
            f"METADATA:\n{meta}"
        ),
        user=query,
    )


def _dataset_resolver(query: str) -> list[dict[str, Any]]:
    reg = _registry()
    schema_summary = {
        branch: {
            resource: {
                "columns": list(rdata["columns"].keys()),
            }
            for resource, rdata in bdata["resources"].items()
        }
        for branch, bdata in reg["branches"].items()
    }

    raw = _llm(
        system=(
            "Resolve required datasets. Output a JSON array with objects:\n"
            "{\"branch\": ..., \"resource\": ..., \"columns\": [...]}\n"
            "Resources are one of products, sales, inventory.\n"
            "If question compares branches or asks 'which branch', include all branches.\n"
            "If sales/inventory data is needed with product names/categories, include products too.\n"
            "Output only JSON.\n"
            f"SCHEMA:\n{json.dumps(schema_summary, indent=2)}"
        ),
        user=query,
    )

    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        arr = json.loads(clean)
    except json.JSONDecodeError:
        arr = []

    enriched: list[dict[str, Any]] = []
    for entry in arr:
        branch = entry.get("branch", "")
        resource = entry.get("resource", "")
        if branch not in reg["branches"]:
            continue
        if resource not in reg["branches"][branch]["resources"]:
            continue
        rdata = reg["branches"][branch]["resources"][resource]
        enriched.append(
            {
                "branch": branch,
                "resource": resource,
                "files": rdata["files"],
                "path": rdata["path"],
                "all_columns": list(rdata["columns"].keys()),
            }
        )

    existing = {(d["branch"], d["resource"]) for d in enriched}
    for d in list(enriched):
        if d["resource"] in {"sales", "inventory"}:
            key = (d["branch"], "products")
            if key not in existing:
                pdata = reg["branches"][d["branch"]]["resources"]["products"]
                enriched.append(
                    {
                        "branch": d["branch"],
                        "resource": "products",
                        "files": pdata["files"],
                        "path": pdata["path"],
                        "all_columns": list(pdata["columns"].keys()),
                    }
                )
                existing.add(key)

    return enriched


def _temporal_resolver(query: str, datasets: list[dict[str, Any]]) -> tuple[Optional[str], list[dict[str, Any]]]:
    answer = _llm(
        system=(
            "Extract date scope.\n"
            "If specific date appears, output YYYY-MM-DD.\n"
            "If month/year appears, output first day of month.\n"
            "If no date, output LATEST.\n"
            "Output only one token."
        ),
        user=query,
    ).strip()

    target = None if answer == "LATEST" else answer

    resolved = []
    for ds in datasets:
        files = sorted(ds["files"])
        if not files:
            continue
        if answer == "LATEST":
            chosen = files[-1]
        else:
            chosen = files[0]
            for fname in files:
                sdate = fname.rsplit("_", 1)[-1].replace(".csv", "")
                if sdate <= answer:
                    chosen = fname
                elif sdate > answer:
                    break
        resolved.append(
            {
                **ds,
                "snapshot": chosen.rsplit("_", 1)[-1].replace(".csv", ""),
            }
        )

    return target, resolved


def _generate_opensearch_query(query: str, datasets: list[dict[str, Any]], target_snapshot: Optional[str]) -> dict[str, Any]:
    branches = sorted({d["branch"] for d in datasets})
    resources = sorted({d["resource"] for d in datasets})
    snapshots = sorted({d["snapshot"] for d in datasets if d.get("snapshot")})

    available_fields = sorted({c for d in datasets for c in d["all_columns"]})

    raw = _llm(
        system=(
            "Generate an OpenSearch search body JSON for the user question.\n"
            "Output JSON object only (no markdown, no explanation).\n"
            "Must be compatible with POST /{index}/_search.\n"
            "Use aggregations (sum, avg, terms) when the question asks for totals, averages, or groupings.\n"
            "Never use script queries.\n"
            "Prefer size <= 50, or size 0 when only aggregations are needed.\n\n"
            "IMPORTANT — do NOT add filters for branch, resource, or snapshot.\n"
            "Those are added automatically after your query. Only add filters\n"
            "for field values the user explicitly mentions (e.g. stock_quantity=0,\n"
            "a product name, a category).\n\n"
            f"Available data fields (use exact names): {available_fields}\n"
            "Common field mappings:\n"
            "  - sales total  → sum of 'total_amount'\n"
            "  - out of stock → filter stock_quantity = 0 or stock_quantity <= 0\n"
            "  - price        → field 'price'\n"
            "  - product name → field 'product_name'\n"
        ),
        user=query,
    )

    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        body = json.loads(clean)
    except json.JSONDecodeError:
        body = {"query": {"match_all": {}}, "size": 25}

    if "query" not in body:
        body["query"] = {"match_all": {}}

    if "bool" not in body["query"]:
        body["query"] = {"bool": {"must": [body["query"]], "filter": []}}

    qbool = body["query"]["bool"]
    qbool.setdefault("filter", [])

    if branches:
        qbool["filter"].append({"terms": {"branch": branches}})
    if resources:
        qbool["filter"].append({"terms": {"resource": resources}})
    if snapshots:
        qbool["filter"].append({"terms": {"snapshot": snapshots}})

    if target_snapshot:
        qbool["filter"].append({"range": {"snapshot": {"lte": target_snapshot}}})

    if "size" not in body:
        body["size"] = 25

    return body


def _validate_query(body: dict[str, Any]) -> tuple[bool, Optional[str]]:
    text = json.dumps(body).lower()
    forbidden = ["delete", "update", "_delete_by_query", "_update_by_query", "script"]
    for kw in forbidden:
        if kw in text:
            return False, f"Rejected for safety: found forbidden term '{kw}'."
    return True, None


def _execute_query(body: dict[str, Any]) -> dict[str, Any]:
    client = get_client()
    res = client.search(index=DEFAULT_INDEX, body=body)

    hits = [h.get("_source", {}) for h in res.get("hits", {}).get("hits", [])]
    aggs = res.get("aggregations", {})
    return {
        "total": res.get("hits", {}).get("total", {}),
        "hits": hits,
        "aggregations": aggs,
    }


def _synthesize_answer(query: str, results: dict[str, Any]) -> str:
    return _llm(
        system=(
            "You are a helpful grocery data analyst.\n"
            "Summarize result JSON clearly and concisely.\n"
            "For currency fields, use dollar format where appropriate."
        ),
        user=(
            f"Question: {query}\n\n"
            f"Result JSON:\n{json.dumps(results, indent=2, default=str)}"
        ),
    )


def run_agent_opensearch(query: str, verbose: bool = False) -> str:
    state: OSAgentState = {
        "user_query": query,
        "is_relevant": False,
        "query_type": "",
        "relevant_datasets": [],
        "target_snapshot": None,
        "os_query": {},
        "query_results": {},
        "final_response": "",
        "error": None,
    }

    state["is_relevant"] = _intent_classifier(query)
    if not state["is_relevant"]:
        return (
            "I can only answer questions about FreshMart Co. and its data. "
            "Please ask something related to products, branches, inventory, or sales."
        )

    state["query_type"] = _route(query)
    if state["query_type"] == "company":
        return _metadata_response(query)

    state["relevant_datasets"] = _dataset_resolver(query)
    state["target_snapshot"], state["relevant_datasets"] = _temporal_resolver(
        query, state["relevant_datasets"]
    )

    state["os_query"] = _generate_opensearch_query(
        query,
        state["relevant_datasets"],
        state["target_snapshot"],
    )

    ok, err = _validate_query(state["os_query"])
    if not ok:
        state["error"] = err
        return f"OpenSearch query rejected: {err}"

    try:
        state["query_results"] = _execute_query(state["os_query"])
    except Exception as e:  # pragma: no cover
        state["error"] = f"OpenSearch execution error: {e}"
        return (
            "I could not execute the OpenSearch query. "
            "Ensure OpenSearch is running and data has been indexed.\n"
            f"Details: {state['error']}"
        )

    if verbose:
        print(f"\n[debug] backend        : opensearch")
        print(f"[debug] os_query       : {json.dumps(state['os_query'], indent=2)}")
        print(f"[debug] target_snapshot: {state.get('target_snapshot')}")
        print(f"[debug] error          : {state.get('error')}\n")

    return _synthesize_answer(query, state["query_results"])
