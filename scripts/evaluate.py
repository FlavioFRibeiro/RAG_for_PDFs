from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_genai_core.catalog import load_catalog, validate_catalog  # noqa: E402
from rag_genai_core.config import Settings  # noqa: E402
from rag_genai_core.exceptions import ManualAssistantError  # noqa: E402
from rag_genai_core.runtime import build_runtime  # noqa: E402
from rag_genai_core.schemas import ChatTurn  # noqa: E402

REQUIRED = {
    "id",
    "split",
    "manual_id",
    "question",
    "expected_behavior",
    "expected",
    "evidence_pages",
    "required_terms",
    "category",
    "review_status",
}


def load_cases(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    ids = set()
    for case in cases:
        missing = REQUIRED - set(case)
        if missing:
            raise ValueError(f"{case.get('id', '<unknown>')} is missing {sorted(missing)}")
        if case["id"] in ids:
            raise ValueError(f"Duplicate case id: {case['id']}")
        ids.add(case["id"])
        if case["review_status"] != "human_checked":
            raise ValueError(f"Reference not human checked: {case['id']}")
    if len(cases) != 48:
        raise ValueError(f"Expected 48 cases, found {len(cases)}")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--split", choices=["development", "reserved", "all"], default="development")
    parser.add_argument("--generation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    settings = Settings.from_env()
    cases = load_cases(ROOT / "evals" / "datasets" / "manual_qa.yaml")
    catalog = load_catalog(settings.catalog_path, settings.root_dir)
    validate_catalog(catalog.values())
    unknown = {case["manual_id"] for case in cases} - set(catalog)
    if unknown:
        raise ValueError(f"Unknown manual IDs: {sorted(unknown)}")
    print(f"Validated {len(cases)} human-checked cases.")
    if args.validate_only:
        return 0
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for live evaluation.")

    selected = cases if args.split == "all" else [case for case in cases if case["split"] == args.split]
    runtimes = {
        manual_id: build_runtime(manual_id, settings)
        for manual_id in {case["manual_id"] for case in selected}
    }
    reports = ROOT / "evals" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    checkpoint = reports / f"progress-{args.split}-{'generation' if args.generation else 'retrieval'}.json"
    rows = json.loads(checkpoint.read_text(encoding="utf-8")) if args.resume and checkpoint.is_file() else []
    completed = {row["id"] for row in rows}
    started = time.perf_counter()
    for case in selected:
        if case["id"] in completed:
            continue
        runtime = runtimes[case["manual_id"]]
        history = [ChatTurn("user", item) for item in case.get("history", [])]
        retrieved = runtime.retrieve(case["question"], history)
        pages = [chunk.physical_page for chunk in retrieved]
        relevant = set(case["evidence_pages"])
        ranks = [index + 1 for index, page in enumerate(pages) if page in relevant]
        row = {
            "id": case["id"],
            "retrieved_pages": pages,
            "recall_at_k": (1 if relevant.intersection(pages) else 0) if relevant else None,
            "reciprocal_rank": (1 / min(ranks) if ranks else 0) if relevant else None,
        }
        if args.generation:
            try:
                response = runtime.ask(case["question"], history)
                answer_lower = response.answer.lower().replace(" ", "")
                terms_ok = all(
                    str(term).lower().replace(" ", "") in answer_lower
                    for term in case["required_terms"]
                )
                row.update(
                    {
                        "actual_status": response.status.value,
                        "actual_answer": response.answer,
                        "status_correct": response.status.value == case["expected_behavior"],
                        "required_terms_present": terms_ok,
                        "citation_pages": [source.physical_page for source in response.sources],
                        "latency_ms": response.latency_ms,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    }
                )
            except ManualAssistantError as exc:
                row.update(
                    {
                        "actual_status": "pipeline_error",
                        "status_correct": False,
                        "required_terms_present": False,
                        "citation_pages": [],
                        "error": str(exc),
                    }
                )
        rows.append(row)
        checkpoint.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(case["id"], row)

    retrieval_rows = [row for row in rows if row["recall_at_k"] is not None]
    generation_rows = [row for row in rows if "status_correct" in row]
    citation_labeled = [
        (row, set(next(case["evidence_pages"] for case in cases if case["id"] == row["id"])))
        for row in generation_rows
        if next(case["evidence_pages"] for case in cases if case["id"] == row["id"])
        and row.get("actual_status") == "answered"
    ]
    critical_failures = [
        row["id"]
        for row in generation_rows
        if not row.get("status_correct", False)
        or not row.get("required_terms_present", False)
        or row.get("actual_status") == "pipeline_error"
    ]
    token_rows = [row for row in generation_rows if row.get("input_tokens") is not None]
    report = {
        "sample_size": len(rows),
        "split": args.split,
        "generation": args.generation,
        "retrieval_labeled_cases": len(retrieval_rows),
        "retrieval_recall_at_k": sum(row["recall_at_k"] for row in retrieval_rows) / len(retrieval_rows),
        "retrieval_mrr": sum(row["reciprocal_rank"] for row in retrieval_rows) / len(retrieval_rows),
        "generation_status_accuracy": (
            sum(bool(row["status_correct"]) for row in generation_rows) / len(generation_rows)
            if generation_rows
            else None
        ),
        "required_terms_accuracy": (
            sum(bool(row["required_terms_present"]) for row in generation_rows) / len(generation_rows)
            if generation_rows
            else None
        ),
        "citation_page_accuracy": (
            sum(bool(set(row.get("citation_pages", [])).intersection(relevant)) for row, relevant in citation_labeled)
            / len(citation_labeled)
            if citation_labeled
            else None
        ),
        "pipeline_error_count": sum(row.get("actual_status") == "pipeline_error" for row in generation_rows),
        "critical_failure_count": len(critical_failures),
        "critical_failure_ids": critical_failures,
        "total_input_tokens": sum(row["input_tokens"] for row in token_rows),
        "total_output_tokens": sum(row["output_tokens"] for row in token_rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limitations": "Finite human-checked sample; passing is not a guarantee of universal accuracy.",
        "cases": rows,
    }
    output = reports / f"baseline-{args.split}-{'generation' if args.generation else 'retrieval'}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
