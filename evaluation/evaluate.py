import csv
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.complaint_agent import ComplaintResolutionAgent
from src.llm import LLMClient
from src.config import BASE_DIR

CSV_PATH = BASE_DIR / "evaluation" / "test_complaints.csv"


def load_test_dataset(csv_path: Path = CSV_PATH) -> List[Dict[str, str]]:
    """Loads benchmark test dataset from CSV."""
    cases = []
    if not csv_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append({
                "id": row["id"],
                "complaint": row["complaint"].strip(),
                "expected_category": row["expected_category"].strip(),
                "expected_policy": row["expected_policy"].strip()
            })
    return cases


def run_benchmark(
    agent: ComplaintResolutionAgent = None,
    prompt_type: str = "optimized",
    csv_path: Path = CSV_PATH
) -> Dict[str, Any]:
    """
    Evaluates agent against the benchmark dataset.
    Calculates:
    - category match rate
    - retrieval policy match rate
    - valid structured JSON rate
    """
    agent = agent or ComplaintResolutionAgent()
    dataset = load_test_dataset(csv_path)

    total = len(dataset)
    category_matches = 0
    retrieval_matches = 0
    valid_json_count = 0
    detailed_results = []

    for item in dataset:
        result = agent.resolve(item["complaint"], prompt_type=prompt_type)

        pred_cat = result.get("category", "")
        chunks = result.get("retrieved_chunks", [])
        top_doc = chunks[0]["source"] if chunks else ""
        is_valid_json = result.get("is_valid_json", False)

        cat_correct = (pred_cat.lower() == item["expected_category"].lower())
        retrieval_correct = (top_doc.lower() == item["expected_policy"].lower())

        if cat_correct:
            category_matches += 1
        if retrieval_correct:
            retrieval_matches += 1
        if is_valid_json:
            valid_json_count += 1

        detailed_results.append({
            "id": item["id"],
            "complaint": item["complaint"],
            "expected_category": item["expected_category"],
            "predicted_category": pred_cat,
            "category_match": cat_correct,
            "expected_policy": item["expected_policy"],
            "retrieved_policy": top_doc,
            "policy_match": retrieval_correct,
            "valid_json": is_valid_json,
            "resolution": result.get("resolution", "")[:100] + "...",
            "customer_response": result.get("customer_response", "")[:100] + "..."
        })

    cat_rate = (category_matches / total) * 100 if total else 0.0
    ret_rate = (retrieval_matches / total) * 100 if total else 0.0
    json_rate = (valid_json_count / total) * 100 if total else 0.0

    return {
        "total_complaints": total,
        "prompt_type": prompt_type,
        "category_matches": category_matches,
        "category_match_rate": cat_rate,
        "retrieval_matches": retrieval_matches,
        "retrieval_match_rate": ret_rate,
        "valid_json_count": valid_json_count,
        "valid_json_rate": json_rate,
        "mode": agent.llm.is_configured(),
        "detailed_results": detailed_results
    }


def print_evaluation_report(metrics: Dict[str, Any]):
    print("=" * 65)
    print(f"BENCHMARK REPORT (Prompt: {metrics['prompt_type']})")
    print("=" * 65)
    print(f"Total Test Cases:            {metrics['total_complaints']}")
    print(f"Category Match Rate:         {metrics['category_matches']}/{metrics['total_complaints']} ({metrics['category_match_rate']:.1f}%)")
    print(f"Retrieval Match Rate:        {metrics['retrieval_matches']}/{metrics['total_complaints']} ({metrics['retrieval_match_rate']:.1f}%)")
    print(f"Valid Structured JSON Rate:  {metrics['valid_json_count']}/{metrics['total_complaints']} ({metrics['valid_json_rate']:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    report = run_benchmark()
    print_evaluation_report(report)
