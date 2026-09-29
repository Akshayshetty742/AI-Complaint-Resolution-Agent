import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.prompts import PromptManager
from src.complaint_analyzer import ComplaintAnalyzer
from src.complaint_agent import ComplaintResolutionAgent


def test_complaint_classifications():
    """Tests complaint classification for cases A through G."""
    analyzer = ComplaintAnalyzer()
    test_cases = [
        {
            "id": "A",
            "text": "I cancelled my order five days ago but I still haven't received my refund.",
            "expected_cat": "Refund"
        },
        {
            "id": "B",
            "text": "My laptop stopped working after two months. Is it covered under warranty?",
            "expected_cat": "Warranty"
        },
        {
            "id": "C",
            "text": "My order was supposed to arrive yesterday. Can you tell me when it will be delivered?",
            "expected_cat": "Delivery"
        },
        {
            "id": "D",
            "text": "My payment was deducted but my order was not confirmed.",
            "expected_cat": "Payment"
        },
        {
            "id": "E",
            "text": "My headphones arrived damaged yesterday and I want a replacement.",
            "expected_cat": "Damaged Product"
        },
        {
            "id": "F",
            "text": "Can I return these shoes within the return period?",
            "expected_cat": "Return"
        },
        {
            "id": "G",
            "text": "I need help changing the address associated with my account.",
            "expected_cat": "General Support"
        }
    ]

    for tc in test_cases:
        res = analyzer.analyze(tc["text"])
        cat = res["category"]
        print(f"[{tc['id']}] Text: {tc['text']} -> Result: {cat} (Expected: {tc['expected_cat']})")
        assert cat == tc["expected_cat"], f"Failed on case {tc['id']}: expected {tc['expected_cat']}, got {cat}"


def test_basic_pipeline():
    print("=== Testing Basic Pipeline & Classifications ===")

    # 1. Test Prompt Manager
    pm = PromptManager()
    optimized_prompt = pm.get_template("optimized")
    assert "CRITICAL INSTRUCTIONS" in optimized_prompt
    print(" PromptManager loaded templates successfully.")

    # 2. Run Classification Tests A-G
    test_complaint_classifications()

    # 3. Test Full Agent Resolution Flow
    agent = ComplaintResolutionAgent()
    resolution_res = agent.resolve("My headphones arrived damaged yesterday and I want a replacement.")
    assert "category" in resolution_res
    assert "retrieved_chunks" in resolution_res
    assert len(resolution_res["retrieved_chunks"]) > 0
    print("\n ComplaintResolutionAgent full flow executed cleanly.")
    print("\nAll basic tests PASSED successfully!")


def test_evaluation_benchmark():
    """Validates the benchmark evaluation pipeline over the 21 test complaints."""
    from evaluation.evaluate import run_benchmark
    report = run_benchmark()
    assert report["total_complaints"] == 21
    assert report["category_match_rate"] >= 80.0
    assert report["valid_json_rate"] == 100.0
    assert report["retrieval_match_rate"] >= 70.0


if __name__ == "__main__":
    test_basic_pipeline()
    test_evaluation_benchmark()
