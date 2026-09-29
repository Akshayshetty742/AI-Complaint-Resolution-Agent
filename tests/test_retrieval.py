import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.vector_store import VectorStore
from src.retriever import PolicyRetriever


def test_retrieval_pipeline():
    print("=== Testing Policy Retrieval Pipeline ===")
    vs = VectorStore()
    count = vs.build_index(force_rebuild=True)
    print(f"Index built successfully with {count} chunks.")

    retriever = PolicyRetriever(vector_store=vs, top_k=2)

    test_cases = [
        {
            "query": "My headphones arrived damaged yesterday and I want a replacement.",
            "expected_doc": "damaged_product_policy.txt"
        },
        {
            "query": "I cancelled my order five days ago but I still haven't received my refund.",
            "expected_doc": "refund_policy.txt"
        },
        {
            "query": "My laptop stopped working after two months. Is it covered?",
            "expected_doc": "warranty_policy.txt"
        },
        {
            "query": "When will my order be delivered?",
            "expected_doc": "delivery_policy.txt"
        }
    ]

    all_passed = True
    for idx, tc in enumerate(test_cases, 1):
        chunks = retriever.retrieve(tc["query"], top_k=2)
        top_source = chunks[0]["source"] if chunks else None
        print(f"\n[Test {idx}] Query: {tc['query']}")
        print(f"Top Retrieved: {top_source} (Score: {chunks[0]['score']:.4f})")
        print(f"Expected Doc:  {tc['expected_doc']}")

        if top_source == tc["expected_doc"]:
            print(" Result: PASS")
        else:
            print(" Result: FAIL (Top doc mismatch)")
            all_passed = False

    assert all_passed, "Some retrieval tests failed."
    print("\nAll retrieval tests PASSED successfully!")


if __name__ == "__main__":
    test_retrieval_pipeline()
