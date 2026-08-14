"""
Simple evaluation script for the RAG pipeline.
Tests whether retrieved chunks contain the expected keywords for each question.
"""
import sys
from app import build_retriever, get_reranker

# --- Define your test cases here ---
# Each test: a question, and keywords that SHOULD appear in a good retrieval
TEST_CASES = [
    {
        "question": "What is climate change?",
        "expected_keywords": ["climate", "long-term", "weather"],
    },
    {
        "question": "What causes climate change?",
        "expected_keywords": ["greenhouse", "fossil", "carbon"],
    },
    {
        "question": "What are the effects of climate change?",
        "expected_keywords": ["temperature", "sea level", "weather"],
    },
    {
        "question": "What can be done to address climate change?",
        "expected_keywords": ["renewable", "emissions", "energy"],
    },
]

def evaluate(pdf_path, top_n=3):
    print(f"Building retriever for: {pdf_path}\n")
    retriever = build_retriever(pdf_path)
    reranker = get_reranker()

    results = []
    for case in TEST_CASES:
        question = case["question"]
        expected = [kw.lower() for kw in case["expected_keywords"]]

        retrieved_docs = retriever.invoke(question)
        pairs = [[question, doc.page_content] for doc in retrieved_docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:top_n]]

        combined_text = " ".join(doc.page_content for doc in top_docs).lower()
        hits = [kw for kw in expected if kw.lower() in combined_text]
        hit_rate = len(hits) / len(expected) if expected else 0

        results.append({
            "question": question,
            "expected": expected,
            "found": hits,
            "hit_rate": hit_rate,
        })

        status = "✅ PASS" if hit_rate == 1.0 else ("⚠️ PARTIAL" if hit_rate > 0 else "❌ FAIL")
        print(f"{status}  [{hit_rate:.0%}]  {question}")
        print(f"   Expected: {expected}")
        print(f"   Found:    {hits}\n")

    avg_hit_rate = sum(r["hit_rate"] for r in results) / len(results)
    print(f"\n=== Overall retrieval accuracy: {avg_hit_rate:.1%} across {len(results)} test questions ===")
    return results

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "Understanding_Climate_Change.pdf"
    evaluate(pdf_path)