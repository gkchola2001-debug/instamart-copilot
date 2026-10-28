import collections
import pandas as pd
import policy_rag as rag

# ---------- Task 1: chunk counts ----------
chunks = rag.load_chunks(40, 10)
counts = collections.Counter(c["source"] for c in chunks)
print("Total chunks:", len(chunks))
for name, n in sorted(counts.items()):
    print(" ", name, n)

# ---------- Task 2: retrieval demo ----------
print("\nTop 3 for 'refund policy late delivery':")
for r in rag.retrieve("refund policy late delivery", k=3):
    print(" ", r["source"], round(r["score"], 3))

# ---------- Task 4: evaluation ----------
qa = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)

def evaluate(retriever, label):
    doc_hits = fact_hits = fallback_hits = 0
    n_ans = n_unans = 0
    failures = []
    for row in qa.itertuples():
        text, sources = rag.answer(row.question, retriever)
        if row.expected_doc == "NONE":
            n_unans += 1
            if text.startswith("I could not find"):
                fallback_hits += 1
            else:
                failures.append((row.question, "should have said 'could not find'", text))
        else:
            n_ans += 1
            d = row.expected_doc in sources
            f = row.expected_fact in text
            doc_hits += d
            fact_hits += f
            if not (d and f):
                failures.append((row.question,
                                 f"doc in top3: {d}, fact in answer: {f}", text))
    print(f"\n===== {label} =====")
    print(f"answerable ({n_ans}): expected doc in top 3: {doc_hits}/{n_ans} | "
          f"answer contains fact: {fact_hits}/{n_ans}")
    print(f"unanswerable ({n_unans}): fallback answer: {fallback_hits}/{n_unans}")
    for q, why, text in failures:
        print("  FAILED:", q, "->", why)
        print("    answer was:", text)

evaluate(rag.Retriever(40, 10), "OVERLAP 10")
evaluate(rag.Retriever(40, 0), "OVERLAP 0")