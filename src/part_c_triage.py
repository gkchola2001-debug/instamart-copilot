import json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import mock_llm
from mock_llm import est_tokens, TransientLLMError

random.seed(42)
ALLOWED = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
os.makedirs("outputs", exist_ok=True)

# ---------- Task 2: redaction (this order, these patterns) ----------
PATTERNS = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("PHONE", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("ORDER_ID", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

def redact(text):
    counts = {}
    for name, pat, placeholder in PATTERNS:
        text, n = pat.subn(placeholder, text)
        counts[name] = n
    return text, counts

def check_no_pii(messages):
    for m in messages:
        for name, pat, _ in PATTERNS:
            assert not pat.search(m["content"]), f"{name} leaked into a prompt"

# ---------- Task 3: two prompts ----------
ZERO_SHOT = ('You are a support-ticket classifier. CLASSIFY the ticket into exactly one '
             'category: billing, delivery, technical or fraud. Reply with JSON only, in the '
             'form {"category": "...", "reason": "..."}.')
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: I was charged twice for order [ORDER_ID]. Please refund the extra payment. -> {"category": "billing", "reason": "double charge"}
Ticket: Someone hacked my account and used my wallet balance. -> {"category": "fraud", "reason": "unauthorized use"}
Ticket: The app crashes every time I open the cart. -> {"category": "technical", "reason": "app error"}
Ticket: My order [ORDER_ID] has not arrived and the rider is not picking up calls. -> {"category": "delivery", "reason": "late delivery"}"""
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT

# ---------- Task 4: retries ----------
def call_with_retry(messages, tid, stats):
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            stats["errors"] += 1
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            print(f"RETRY ticket={tid} retry={n} wait={wait:.4f}s")
            time.sleep(wait)

# ---------- Task 5: validation ----------
def parse(content):
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("category") in ALLOWED:
            return "classified_json", data["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (content or "").lower())
        found = {w for w in words if w in ALLOWED}
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""

# ---------- Task 7b: routing ----------
seg = pd.read_csv("outputs/customer_segments.csv")
seg["is_premium"] = seg["is_premium"].astype(str).str.lower() == "true"
SEG_NAME = seg.set_index("customer_id")["segment_name"].to_dict()
PREMIUM = set(seg.loc[seg["is_premium"], "customer_id"])
HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}

def route(status, category, customer_id):
    if status in HUMAN:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if customer_id in PREMIUM:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"

# ---------- the pipeline ----------
def run(tickets, system_prompt, budget=None):
    mock_llm.reset()
    stats = {"errors": 0, "tokens": 0}
    rows = []
    for t in tickets.itertuples():
        status, cat, used = None, "", 0
        if t.text.strip() == "":
            status = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif budget is not None and stats["tokens"] >= budget:
            status = "budget_exhausted"
        else:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": redact(t.text)[0]}]
            check_no_pii(messages)
            resp = call_with_retry(messages, t.ticket_id, stats)
            used = resp.usage.total_tokens
            stats["tokens"] += used
            status, cat = parse(resp.choices[0].message.content)
        rt, pr = route(status, cat, t.customer_id)
        rows.append({"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": status,
                     "category": cat, "segment_name": SEG_NAME.get(t.customer_id, ""),
                     "priority": pr, "route": rt, "tokens_used": used,
                     "true_category": t.true_category})
    return pd.DataFrame(rows), stats

COLS = ["ticket_id", "customer_id", "status", "category", "segment_name",
        "priority", "route", "tokens_used"]

def summarize(name, res, stats):
    ok = res[res["status"].isin(["classified_json", "classified_keyword"])]
    acc = (ok["category"] == ok["true_category"]).mean() * 100
    print(f"\n===== {name} =====")
    print(res["status"].value_counts().to_string())
    print(f"classified: {len(ok)} | accuracy: {acc:.1f}% "
          f"({int((ok['category'] == ok['true_category']).sum())} of {len(ok)})")
    print(f"total tokens: {stats['tokens']} | tokens per classified ticket: "
          f"{stats['tokens'] / max(len(ok), 1):.1f} | TransientLLMErrors: {stats['errors']}")
    return ok

# ================= main =================
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
print("Tickets:", len(tickets))

# redaction report over ALL 164 tickets
tot = {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
redacted_texts = []
for txt in tickets["text"]:
    r, c = redact(txt)
    redacted_texts.append(r)
    for k in tot:
        tot[k] += c[k]
print("\nRedaction substitutions:", tot, "total", sum(tot.values()))
left = sum(len(p.findall(r)) for r in redacted_texts for _, p, _ in PATTERNS)
print("Left after re-applying the patterns:", left)

print("\nEmpty tickets:", int((tickets.text.str.strip() == "").sum()),
      "| long tickets:", int((tickets.text.map(est_tokens) > MAX_INPUT_TOKENS).sum()))

zero, s0 = run(tickets, ZERO_SHOT)
ok0 = summarize("ZERO-SHOT (no budget)", zero, s0)
print("\nZero-shot confusion table (rows = true, columns = predicted):")
print(pd.crosstab(ok0["true_category"], ok0["category"]))

few, s1 = run(tickets, FEW_SHOT)
summarize("FEW-SHOT (no budget)", few, s1)

demo, sd = run(tickets, ZERO_SHOT, budget=6000)
demo[COLS].to_csv("outputs/triage_budget_demo.csv", index=False)
n_ex = int((demo["status"] == "budget_exhausted").sum())
print(f"\nBUDGET DEMO: TOKEN_BUDGET=6000, budget_exhausted tickets = {n_ex}")

final, sf = run(tickets, FEW_SHOT, budget=100000)
final[COLS].to_csv("outputs/triage_results.csv", index=False)
print(f"\nFINAL few-shot run saved: {len(final)} rows")
print(final["route"].value_counts().to_string())
print(final["priority"].value_counts().to_string())