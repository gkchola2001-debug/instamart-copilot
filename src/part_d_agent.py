import json, os, shutil
import pandas as pd
import refund_agent as ra

RUNS, DEMOS = "outputs/agent_runs.jsonl", "outputs/agent_demos.jsonl"
os.makedirs("outputs", exist_ok=True)
for p in (RUNS, DEMOS, ra.LEDGER):
    if os.path.exists(p):
        os.remove(p)
shutil.rmtree("state", ignore_errors=True)
os.makedirs("state")


def refund_of(res):
    amt = 0.0
    for s in res["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            amt = s["observation"]["refund_inr"]
    return amt


# ---------- Task 6: the seven scenarios ----------
sc = pd.read_csv("data/raw/agent_scenarios.csv")
table = ["| Scenario | Status | Refund (INR) | Model calls | Tokens used |", "|---|---|---|---|---|"]
for r in sc.itertuples():
    res = ra.run_agent(r.message, r.scenario_id)
    amt = refund_of(res)
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: res[k] for k in ("scenario_id", "status", "final_answer",
                                                "llm_calls", "tokens_used", "steps")}) + "\n")
    expected = r.expected_outcome if r.expected_outcome != "FALLBACK" else "FALLBACK_MAX_ITERATIONS"
    ok = res["status"] == expected and abs(amt - float(r.expected_refund_inr)) <= 0.01
    print(r.scenario_id, res["status"], f"{amt:.2f}", "calls", res["llm_calls"],
          "tokens", res["tokens_used"], "-> OK" if ok else "-> MISMATCH")
    table.append(f"| {r.scenario_id} | {res['status']} | {amt:.2f} | "
                 f"{res['llm_calls']} | {res['tokens_used']} |")
print("\n--- COPY THIS TABLE INTO THE README ---")
print("\n".join(table))

msg_s1 = sc.iloc[0]["message"]

# ---------- Task 7: stop-safely demos ----------
d1 = ra.run_agent(msg_s1, "S1", token_limit=200, run_id="S1_token200_demo")
d2 = ra.run_agent(msg_s1, "S1", time_limit_seconds=0, run_id="S1_timeout_demo")
with open(DEMOS, "a", encoding="utf-8") as f:
    for d in (d1, d2):
        f.write(json.dumps({k: d[k] for k in ("run_id", "status", "final_answer",
                                              "llm_calls", "tokens_used", "steps")}) + "\n")
print("\nDEMO S1 with token_limit=200 ->", d1["status"], "| model calls:", d1["llm_calls"])
print("DEMO S1 with time_limit_seconds=0 ->", d2["status"], "| model calls:", d2["llm_calls"])

# ---------- Task 8: crash and resume ----------
try:
    ra.run_agent(msg_s1, "S1", run_id="S1_crash_demo", crash_after_tool=2)
    print("ERROR: the crash did not happen")
except ra.SimulatedCrash as e:
    print("\nCRASH:", e)
res = ra.resume_agent("S1_crash_demo")
print("RESUMED -> status:", res["status"], "| refund:", f"{refund_of(res):.2f}")
print("Tools executed across crash + resume:", res["tools_executed"])
assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert res["status"] == "REFUND" and abs(refund_of(res) - 128.25) <= 0.01
print("CRASH-AND-RESUME CHECK PASSED")