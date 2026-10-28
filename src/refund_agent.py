import datetime, json, os, sqlite3, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm
import policy_rag

DB = os.path.join("data", "instamart.db")
LEDGER = os.path.join("outputs", "refund_ledger.jsonl")
STATE_DIR = "state"
_CTX = {"run_id": None}


class SimulatedCrash(Exception):
    """Raised on purpose to demonstrate crash-and-resume."""


# ---------------- Task 5: tools ----------------
def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes "
                      "FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1],
            "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": policy_rag.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}


def _ledger(tool, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    line = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "tool": tool, "run_id": _CTX["run_id"],
            "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def issue_refund(order_id, refund_inr):
    _ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def _spec(name, description, props):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}

TOOLS_SPEC = [
    _spec("get_order", "Look up an order in the database by its order id.",
          {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents and return the best matching chunks.",
          {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}

AGENT_PROMPT = ("You are a refund AGENT for a quick-commerce support desk. Decide the refund "
                "request by calling tools one at a time: get_order, search_policy, "
                "calculate_refund, issue_refund, escalate_to_human. Finish with one line that "
                "starts with REFUND, ESCALATE, DENIED or NOT_FOUND, then a colon and a reason.")


# ---------------- Task 8: checkpoints ----------------
def _checkpoint_path(run_id):
    return os.path.join(STATE_DIR, f"{run_id}.json")


def _save_checkpoint(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    keys = ("run_id", "scenario_id", "messages", "tokens_used", "llm_calls", "steps",
            "tools_executed", "max_iterations", "token_limit", "time_limit_seconds")
    data = {k: state[k] for k in keys}
    data["step"] = len(state["steps"])
    with open(_checkpoint_path(state["run_id"]), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------- Tasks 6 and 7: the ReAct loop with safe stops ----------------
def _result(state, status, final):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"],
            "status": status, "final_answer": final, "steps": state["steps"],
            "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"],
            "tools_executed": list(state["tools_executed"])}


def _fallback(state, status):
    done = ", ".join(state["tools_executed"]) or "none"
    final = (f"{status}: stopped safely after {state['llm_calls']} model calls and "
             f"{state['tokens_used']} tokens. Tools completed: {done}. "
             f"Please hand this case to a human agent.")
    return _result(state, status, final)


def _loop(state, crash_after_tool=None):
    _CTX["run_id"] = state["run_id"]
    t0 = time.monotonic()
    while state["llm_calls"] < state["max_iterations"]:
        if time.monotonic() - t0 >= state["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > state["token_limit"]:
            return _fallback(state, "FALLBACK_TOKEN_LIMIT")   # tool NOT executed
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content
            return _result(state, final.split(":")[0].strip(), final)
        tc = msg.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name,
                                                  "arguments": tc.arguments}]})
        fn = TOOL_MAP.get(tc.name)
        try:
            obs = fn(**tc.arguments) if fn else {"error": f"unknown tool {tc.name}"}
        except Exception as e:
            obs = {"error": str(e)}
        state["messages"].append({"role": "tool", "tool_call_id": tc.id,
                                  "name": tc.name, "content": json.dumps(obs)})
        state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        state["tools_executed"].append(tc.name)
        _save_checkpoint(state)
        if crash_after_tool and len(state["tools_executed"]) == crash_after_tool:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tool}")
    return _fallback(state, "FALLBACK_MAX_ITERATIONS")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tool=None):
    state = {"run_id": run_id or f"run_{scenario_id}", "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": AGENT_PROMPT},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "steps": [], "tools_executed": [],
             "max_iterations": max_iterations, "token_limit": token_limit,
             "time_limit_seconds": time_limit_seconds}
    return _loop(state, crash_after_tool)


def resume_agent(run_id):
    with open(_checkpoint_path(run_id), encoding="utf-8") as f:
        state = json.load(f)
    return _loop(state, None)