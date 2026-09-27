"""
Standard event format for an agent run.

Why this exists: the UI (Streamlit app, command line) should not care WHICH
"brain" is doing the thinking - a scripted rule-based planner, a local LLM, or
Claude. Every brain is a Python generator that yields events in this format,
and the UI simply displays events as they arrive.

Every event is a plain dict with a "type" key, so it can be printed, saved as
JSON, or sent anywhere. Use the helper functions below to build events rather
than writing the dicts by hand - they guarantee the required fields are there.

The event types, in the order they usually appear:

run_start    The run has begun.
             {"type": "run_start", "question": str, "brain": str}
             brain = which planner is running, e.g. "scripted" or "claude-sonnet-5".

note         Commentary from the brain between steps, e.g. "Shipping Mode shows a
             large spread, drilling into First Class." Shown in the reasoning trail.
             {"type": "note", "text": str}

issue_tree   The current issue tree. Sent when the tree is first built (all branches
             "untested") and again whenever branch statuses change. The UI shows
             the latest one.
             {"type": "issue_tree", "root": str, "branches": [branch, ...]}
             branch = {"name": str, "hypothesis": str,
                       "status": "untested" | "kept" | "dropped",
                       "evidence": str (optional),
                       "sub_branches": [branch, ...] (optional)}

tool_call    The brain is about to run an analysis tool from src/tools.py.
             {"type": "tool_call", "call_id": str, "tool": str, "inputs": dict}
             call_id links this call to its tool_result (e.g. "c1", "c2").

tool_result  The tool's output: the ONLY place numbers come from.
             {"type": "tool_result", "call_id": str, "tool": str,
              "result": dict, "is_error": bool}
             Successful analysis results carry a "result_id" (e.g. "R1") so charts
             and later steps can refer to them.

chart        A chart was saved.
             {"type": "chart", "path": str, "title": str, "chart_type": "bar" | "line",
              "result_id": str}
             result_id = the tool result the chart was drawn from, so the UI can
             redraw it as an interactive chart.

summary      The final report in markdown, executive summary first.
             {"type": "summary", "text": str}

error        Something went wrong and the run could not finish.
             {"type": "error", "message": str}

run_end      The run is over (sent after a summary, or after an error).
             {"type": "run_end", "status": "complete" | "failed", "stats": dict}
             stats = anything useful to show, e.g. {"tool_calls": 7, "seconds": 12}.
"""

import json

# Required fields for each event type (besides "type" itself)
REQUIRED_FIELDS = {
    "run_start": ["question", "brain"],
    "note": ["text"],
    "issue_tree": ["root", "branches"],
    "tool_call": ["call_id", "tool", "inputs"],
    "tool_result": ["call_id", "tool", "result", "is_error"],
    "chart": ["path", "title", "chart_type", "result_id"],
    "summary": ["text"],
    "error": ["message"],
    "run_end": ["status", "stats"],
}

BRANCH_STATUSES = ["untested", "kept", "dropped"]
RUN_STATUSES = ["complete", "failed"]


# --- Building events -----------------------------------------------------------
def run_start(question, brain):
    return {"type": "run_start", "question": question, "brain": brain}


def note(text):
    return {"type": "note", "text": text}


def issue_tree(root, branches):
    return {"type": "issue_tree", "root": root, "branches": branches}


def tool_call(call_id, tool, inputs):
    return {"type": "tool_call", "call_id": call_id, "tool": tool, "inputs": inputs}


def tool_result(call_id, tool, result):
    # A result is an error if the tool returned {"error": ...}
    return {"type": "tool_result", "call_id": call_id, "tool": tool,
            "result": result, "is_error": "error" in result}


def chart(path, title, chart_type, result_id):
    return {"type": "chart", "path": path, "title": title,
            "chart_type": chart_type, "result_id": result_id}


def summary(text):
    return {"type": "summary", "text": text}


def error(message):
    return {"type": "error", "message": message}


def run_end(status, stats=None):
    return {"type": "run_end", "status": status, "stats": stats or {}}


# --- Checking events -------------------------------------------------------------
def validate(event):
    """Raise ValueError with a clear message if `event` doesn't follow the format."""
    if not isinstance(event, dict) or "type" not in event:
        raise ValueError(f"An event must be a dict with a 'type' key, got: {event!r}")

    event_type = event["type"]
    if event_type not in REQUIRED_FIELDS:
        raise ValueError(f"Unknown event type '{event_type}'. "
                         f"Valid types: {', '.join(REQUIRED_FIELDS)}.")

    missing = [f for f in REQUIRED_FIELDS[event_type] if f not in event]
    if missing:
        raise ValueError(f"'{event_type}' event is missing: {', '.join(missing)}.")

    if event_type == "issue_tree":
        for branch in event["branches"]:
            _validate_branch(branch)
    if event_type == "run_end" and event["status"] not in RUN_STATUSES:
        raise ValueError(f"run_end status must be one of {RUN_STATUSES}, got '{event['status']}'.")

    # Events must be saveable as JSON (no pandas or numpy objects)
    try:
        json.dumps(event)
    except TypeError as err:
        raise ValueError(f"'{event_type}' event is not JSON-serializable: {err}") from err


def _validate_branch(branch):
    for field in ["name", "status"]:
        if field not in branch:
            raise ValueError(f"Issue tree branch is missing '{field}': {branch!r}")
    if branch["status"] not in BRANCH_STATUSES:
        raise ValueError(f"Branch status must be one of {BRANCH_STATUSES}, "
                         f"got '{branch['status']}'.")
    for sub in branch.get("sub_branches", []):
        _validate_branch(sub)


def validate_run(events):
    """
    Check a whole run: every event is valid, it starts with run_start and ends
    with run_end, and every tool_result answers an earlier tool_call.
    Brains' tests use this to prove they follow the format.
    """
    if not events:
        raise ValueError("A run must contain at least run_start and run_end.")
    for event in events:
        validate(event)
    if events[0]["type"] != "run_start":
        raise ValueError("A run must start with a run_start event.")
    if events[-1]["type"] != "run_end":
        raise ValueError("A run must end with a run_end event.")

    open_calls = set()
    for event in events:
        if event["type"] == "tool_call":
            open_calls.add(event["call_id"])
        elif event["type"] == "tool_result":
            if event["call_id"] not in open_calls:
                raise ValueError(f"tool_result '{event['call_id']}' has no matching tool_call.")
            open_calls.remove(event["call_id"])
    if open_calls:
        raise ValueError(f"tool_call(s) never got a result: {', '.join(sorted(open_calls))}.")
