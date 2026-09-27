"""
Local LLM brain: a free model (qwen2.5:7b) running on this computer through
Ollama, driving the same tools as every other brain.

How it works - the tool-use loop:
1. Send the model the system prompt (the consulting workflow), the question,
   and descriptions of the tools.
2. The model replies either with tool calls ("run segment_analysis with ...")
   or with its final answer.
3. For each tool call, Python runs the real function in src/tools.py and sends
   the result back. Every call and result is yielded as an event.
4. Repeat, at most MAX_ITERATIONS times. On the last one, tools are switched off
   so the model must write its summary.

Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1, so this
file sends standard "chat completions" requests with the `requests` library.

Setup (once): install Ollama from ollama.com, then run: ollama pull qwen2.5:7b
"""

import json
import time

import requests

from src import events, tool_specs

BRAIN_NAME = "ollama:qwen2.5:7b"
MODEL = "qwen2.5:7b"
BASE_URL = "http://localhost:11434/v1"
MAX_ITERATIONS = 15
REQUEST_TIMEOUT = 600   # seconds per model reply; generous because CPU-only laptops are slow
TEMPERATURE = 0.2       # low = more consistent tool use

SYSTEM_PROMPT = """\
You are a supply chain consultant. Answer the user's business question about the DataCo \
supply chain dataset using ONLY the tools provided. One row in the data is one order line \
(a product within an order), not a whole order.

Work through these steps in order:
1. Restate the question in one sentence and decide which metric measures it: late_rate for \
delivery questions, loss_rate (and profit) for profitability questions.
2. Build a MECE issue tree (mutually exclusive, collectively exhaustive) of 4-5 possible \
drivers, for example Shipping Mode, Market, Customer Segment, Category Name and the time \
trend. Call record_issue_tree with every branch "untested".
3. Test each branch: call segment_analysis once per branch (trend_over_time with \
freq "quarter" for the time branch). A branch matters if a large segment has a clearly \
positive gap_vs_baseline; drop branches where every segment is close to the baseline. \
Ignore segments with small_sample: true and periods with low_volume: true.
4. For each branch that matters: call drill_down on its worst segment, then \
impact_estimate for that segment.
5. Call record_issue_tree again, marking each branch "kept" or "dropped" with one line \
of evidence quoting tool numbers.
6. Call make_chart for the main driver, using the result_id of its segment_analysis result.
7. Stop calling tools and write the final answer.

Final answer format (markdown):
## Executive summary
One sentence that answers the question, then three bullet points with numbers, then 2-3 \
numbered recommendations, each with "Impact: High/Medium/Low" and "Effort: High/Medium/Low", \
highest impact and lowest effort first.
## Question
## Issue tree and findings
## Impact
## Recommendations
## Caveats

Rules for numbers (critical):
- Only write numbers that appear in tool results. Never calculate, estimate or invent numbers.
- You may reformat them: 0.9532 as 95.3%, a gap_vs_baseline of 0.4049 as +40.5 percentage \
points, 5408069.35 as $5.41M.
- Impact figures are sales and profit exposed to late delivery, not measured losses.
"""

FINAL_CALL_NOTE = (
    "You have reached the tool-call limit. Do not call any more tools. Write the final answer "
    "now in the required format, using only numbers from the tool results above."
)

# The shared tool definitions, converted to the OpenAI "function" format Ollama expects
OPENAI_TOOLS = [
    {"type": "function",
     "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in tool_specs.TOOL_DEFINITIONS
]
TOOL_NAMES = {t["name"] for t in tool_specs.TOOL_DEFINITIONS}


# --- Talking to Ollama -----------------------------------------------------------
def is_available():
    """True if Ollama is running and the model has been downloaded."""
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=1)
        response.raise_for_status()
    except requests.RequestException:
        return False
    model_ids = {m["id"] for m in response.json().get("data", [])}
    return MODEL in model_ids or f"{MODEL}:latest" in model_ids


def chat(messages, use_tools):
    """One request to the model. Returns the reply message (a dict)."""
    body = {"model": MODEL, "messages": messages, "temperature": TEMPERATURE}
    if use_tools:
        body["tools"] = OPENAI_TOOLS
    response = requests.post(f"{BASE_URL}/chat/completions", json=body, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


# --- Reading the model's tool calls ----------------------------------------------
def parse_arguments(raw):
    """Tool arguments arrive as a JSON string (sometimes already a dict). None if unreadable."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def tool_calls_in(message):
    """
    The tool calls in a reply, as a list of (call_id, name, raw_arguments).
    Small local models sometimes write a call as plain JSON text instead of a
    proper tool call, e.g. {"name": "segment_analysis", "arguments": {...}};
    that case is recognised too.
    """
    calls = [(c.get("id"), c["function"]["name"], c["function"].get("arguments"))
             for c in message.get("tool_calls") or []]
    if calls:
        return calls
    text = (message.get("content") or "").strip()
    if text.startswith("{") and text.endswith("}"):
        parsed = parse_arguments(text)
        if parsed and parsed.get("name") in TOOL_NAMES:
            return [(None, parsed["name"], parsed.get("arguments", parsed.get("parameters", {})))]
    return []


# --- Running one tool call as events ---------------------------------------------
def run_tool(name, raw_arguments, call_id, saved_results):
    """
    Run one tool call: yields tool_call and tool_result events (plus issue_tree or
    chart events where relevant), and returns the result for the model.
    Use as: result = yield from run_tool(...)
    """
    arguments = parse_arguments(raw_arguments)
    yield events.tool_call(call_id, name, arguments if arguments is not None else {"raw": str(raw_arguments)})

    if arguments is None:
        result = {"error": "The tool arguments were not valid JSON. Send a JSON object."}
    else:
        result = tool_specs.execute_tool(name, arguments, saved_results)

    # An issue tree must follow the event format; if it doesn't, tell the model what's wrong
    tree_event = None
    if name == "record_issue_tree" and "error" not in result:
        tree_event = events.issue_tree(arguments.get("root", ""), arguments["branches"])
        try:
            events.validate(tree_event)
        except ValueError as err:
            result, tree_event = {"error": f"Invalid issue tree: {err}"}, None

    yield events.tool_result(call_id, name, result)
    if tree_event:
        yield tree_event
    if name == "make_chart" and "error" not in result:
        yield events.chart(result["path"], arguments.get("title", ""),
                           arguments.get("chart_type", "bar"), arguments.get("result_id", ""))
    return result


# --- The tool-use loop ---------------------------------------------------------------
def run(question, chat_fn=chat, max_iterations=MAX_ITERATIONS):
    """Answer `question` with the local model, yielding events as the work happens.
    chat_fn can be replaced by a fake in tests."""
    started = time.time()
    yield events.run_start(question, BRAIN_NAME)

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    saved_results = {}   # result_id -> result, for make_chart
    calls = 0
    summary = ""

    try:
        for iteration in range(1, max_iterations + 1):
            # 1. Ask the model. On the last allowed call, tools are switched off.
            last_call = iteration == max_iterations
            message = chat_fn(messages, use_tools=not last_call)
            tool_calls = [] if last_call else tool_calls_in(message)
            text = (message.get("content") or "").strip()

            # 2. No tool calls -> this is the final answer
            if not tool_calls:
                summary = text
                break

            # Keep the model's own words between tool calls as commentary
            if text and not text.startswith("{"):
                yield events.note(text)

            # 3. Keep the model's turn in the history, then run each tool it asked for
            messages.append({"role": "assistant", "content": message.get("content") or "",
                             "tool_calls": message.get("tool_calls") or []})
            for call_id, name, raw_arguments in tool_calls:
                calls += 1
                result = yield from run_tool(name, raw_arguments, call_id or f"c{calls}", saved_results)
                # 4. Send the result back, linked to the call it answers
                if call_id:
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
                else:   # the call was written as text, so there's no id to link to
                    messages.append({"role": "user", "content": f"Result of {name}: {json.dumps(result)}"})

            # 5. Before the last allowed call, tell the model to wrap up
            if iteration + 1 == max_iterations:
                messages.append({"role": "user", "content": FINAL_CALL_NOTE})

    except requests.RequestException as err:
        yield events.error(f"Could not get a reply from Ollama ({type(err).__name__}). "
                           f"Is Ollama running with {MODEL} downloaded?")
        yield events.run_end("failed", {"tool_calls": calls, "seconds": round(time.time() - started, 1)})
        return

    stats = {"tool_calls": calls, "iterations": iteration, "model": MODEL,
             "seconds": round(time.time() - started, 1)}
    if not summary:
        yield events.error("The model finished without writing a summary.")
        yield events.run_end("failed", stats)
        return
    yield events.summary(summary)
    yield events.run_end("complete", stats)
