"""
Supply Chain Diagnostic Agent.

Claude plans the analysis and explains the results. Every number comes from
the Python functions in src/tools.py, which Claude calls through tool use.

Run from the project folder (either form works):
    python src/agent.py "Why are so many of our deliveries late?"
    python -m src.agent "Why are so many of our deliveries late?"
    python -m src.agent --model claude-haiku-4-5 "..."    # cheaper, for testing
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# When run as "python src/agent.py", Python only looks inside src/ for imports.
# Adding the project folder lets "from src import tools" work either way.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import number_check, tools  # noqa: E402  (must come after the path fix above)

# --- Settings ----------------------------------------------------------------
DEFAULT_MODEL = "claude-sonnet-5"
MAX_ITERATIONS = 15            # max calls to Claude in one run
MAX_TOKENS = 16000             # max length of one Claude response
RUNS_DIR = tools.OUTPUT_DIR / "runs"   # where each run's reasoning trail is saved

DEFAULT_QUESTION = (
    "More than half of our deliveries arrive late. "
    "What is driving this, and what should we do about it?"
)

# --- System prompt: the consulting workflow ----------------------------------
SYSTEM_PROMPT = """\
You are a supply chain diagnostic consultant. You answer business questions about the \
DataCo supply chain dataset the way a strategy consultant would, using the analysis tools \
provided. One row in the data is one order line (a product within an order), not a whole order.

Follow this workflow:

1. Restate the question. In one or two sentences: what decision this informs, and which \
metric measures it (late_rate, avg_delay, sales or profit).

2. Build an issue tree. Before calling any tool, break the problem into MECE drivers \
(mutually exclusive, collectively exhaustive), each tied to a column a tool can test. \
For example: logistics (Shipping Mode), geography (Market, Order Region), customer \
(Customer Segment), product (Category Name, Department Name), and time (trend_over_time). \
Give each branch a hypothesis. Record the tree with record_issue_tree (every branch "untested"), \
in the same turn as your first analysis tool calls.

3. Test each branch with tools. Branches are independent, so test them in parallel. \
A branch matters when its groups differ clearly from the baseline and the groups are large; \
drop branches where every group sits close to the baseline, and cite the numbers that justify \
dropping them. Treat groups with small_sample: true and periods with low_volume: true as \
unreliable. For a branch that matters, use drill_down inside it to check whether the effect \
holds across the other dimensions. Once every branch is tested, call record_issue_tree again \
with each branch marked "kept" or "dropped" and one line of evidence citing tool numbers \
(sub-branches can hold what you found inside a kept branch).

4. Quantify the impact of the real drivers with impact_estimate. These figures are sales and \
profit exposed to late delivery, not measured losses. Say so.

5. Recommend 2-3 actions, prioritised by impact versus effort. For each: the action, the \
finding that supports it (with numbers), impact (High/Medium/Low), effort (High/Medium/Low), \
and a concrete first step. Rate impact and effort qualitatively; do not invent targets or savings.

6. Write an executive summary using the pyramid principle: the answer in one sentence first, \
then three supporting points, each backed by numbers, then next steps. Write it last, but put it \
at the top of your final answer.

Rules for numbers (critical):
- Every number you write must come from a tool result. Never calculate, estimate or invent \
numbers yourself: no ratios, differences, sums, averages or projections. If a number you need \
isn't available from a tool, say what analysis would produce it.
- Reformatting is allowed: rates in tool results are decimals from 0 to 1, so 0.9532 can be \
written as 95.3%, a gap_vs_baseline of 0.4049 as +40.5 percentage points, and 5408069.35 as $5.41M.

Charts: when the key findings are clear, call make_chart for the one or two results that best \
support the answer, referring to them by result_id.

While working, keep your commentary between tool calls brief. Format the final answer in \
markdown with these sections: Executive summary, 1. Question, 2. Issue tree, \
3. Findings (branches kept and dropped), 4. Impact, 5. Recommendations (a table), \
Caveats, Charts (file paths).
"""

# Added to the last allowed call when the iteration limit is reached
FINAL_CALL_NOTE = (
    "You have reached the tool-call limit. Do not call more tools. Write your final answer now "
    "using only the results you already have, and list any branches you could not test under Caveats."
)

# --- Tools shown to Claude (shared with the other LLM brains) ------------------
from src.tool_specs import ANALYSIS_TOOLS, TOOL_DEFINITIONS, execute_tool  # noqa: E402,F401


# --- The tool-use loop ----------------------------------------------------------
def build_request(model, messages, last_call):
    """Parameters for one call to Claude."""
    request = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "tools": TOOL_DEFINITIONS,
        "messages": messages,
        # Cache the repeated prefix (tools, system prompt, earlier messages) to cut cost
        "cache_control": {"type": "ephemeral"},
    }
    # Adaptive thinking lets Claude reason before acting. Haiku 4.5 doesn't support it.
    if not model.startswith("claude-haiku"):
        request["thinking"] = {"type": "adaptive"}
    # On the last allowed call, switch tools off so Claude must write its answer
    if last_call:
        request["tool_choice"] = {"type": "none"}
    return request


def run_agent(question, model=DEFAULT_MODEL, max_iterations=MAX_ITERATIONS,
              client=None, verbose=True, runs_dir=None, on_step=None):
    """
    Answer a business question. Returns a dict with the answer, the full
    reasoning trail (every tool call and result), and where the run was saved.
    on_step, if given, is called with each new trail entry as it happens
    (the Streamlit app uses it to show the trail live).
    """
    client = client or anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]
    trail = []            # every step: Claude's commentary, tool calls and results
    saved_results = {}    # tool results by result_id, for make_chart
    usage = {"input_tokens": 0, "output_tokens": 0}
    answer = ""
    stop_reason = None

    for iteration in range(1, max_iterations + 1):
        # 1. Send the whole conversation so far to Claude
        response = client.messages.create(
            **build_request(model, messages, last_call=(iteration == max_iterations))
        )
        stop_reason = response.stop_reason
        usage["input_tokens"] += getattr(response.usage, "input_tokens", 0) or 0
        usage["output_tokens"] += getattr(response.usage, "output_tokens", 0) or 0

        text = "\n".join(b.text for b in response.content if b.type == "text").strip()

        # 2. No tool calls requested -> this is the final answer
        if stop_reason != "tool_use":
            answer = text
            break

        # Log Claude's commentary before its tool calls (e.g. the issue tree)
        if text:
            trail.append({"iteration": iteration, "type": "assistant_text", "text": text})
            if verbose:
                print(f"\n[{iteration}] Claude: {text}")
            if on_step:
                on_step(trail[-1])

        # 3. Keep Claude's full turn (including thinking and tool_use blocks) in the history
        messages.append({"role": "assistant", "content": response.content})

        # 4. Run every tool Claude asked for (there can be several in parallel)
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = execute_tool(block.name, block.input, saved_results)
            is_error = "error" in result
            trail.append({
                "iteration": iteration,
                "type": "tool_call",
                "tool": block.name,
                "input": block.input,
                "result": result,
                "is_error": is_error,
            })
            if verbose:
                print(f"[{iteration}] -> {block.name}({json.dumps(block.input)})")
                print(f"      <- {result.get('error') or result.get('description', '')}")
            if on_step:
                on_step(trail[-1])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,          # links this result to Claude's request
                "content": json.dumps(result),
                "is_error": is_error,
            })

        # 5. If the next call is the last one allowed, tell Claude to wrap up
        if iteration + 1 == max_iterations:
            tool_results.append({"type": "text", "text": FINAL_CALL_NOTE})

        # 6. Send all results back in ONE user message, then loop
        messages.append({"role": "user", "content": tool_results})

    if not answer:
        answer = f"(No final answer: the run stopped with stop_reason '{stop_reason}'.)"
    elif stop_reason == "max_tokens":
        answer += "\n\n(Warning: the answer was cut off because it hit the length limit.)"

    # Check the answer AND the issue tree's evidence lines, since both are shown to the user
    tree = latest_issue_tree(trail)
    unsupported = find_unsupported_numbers(answer + "\n" + tree_evidence_text(tree), trail, question)
    run = {
        "question": question,
        "model": model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "iterations": iteration,
        "stop_reason": stop_reason,
        "usage": usage,
        "unsupported_numbers": unsupported,
        "trail": trail,
        "answer": answer,
    }
    run["log_path"] = save_run(run, runs_dir or RUNS_DIR)
    return run


# --- Reading results back out of the trail ---------------------------------------
def latest_issue_tree(trail):
    """The most recent issue tree Claude recorded, or None."""
    for step in reversed(trail):
        if step["type"] == "tool_call" and step["tool"] == "record_issue_tree" and not step["is_error"]:
            return step["input"]
    return None


def tree_evidence_text(tree):
    """All evidence lines in an issue tree, joined into one string."""
    if not tree:
        return ""
    lines = []
    for branch in tree.get("branches", []):
        lines.append(branch.get("evidence", ""))
        lines += [sub.get("evidence", "") for sub in branch.get("sub_branches", [])]
    return "\n".join(lines)


def charts_in_trail(trail):
    """
    (chart title, chart type, source result) for every chart Claude made,
    so an app can redraw them. Source results are looked up by result_id.
    """
    results = {step["result"]["result_id"]: step["result"]
               for step in trail
               if step["type"] == "tool_call" and "result_id" in step["result"]}
    charts = []
    for step in trail:
        if step["type"] == "tool_call" and step["tool"] == "make_chart" and not step["is_error"]:
            source = results.get(step["input"]["result_id"])
            if source:
                charts.append((step["input"]["title"], step["input"]["chart_type"], source))
    return charts


# --- Saving the reasoning trail -------------------------------------------------
def save_run(run, runs_dir):
    """Save the full run as JSON (for review) and the answer as markdown. Returns the JSON path."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = runs_dir / f"{stamp}_trail.json"
    json_path.write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    (runs_dir / f"{stamp}_answer.md").write_text(run["answer"], encoding="utf-8")
    return str(json_path)


# --- Number audit: enforce "only cite numbers returned by tools" ----------------
def find_unsupported_numbers(answer, trail, question=""):
    """Numbers in the answer that don't match any tool result (see src/number_check.py)."""
    results = [step["result"] for step in trail if step["type"] == "tool_call"]
    return number_check.unsupported_numbers(answer, results, question)


# --- Command-line entry point ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ask the supply chain diagnostic agent a question.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION,
                        help="Business question (a default question is used if omitted).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Claude model, e.g. claude-haiku-4-5 for cheaper test runs.")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args()

    # Windows consoles may not print some characters in Claude's answer otherwise
    sys.stdout.reconfigure(encoding="utf-8")
    # Read ANTHROPIC_API_KEY from the .env file in the project folder
    load_dotenv(tools.PROJECT_ROOT / ".env")

    print(f"Question: {args.question}\nModel: {args.model}")
    try:
        run = run_agent(args.question, model=args.model, max_iterations=args.max_iterations)
    except anthropic.AuthenticationError:
        sys.exit("Authentication failed: check ANTHROPIC_API_KEY in your .env file.")
    except anthropic.APIConnectionError:
        sys.exit("Could not reach the Anthropic API: check your internet connection.")
    except anthropic.APIStatusError as error:
        sys.exit(f"API error {error.status_code}: {error.message}")

    print("\n" + "=" * 70 + "\n" + run["answer"] + "\n" + "=" * 70)
    if run["unsupported_numbers"]:
        print("WARNING - numbers not found in any tool result:", ", ".join(run["unsupported_numbers"]))
    else:
        print("Number check: every number in the answer traces back to a tool result.")
    print(f"Iterations: {run['iterations']} | tokens in/out: "
          f"{run['usage']['input_tokens']:,} / {run['usage']['output_tokens']:,}")
    print(f"Reasoning trail saved to: {run['log_path']}")


if __name__ == "__main__":
    main()
