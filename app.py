"""
Streamlit front end for the Supply Chain Diagnostic Agent.

Run from the project folder:
    streamlit run app.py

The app never talks to an AI model directly. It asks the selected "brain"
(src/brains/) for a stream of events (src/events.py) and draws them, so it
works the same with the scripted brain, a local LLM, or Claude, and it needs
no .env file or API key.

Each run is saved to demo_runs/ as JSON and can be replayed from the sidebar.
"""

import html
import re
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from src import brains, events, number_check, run_store, tools
from src.brains import ollama_brain, scripted

load_dotenv(tools.PROJECT_ROOT / ".env")   # optional: only matters for API-based brains

EXAMPLE_QUESTIONS = scripted.EXAMPLE_QUESTIONS
SUBTITLE = "Diagnoses supply chain problems like a consultant, and every number comes from code, not the AI."

# Dropdown label -> the brain id that brain puts in its run_start event
BRAIN_IDS = {"Scripted (no LLM)": "scripted",
             "Local LLM (Ollama)": f"ollama:{ollama_brain.MODEL}",
             "Claude API": "claude"}
REPLAY_SPEEDS = {"1x": 1, "2x": 2, "4x": 4}
CHART_FONT_SIZE = 14    # readable in a 1080p screen recording
CHART_HEIGHT = 380
PANEL_HEIGHT = 480      # trail panel: trail and tree fit on one screen at 125% zoom

TOOL_LABELS = {
    "segment_analysis": "Segment analysis",
    "trend_over_time": "Trend",
    "impact_estimate": "Impact estimate",
    "drill_down": "Drill-down",
    "make_chart": "Chart",
}
METRIC_LABELS = {"late_rate": "late rate", "loss_rate": "loss-making share",
                 "avg_delay": "avg delay", "sales": "sales", "profit": "profit"}

# --- Styling -----------------------------------------------------------------
PAGE_CSS = """
<style>
/* Hide Streamlit's own chrome for a clean recording (the sidebar toggle stays) */
#MainMenu, footer,
[data-testid="stToolbar"],       /* Deploy button and menu */
[data-testid="stDecoration"],    /* coloured strip at the top */
[data-testid="stStatusWidget"]   /* "Running..." animation, top right */
{display: none !important;}

.block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem;}
h1 {font-weight: 650; letter-spacing: -0.01em;}
.subtitle {font-size: 1.1rem; color: #3d3c39; margin: -0.6rem 0 0.6rem 0;}
.brain-badge {display: inline-block; font-size: 0.82rem; font-weight: 600; padding: 0.18rem 0.7rem;
              border-radius: 999px; background: #eaf2fd; color: #1c5cab; border: 1px solid #cfe0f7;}
.run-meta {font-size: 0.85rem; color: #52514e; margin-left: 0.5rem;}

/* The highlighted executive summary box (st.container with key="exec_summary") */
.st-key-exec_summary {
    background: #f2f7fe; border: 1px solid #d6e5f8; border-left: 5px solid #2a78d6;
    border-radius: 8px; padding: 1.3rem 1.6rem 0.6rem 1.6rem;
}
.st-key-exec_summary p, .st-key-exec_summary li {font-size: 1.05rem; line-height: 1.55;}
.eyebrow {font-size: 0.78rem; font-weight: 650; letter-spacing: 0.08em;
          text-transform: uppercase; color: #2a78d6; margin-bottom: 0.2rem;}
/* Darker captions: light grey text blurs after video compression */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
    color: #52514e !important; opacity: 1 !important;}
</style>
"""

TREE_CSS = """
<style>
.tree {font-size: 1rem; line-height: 1.45; color: #1b1b1a;}
.tree .root {font-weight: 650; padding: 0.55rem 0.8rem; background: #f5f6f8;
             border-radius: 6px; margin-bottom: 0.4rem;}
.tree ul {list-style: none; margin: 0; padding-left: 1.1rem; border-left: 2px solid #cfd3d9;}
.tree li {margin: 0.55rem 0; position: relative;}
.tree li::before {content: ""; position: absolute; left: -1.1rem; top: 0.75rem;
                  width: 0.8rem; border-top: 2px solid #cfd3d9;}
.tree .name {font-weight: 600;}
.tree .name.dropped {color: #6b6a66;}
.tree .badge {display: inline-block; font-size: 0.7rem; font-weight: 650; padding: 0.05rem 0.5rem;
              border-radius: 999px; margin-left: 0.45rem; vertical-align: 1px;}
.tree .badge.kept {background: #e3f4e8; color: #0b6b2c;}
.tree .badge.dropped {background: #efeeeb; color: #5f5e5a;}
.tree .badge.untested {background: #fff3d6; color: #7a5200;}
.tree .hyp {color: #52514e; font-size: 0.92rem;}
.tree .ev {font-size: 0.92rem; margin-top: 0.1rem;}
.tree .empty {color: #52514e; font-size: 0.95rem;}
</style>
"""


# --- Small helpers -----------------------------------------------------------------
def md_safe(text):
    """Streamlit reads $...$ as maths; escape dollar signs so '$5.4M' shows as text."""
    return str(text).replace("$", "\\$")


def eyebrow(text):
    st.markdown(f"<div class='eyebrow'>{html.escape(text)}</div>", unsafe_allow_html=True)


def brain_name(brain_id):
    """Readable name for a brain id, e.g. 'ollama:qwen2.5:7b' -> 'Local LLM: qwen2.5:7b'."""
    if brain_id.startswith("ollama:"):
        return "Local LLM: " + brain_id.split(":", 1)[1]
    if brain_id == "scripted":
        return "Scripted rules · no LLM"
    if brain_id.startswith("claude"):
        return "Claude API"
    return brain_id


def brain_badge(brain_id):
    """Small pill naming the brain."""
    return f"<span class='brain-badge'>{html.escape(brain_name(brain_id))}</span>"


def brain_id_of(run):
    """The brain id from a run's run_start event (older saved runs may lack one)."""
    first = run["events"][0] if run["events"] else {}
    return first.get("brain") or BRAIN_IDS.get(run["brain"], run["brain"])


def describe_call(event):
    """One-line summary of a tool_call's inputs, e.g. 'late rate by Market'."""
    args, tool = event["inputs"], event["tool"]
    metric = METRIC_LABELS.get(args.get("metric"), args.get("metric"))
    if tool == "segment_analysis":
        return f"{metric} by {args.get('group_by')}"
    if tool == "drill_down":
        filters = ", ".join(f"{k} = {v}" for k, v in (args.get("filters") or {}).items())
        return f"{metric} by {args.get('group_by')}, within {filters}"
    if tool == "trend_over_time":
        return f"{metric} per {args.get('freq')}"
    if tool == "impact_estimate":
        return f"{args.get('filter_column')} = {args.get('filter_value')}"
    if tool == "make_chart":
        return f"“{args.get('title')}”"
    return ""


def render_trail_event(event, call_number):
    """Draw one event in the reasoning trail (inside the current container)."""
    kind = event["type"]
    if kind == "note":
        st.caption(md_safe(event["text"]))
    elif kind == "tool_call":
        label = TOOL_LABELS.get(event["tool"], event["tool"])
        st.markdown(f"**{call_number}. {label}** &nbsp;·&nbsp; {md_safe(describe_call(event))}")
    elif kind == "tool_result":
        if event["is_error"]:
            st.error(md_safe(event["result"]["error"]), icon=":material/error:")
        else:
            st.caption("↳ " + md_safe(event["result"].get("description", "done")))
    elif kind == "error":
        st.error(md_safe(event["message"]), icon=":material/error:")


def tree_html(tree, final=False):
    """The issue tree as an indented diagram (HTML). Every text value is escaped."""
    if not tree:
        return TREE_CSS + "<div class='tree'><p class='empty'>The issue tree appears here " \
                          "as soon as the brain builds it.</p></div>"
    labels = {"kept": "Driver", "dropped": "Ruled out",
              "untested": "Not tested" if final else "Testing"}

    def node(branch):
        status = branch.get("status", "untested")
        parts = [f"<span class='name {status}'>{html.escape(branch.get('name', ''))}</span>",
                 f"<span class='badge {status}'>{labels.get(status, status)}</span>"]
        if branch.get("hypothesis"):
            parts.append(f"<div class='hyp'>Hypothesis: {html.escape(branch['hypothesis'])}</div>")
        if branch.get("evidence"):
            parts.append(f"<div class='ev'>{html.escape(branch['evidence'])}</div>")
        subs = branch.get("sub_branches") or []
        if subs:
            parts.append("<ul>" + "".join(node(s) for s in subs) + "</ul>")
        return "<li>" + "".join(parts) + "</li>"

    branches = "".join(node(b) for b in tree.get("branches", []))
    return (TREE_CSS + f"<div class='tree'><div class='root'>{html.escape(tree.get('root', ''))}"
            f"</div><ul>{branches}</ul></div>")


def split_executive_summary(answer):
    """
    Pull the 'Executive summary' section out of a markdown summary.
    Returns the section's text, or the whole summary if there's no such heading.
    """
    lines = answer.splitlines()
    for start, line in enumerate(lines):
        heading = line.strip().lstrip("#").strip().strip("*").strip().lower()
        if heading.startswith("executive summary"):
            level = len(line) - len(line.lstrip("#"))   # 0 if the heading is bold text
            body = []
            for later in lines[start + 1:]:
                later_level = len(later) - len(later.lstrip("#"))
                ends_section = (later_level and (level == 0 or later_level <= level)) or \
                               (level == 0 and re.match(r"^\*\*\d", later.strip()))
                if ends_section:
                    break
                body.append(later)
            return "\n".join(body).strip()
    return answer


# --- Reading a finished run (a list of events) --------------------------------------
def latest_tree(evts):
    trees = [e for e in evts if e["type"] == "issue_tree"]
    return trees[-1] if trees else None


def summary_of(evts):
    return next((e["text"] for e in evts if e["type"] == "summary"), None)


def results_by_id(evts):
    return {e["result"]["result_id"]: e["result"] for e in evts
            if e["type"] == "tool_result" and "result_id" in e["result"]}


def check_numbers(evts, question):
    """Numbers in the summary and tree evidence that no tool returned (the app checks
    every brain itself, rather than trusting the brain's own report)."""
    shown = summary_of(evts) or ""
    tree = latest_tree(evts)
    if tree:
        for branch in tree["branches"]:
            shown += "\n" + branch.get("evidence", "")
            shown += "".join("\n" + s.get("evidence", "") for s in branch.get("sub_branches", []))
    all_results = [e["result"] for e in evts if e["type"] == "tool_result"]
    return number_check.unsupported_numbers(shown, all_results, question,
                                            allowed=[scripted.MIN_ROWS_FOR_DRIVER])


def report_markdown(run):
    """The markdown file behind the 'Download summary' button."""
    flagged = run["flagged"]
    check = ("**Number check:** every number above traces back to a tool result." if not flagged else
             "**Number check:** these numbers were not found in any tool result - verify before use: "
             + ", ".join(flagged))
    return (f"# {run['question']}\n\n"
            f"_Supply Chain Diagnostic Agent - {run['timestamp']} - {run['brain']}_\n\n"
            f"{summary_of(run['events'])}\n\n---\n\n{check}\n")


# --- Streaming a run ------------------------------------------------------------------------
def stream_run(event_stream, question, brain_label):
    """Draw events as they arrive; return the finished run as a dict."""
    trail_col, tree_col = st.columns([3, 2], gap="large")
    with tree_col:
        eyebrow("Issue tree")
        tree_slot = st.empty()
        tree_slot.html(tree_html(None))
    name = brain_name(BRAIN_IDS.get(brain_label, brain_label))
    with trail_col:
        eyebrow("Reasoning trail")
        status = st.status(f"{name} · starting...", expanded=True)
        with status:
            steps_box = st.container(height=PANEL_HEIGHT, border=False, autoscroll=True)

    collected, calls = [], 0
    started = time.time()
    try:
        for event in event_stream:
            events.validate(event)   # a brain that breaks the format fails loudly here
            collected.append(event)
            if event["type"] == "run_start":
                name = brain_name(event["brain"])   # the brain names itself
                status.update(label=f"{name} · starting...")
            elif event["type"] == "tool_call":
                calls += 1
                status.update(label=f"{name} · analysing... {calls} tool calls so far")
            elif event["type"] == "issue_tree":
                tree_slot.html(tree_html(event))
            with steps_box:
                render_trail_event(event, calls)
    except Exception as err:   # the brain itself crashed: record it like any other failure
        collected += [events.error(f"{type(err).__name__}: {err}"), events.run_end("failed")]
        with steps_box:
            render_trail_event(collected[-2], calls)

    failed = any(e["type"] == "error" for e in collected)
    status.update(label=f"{name} · run failed" if failed else f"{name} · analysis complete",
                  state="error" if failed else "complete")
    return {
        "question": question,
        "brain": brain_label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "seconds": round(time.time() - started),
        "events": collected,
    }


# --- The finished-run view ----------------------------------------------------------------
def show_results(run):
    evts = run["events"]
    tree = latest_tree(evts)
    summary = summary_of(evts)
    tool_calls = [e for e in evts if e["type"] == "tool_call"]
    errors = [e for e in evts if e["type"] == "error"]

    st.markdown(f"### {md_safe(run['question'])}")
    source = "Replay of a saved run" if run.get("replayed") else "Live run"
    st.markdown(f"{brain_badge(brain_id_of(run))}<span class='run-meta'>{source} · "
                f"{html.escape(run['timestamp'])}</span>", unsafe_allow_html=True)

    if errors and not summary:
        st.error("The run could not finish: " + md_safe(errors[0]["message"]), icon=":material/error:")
        return

    # 1. Executive summary, highlighted, with the download button
    with st.container(key="exec_summary"):
        eyebrow("Executive summary")
        st.markdown(md_safe(split_executive_summary(summary)))
    st.download_button("Download summary", data=report_markdown(run),
                       file_name=f"executive_summary_{run['timestamp'][:10]}.md",
                       mime="text/markdown", icon=":material/download:", type="primary")

    # 2. Key figures about the analysis itself
    branches = (tree or {}).get("branches", [])
    kept = [b for b in branches if b.get("status") == "kept"]
    flagged = run["flagged"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drivers found", f"{len(kept)} of {len(branches)}" if branches else "-")
    c2.metric("Tool calls", len(tool_calls))
    c3.metric("Number check", "Passed" if not flagged else f"{len(flagged)} flagged")
    c4.metric("Analysis time", f"{run['seconds']} s")
    if flagged:
        st.warning("These numbers were not found in any tool result - verify them before "
                   "sharing: " + ", ".join(flagged), icon=":material/warning:")

    st.divider()

    # 3. Issue tree and reasoning trail side by side
    tree_col, trail_col = st.columns([2, 3], gap="large")
    with tree_col:
        eyebrow("Issue tree")
        st.html(tree_html(tree, final=True))
    with trail_col:
        eyebrow("Reasoning trail")
        with st.container(height=PANEL_HEIGHT, border=True):
            calls = 0
            for event in evts:
                if event["type"] == "tool_call":
                    calls += 1
                render_trail_event(event, calls)

    # 4. Charts, redrawn as interactive charts from the saved tool results
    results = results_by_id(evts)
    charts = [e for e in evts if e["type"] == "chart" and e["result_id"] in results]
    if charts:
        st.divider()
        eyebrow("Charts")
        for row_start in range(0, len(charts), 2):
            cols = st.columns(2, gap="large")
            for col, chart in zip(cols, charts[row_start:row_start + 2]):
                with col:
                    try:
                        fig = tools.build_chart(results[chart["result_id"]], chart["chart_type"], chart["title"])
                        # Presentation only: larger text for the screen, same data
                        fig.update_layout(font_size=CHART_FONT_SIZE, height=CHART_HEIGHT)
                        st.plotly_chart(fig, width="stretch")
                    except tools.ToolInputError as err:
                        st.caption(f"Could not draw '{chart['title']}': {err}")

    # 5. Full report
    st.divider()
    with st.expander("Full report"):
        st.markdown(md_safe(summary))


# --- Page ------------------------------------------------------------------------------------
def choose_example(question):
    """Button callback: put the example in the text box and start the analysis."""
    st.session_state.question = question
    st.session_state.action = "run"


def request_run():
    st.session_state.action = "run"


def request_replay():
    st.session_state.action = "replay"


def finish(run, replayed=False):
    """Store a finished run and redraw the page in the results layout."""
    run["replayed"] = replayed
    run["flagged"] = check_numbers(run["events"], run["question"])
    st.session_state.run = run
    st.rerun()


def replay_controls(runs, primary=False):
    """Saved-run picker, speed choice and Replay button (sidebar, or main area in replay mode)."""
    st.selectbox("Saved run", runs, format_func=run_store.describe, key="replay_file")
    st.radio("Replay speed", list(REPLAY_SPEEDS), horizontal=True, key="replay_speed")
    st.button("Replay", on_click=request_replay, icon=":material/replay:", width="stretch",
              type="primary" if primary else "secondary")


def how_it_works():
    st.markdown("### How it works")
    st.markdown(
        "1. The brain restates the question and builds a MECE issue tree.\n"
        "2. It tests each branch with Python analysis tools and rules out the ones "
        "that don't matter.\n"
        "3. It sizes the real drivers and recommends actions by impact and effort.\n\n"
        "**Every number comes from a tool call.** The app checks each summary "
        "against the tool results."
    )
    st.caption("Data: DataCo Smart Supply Chain dataset (one row = one order line).")


def main():
    st.set_page_config(page_title="Supply Chain Diagnostic Agent", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    brain_labels = brains.available_brains()
    runs = run_store.saved_runs()
    # Replay mode: no brain can run (the dataset isn't installed), so only saved runs are shown
    replay_mode = not brain_labels
    brain_label = None

    with st.sidebar:
        if replay_mode:
            st.markdown("### Replay mode")
            st.caption("Live analysis needs the DataCo dataset in data/. "
                       "See the README to run it yourself.")
        else:
            st.markdown("### Brain")
            brain_label = st.selectbox("Which planner runs the analysis", brain_labels)
            st.caption("Other brains appear here when they're available: a local LLM when "
                       "Ollama is running, Claude when an API key is set.")
            st.markdown("### Replay a saved run")
            if runs:
                replay_controls(runs)
            else:
                st.caption("Runs are saved to demo_runs/ automatically; they will appear here.")
        how_it_works()

    st.title("Supply Chain Diagnostic Agent")
    st.markdown(f"<p class='subtitle'>{SUBTITLE}</p>", unsafe_allow_html=True)

    if replay_mode:
        if runs:
            st.info("**Replay mode.** The dataset isn't installed here, so live analysis is off. "
                    "Pick a saved run to watch the agent work through it step by step, exactly "
                    "as it was recorded.", icon=":material/replay:")
            replay_controls(runs, primary=True)
        else:
            st.warning("No dataset and no saved runs found. Add data/DataCoSupplyChainDataset.csv "
                       "(see the README) to run live analyses.", icon=":material/warning:")
    else:
        st.text_area("Business question", key="question", height=80,
                     placeholder="e.g. Why are our deliveries late, and what should we do about it?")
        st.caption("Or try an example:")
        for col, example in zip(st.columns(3), EXAMPLE_QUESTIONS):
            # wrap=True shows the whole question instead of cutting it off with "..."
            col.button(example, on_click=choose_example, args=(example,), width="stretch", wrap=True)
        st.button("Run analysis", type="primary", on_click=request_run, icon=":material/play_arrow:")
    st.write("")

    action = st.session_state.pop("action", None)
    if action == "run" and not replay_mode:
        question = st.session_state.get("question", "").strip()
        if not question:
            st.warning("Type a question or pick an example first.")
            return
        st.session_state.pop("run", None)   # clear the previous result
        run = stream_run(brains.run_brain(brain_label, question), question, brain_label)
        if summary_of(run["events"]):   # keep only runs worth replaying
            run_store.save_run(run)
        finish(run)
    elif action == "replay" and st.session_state.get("replay_file"):
        saved = run_store.load_run(st.session_state.replay_file)
        st.session_state.pop("run", None)
        speed = REPLAY_SPEEDS[st.session_state.get("replay_speed", "1x")]
        stream_run(run_store.replay(saved["events"], speed), saved["question"], saved["brain"])
        finish(saved, replayed=True)
    elif "run" in st.session_state:
        show_results(st.session_state.run)


# Streamlit runs this file as __main__; tests can import it without drawing the page
if __name__ == "__main__":
    main()
