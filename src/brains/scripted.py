"""
Scripted brain: answers a business question with a fixed consulting workflow.
No LLM and no API key. Every decision is a written rule below, and every
number comes from the analysis functions in src/tools.py.

The workflow:
1. Match the question to one of three supported question types.
2. Emit a predefined issue tree: shipping mode, market, customer segment,
   category and time trend.
3. Test each branch with segment_analysis (trend_over_time for time). A branch
   is a driver ("kept") if its worst reliable segment is more than 5 percentage
   points worse than the baseline; otherwise it's "dropped".
4. For each driver: drill_down into its worst segment, and impact_estimate it.
5. Chart the drivers, the question's focus and the trend.
6. Fill a summary template: the answer first, then the key drivers with
   numbers, then 2-3 recommendations picked from a rule list per driver type.

Usage:
    for event in run("Why are our deliveries late?"):
        print(event)
Every event follows the standard format in src/events.py.
"""

import time

from src import events, number_check, tools

BRAIN_NAME = "scripted"

# --- Rules: change these to tune the brain ------------------------------------
DRIVER_THRESHOLD = 0.05     # driver = worst segment is more than 5 pp worse than the baseline...
MIN_ROWS_FOR_DRIVER = 1000  # ...counting only segments with at least this many order lines
DEFAULT_DELAY = 0.25        # seconds to pause before each event, so the UI can stream them

# --- The predefined issue tree -------------------------------------------------
# column=None means the time branch, tested with trend_over_time instead
BRANCHES = [
    {"key": "Shipping Mode", "name": "Logistics: Shipping Mode", "column": "Shipping Mode"},
    {"key": "Market", "name": "Geography: Market", "column": "Market"},
    {"key": "Customer Segment", "name": "Customer: Customer Segment", "column": "Customer Segment"},
    {"key": "Category Name", "name": "Product: Category", "column": "Category Name"},
    {"key": "trend", "name": "Time: quarterly trend", "column": None},
]

LATE_HYPOTHESES = {
    "Shipping Mode": "Some shipping modes miss their promised lead time far more often.",
    "Market": "Deliveries are late in particular regions.",
    "Customer Segment": "Some customer types get worse service.",
    "Category Name": "Certain products are harder to deliver on time.",
    "trend": "Lateness has got worse recently.",
}

# --- The three supported question types ---------------------------------------
# metric: what the tree is tested on (higher = worse for every metric used here)
# kind: which recommendation rules apply ("service" or "profit")
# focus: the dimension the question is about; it is always discussed and charted
# context: extra analyses reported alongside the tree, as (metric, column)
QUESTION_TYPES = {
    "late_deliveries": {
        "example": "Why are our deliveries late, and what should we do about it?",
        "keywords": ["late", "deliver", "delay", "on time", "on-time", "lead time", "shipping"],
        "root": "What is driving late deliveries, and what should we do about it?",
        "metric": "late_rate", "noun": "late-delivery rate", "kind": "service",
        "focus": "Shipping Mode",
        "hypotheses": LATE_HYPOTHESES,
        "context": [("sales", "Shipping Mode")],
    },
    "profitability": {
        "example": "Which markets are most and least profitable, and why?",
        "keywords": ["profit", "margin", "loss", "losing", "money"],
        "root": "Where are we losing money, and is it a market problem?",
        "metric": "loss_rate", "noun": "share of loss-making order lines", "kind": "profit",
        "focus": "Market",
        "hypotheses": {
            "Shipping Mode": "Shipping costs make some modes unprofitable.",
            "Market": "Some markets sell at a loss more often (pricing, discounts, costs).",
            "Customer Segment": "Some customer types get unprofitable deals.",
            "Category Name": "Some product categories are sold below cost.",
            "trend": "Losses have grown recently.",
        },
        "context": [("profit", "Market"), ("sales", "Market")],
    },
    "segment_performance": {
        "example": "How do our customer segments compare on performance?",
        "keywords": ["segment", "customer", "consumer", "corporate", "home office"],
        "root": "How do our customer segments compare on service and profitability?",
        "metric": "late_rate", "noun": "late-delivery rate", "kind": "service",
        "focus": "Customer Segment",
        "hypotheses": {**LATE_HYPOTHESES,
                       "Customer Segment": "One customer segment gets clearly worse service."},
        "context": [("sales", "Customer Segment"), ("profit", "Customer Segment"),
                    ("loss_rate", "Customer Segment")],
    },
}
EXAMPLE_QUESTIONS = [spec["example"] for spec in QUESTION_TYPES.values()]

# --- Recommendation rules, per (kind, driver type) ------------------------------
# {group} is filled with the driver's worst segment. The (kind, None) entries are
# company-wide actions, used when there are fewer than three driver-specific ones.
RECOMMENDATIONS = {
    ("service", "Shipping Mode"): [
        {"action": "Reset the promised lead time for {group} to what the network actually delivers",
         "impact": "High", "effort": "Low",
         "first_step": "Compare promised and actual shipping days for {group}, then update the customer promise."},
        {"action": "Run an execution review of {group} with carriers and warehouses",
         "impact": "High", "effort": "High",
         "first_step": "Agree an on-time target for {group} with each carrier and track it weekly."},
    ],
    ("service", "Market"): [
        {"action": "Review carrier performance in {group}", "impact": "High", "effort": "Medium",
         "first_step": "Rank the carriers serving {group} by on-time rate."},
    ],
    ("service", "Customer Segment"): [
        {"action": "Set up a service recovery plan for {group} customers", "impact": "Medium",
         "effort": "Medium", "first_step": "Contact the largest {group} accounts hit by late orders."},
    ],
    ("service", "Category Name"): [
        {"action": "Review how {group} products are stocked and picked", "impact": "Medium",
         "effort": "Medium", "first_step": "Map and time each order-to-ship step for {group}."},
    ],
    ("service", "trend"): [
        {"action": "Investigate what changed in {group}", "impact": "Medium", "effort": "Low",
         "first_step": "List process, carrier and volume changes around {group}."},
    ],
    ("service", None): [
        {"action": "Audit promised versus actual lead times across every shipping mode",
         "impact": "High", "effort": "Medium",
         "first_step": "Pull promised and actual shipping days per order for one month."},
        {"action": "Publish a monthly on-time dashboard by shipping mode, market and category",
         "impact": "Medium", "effort": "Low", "first_step": "Automate this analysis as a monthly report."},
    ],
    ("profit", "Shipping Mode"): [
        {"action": "Reprice {group} shipping so it covers its cost", "impact": "Medium", "effort": "Low",
         "first_step": "Compare {group} shipping charges with carrier costs."},
    ],
    ("profit", "Market"): [
        {"action": "Review pricing and discount levels in {group}", "impact": "High", "effort": "Medium",
         "first_step": "List the discounts behind loss-making lines in {group}."},
    ],
    ("profit", "Customer Segment"): [
        {"action": "Tighten discount approval for {group} deals", "impact": "Medium", "effort": "Low",
         "first_step": "Require sign-off for {group} discounts above the usual level."},
    ],
    ("profit", "Category Name"): [
        {"action": "Reprice or delist loss-making products in {group}", "impact": "High", "effort": "Medium",
         "first_step": "Rank {group} products by loss-making share."},
    ],
    ("profit", "trend"): [
        {"action": "Investigate margin erosion in {group}", "impact": "Medium", "effort": "Low",
         "first_step": "Compare prices, discounts and costs before and during {group}."},
    ],
    ("profit", None): [
        {"action": "Set a price floor so no order line is sold below cost", "impact": "High",
         "effort": "Medium", "first_step": "List the products and discount levels behind loss-making lines."},
        {"action": "Run a product-level profitability review", "impact": "High", "effort": "Medium",
         "first_step": "Rank products by loss-making share and total profit."},
        {"action": "Track the loss-making share monthly by market and category", "impact": "Medium",
         "effort": "Low", "first_step": "Automate this analysis as a monthly report."},
    ],
}
IMPACT_ORDER = {"High": 0, "Medium": 1, "Low": 2}
EFFORT_ORDER = {"Low": 0, "Medium": 1, "High": 2}


# --- Formatting (reformats tool numbers; never computes new ones) ----------------
def fmt(metric, value):
    return tools.format_value(metric, value)


def pp(gap):
    """A gap from a tool result (e.g. 0.4049) as percentage points: '+40.5 pp'."""
    return f"{gap * 100:+.1f} pp"


def pct(share):
    return f"{share:.1%}"


def money(value):
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e6:
        return f"{sign}${value / 1e6:,.2f}M"
    if value >= 1e3:
        return f"{sign}${value / 1e3:,.0f}K"
    return f"{sign}${value:,.0f}"


def label(column):
    return "Category" if column == "Category Name" else column


PLURALS = {"Shipping Mode": "Shipping modes", "Market": "Markets",
           "Customer Segment": "Customer segments", "Category Name": "Categories"}


# --- Question routing ----------------------------------------------------------------
def classify_question(question):
    """The supported question type whose keywords best match, or None if none match."""
    text = question.lower()
    scores = {qtype: sum(word in text for word in spec["keywords"])
              for qtype, spec in QUESTION_TYPES.items()}
    best = max(scores, key=scores.get)   # on a tie, the type listed first wins
    return best if scores[best] > 0 else None


# --- Running tools and emitting events ----------------------------------------------
class Session:
    """Keeps track of one run: tool results, call numbering, and pacing."""

    def __init__(self, delay):
        self.delay = delay
        self.results = {}   # result_id -> result, e.g. "R1" -> {...}
        self.calls = 0
        self.charts = 0

    def emit(self, event):
        """Pause briefly (so the UI can stream), then hand the event back to be yielded."""
        if self.delay:
            time.sleep(self.delay)
        return event

    def call(self, tool, **inputs):
        """
        Run one tool: yields a tool_call event and a tool_result event, and
        returns the result. Use it as: result = yield from session.call(...)
        """
        self.calls += 1
        call_id = f"c{self.calls}"
        yield self.emit(events.tool_call(call_id, tool, inputs))
        try:
            if tool == "make_chart":
                source = self.results[inputs["result_id"]]
                result = tools.make_chart(source, inputs["chart_type"], inputs["title"])
            else:
                result = ANALYSIS_TOOLS[tool](**inputs)
        except Exception as err:
            # Report the failure so every tool_call gets a result, then stop the run
            yield self.emit(events.tool_result(call_id, tool, {"error": f"{type(err).__name__}: {err}"}))
            raise
        if tool != "make_chart" and "error" not in result:
            result = {"result_id": f"R{len(self.results) + 1}", **result}
            self.results[result["result_id"]] = result
        yield self.emit(events.tool_result(call_id, tool, result))
        return result

    def chart(self, result, chart_type, title):
        """Save a chart of an earlier result: tool events, then a chart event."""
        saved = yield from self.call("make_chart", result_id=result["result_id"],
                                     chart_type=chart_type, title=title)
        if "error" not in saved:
            self.charts += 1
            yield self.emit(events.chart(saved["path"], title, chart_type, result["result_id"]))
        return saved


ANALYSIS_TOOLS = {
    "segment_analysis": tools.segment_analysis,
    "trend_over_time": tools.trend_over_time,
    "impact_estimate": tools.impact_estimate,
    "drill_down": tools.drill_down,
}


# --- Judging a branch ---------------------------------------------------------------
def assess(branch, result, spec, threshold, min_rows):
    """
    Apply the driver rule to one branch's analysis result. Returns a finding:
    the branch, its verdict ("kept"/"dropped"/"untested") and one line of evidence.
    """
    finding = {**branch, "result": result, "status": "untested", "evidence": ""}
    if "error" in result:
        finding["evidence"] = f"Could not test: {result['error']}"
        return finding

    # Only reliable segments count: enough order lines, or (for time) a normal-volume period
    if branch["column"]:
        candidates = [{"name": g["group"], **g} for g in result["groups"] if g["rows"] >= min_rows]
    else:
        candidates = [{"name": p["period"], **p} for p in result["periods"] if not p["low_volume"]]
    if not candidates:
        finding["evidence"] = "No segment has enough order lines to judge."
        return finding

    worst = max(candidates, key=lambda c: c["gap_vs_baseline"])
    best = min(candidates, key=lambda c: c["gap_vs_baseline"])
    metric, baseline = spec["metric"], result["baseline"]
    finding.update(worst=worst, best=best, baseline=baseline)

    if worst["gap_vs_baseline"] > threshold:
        finding["status"] = "kept"
        finding["evidence"] = (f"{worst['name']} at {fmt(metric, worst['value'])} vs "
                               f"{fmt(metric, baseline)} overall ({pp(worst['gap_vs_baseline'])}).")
    else:
        unit = "segment" if branch["column"] else "period"
        finding["status"] = "dropped"
        finding["evidence"] = (f"Every reliable {unit} is within {threshold * 100:g} pp of "
                               f"{fmt(metric, baseline)}; worst is {worst['name']} at "
                               f"{fmt(metric, worst['value'])} ({pp(worst['gap_vs_baseline'])}).")
    return finding


def tree_branches(findings, spec):
    """The issue tree in the events format, built from the findings so far."""
    branches = []
    for f in findings:
        branch = {"name": f["name"], "hypothesis": spec["hypotheses"][f["key"]],
                  "status": f["status"], "evidence": f["evidence"]}
        if f.get("sub_branches"):
            branch["sub_branches"] = f["sub_branches"]
        branches.append(branch)
    return branches


def drill_dimension(column):
    """Which dimension to drill a driver's worst segment across (a check for mix effects)."""
    return "Market" if column == "Shipping Mode" else "Shipping Mode"


# --- The workflow -------------------------------------------------------------------
def run(question, delay=DEFAULT_DELAY, threshold=DRIVER_THRESHOLD, min_rows=MIN_ROWS_FOR_DRIVER):
    """Answer `question`, yielding events (src/events.py) as the work happens."""
    session = Session(delay)
    started = time.time()
    stats = {}
    yield session.emit(events.run_start(question, BRAIN_NAME))
    try:
        stats = yield from workflow(session, question, threshold, min_rows)
    except Exception as err:   # e.g. the data file is missing
        yield session.emit(events.error(f"{type(err).__name__}: {err}"))
        yield session.emit(events.run_end("failed", run_stats(session, started, stats)))
        return
    yield session.emit(events.run_end("complete", run_stats(session, started, stats)))


def run_stats(session, started, extra):
    return {"tool_calls": session.calls, "charts": session.charts,
            "seconds": round(time.time() - started, 1), **extra}


def workflow(session, question, threshold, min_rows):
    """Steps 1-6. Yields events; returns extra stats for the run_end event."""
    # 1. Route the question
    qtype = classify_question(question)
    if qtype is None:
        qtype = "late_deliveries"
        yield session.emit(events.note(
            "This question doesn't match a supported type (late deliveries, profitability, "
            "customer segments), so I'm answering the closest one: late deliveries."))
    spec = QUESTION_TYPES[qtype]
    metric = spec["metric"]
    yield session.emit(events.note(
        f"Question type: {qtype.replace('_', ' ')}. I'll test every branch on the {spec['noun']}."))

    # 2. Predefined issue tree, every branch untested
    findings = [{**b, "status": "untested", "evidence": ""} for b in BRANCHES]
    yield session.emit(events.issue_tree(spec["root"], tree_branches(findings, spec)))

    # 3. Test each branch and apply the driver rule
    for i, branch in enumerate(BRANCHES):
        if branch["column"]:
            result = yield from session.call("segment_analysis", metric=metric, group_by=branch["column"])
        else:
            result = yield from session.call("trend_over_time", metric=metric, freq="quarter")
        findings[i] = assess(branch, result, spec, threshold, min_rows)
        verdict = {"kept": "driver", "dropped": "ruled out", "untested": "not tested"}[findings[i]["status"]]
        yield session.emit(events.note(f"{branch['name']}: {verdict}. {findings[i]['evidence']}"))
    yield session.emit(events.issue_tree(spec["root"], tree_branches(findings, spec)))

    # Context analyses (reported, not tested)
    context = []
    for context_metric, column in spec["context"]:
        result = yield from session.call("segment_analysis", metric=context_metric, group_by=column)
        if "error" not in result:
            context.append(result)

    # 4. Drill into and size each driver, biggest gap first
    drivers = sorted([f for f in findings if f["status"] == "kept"],
                     key=lambda f: f["worst"]["gap_vs_baseline"], reverse=True)
    for driver in drivers:
        if not driver["column"]:   # time has no segment to drill into or size
            continue
        worst = driver["worst"]["name"]
        across = drill_dimension(driver["column"])
        yield session.emit(events.note(f"Drilling into {worst}: does it hold in every {across}?"))
        drill = yield from session.call("drill_down", filters={driver["column"]: worst},
                                        group_by=across, metric=metric)
        driver["drill"] = drill
        if "error" not in drill:
            driver["sub_branches"] = [drill_sub_branch(drill, worst, across, metric, min_rows)]
        driver["impact"] = yield from session.call("impact_estimate", filter_column=driver["column"],
                                                   filter_value=worst)
    if drivers:
        yield session.emit(events.issue_tree(spec["root"], tree_branches(findings, spec)))

    # 5. Charts: each driver, the question's focus, and the trend
    charted = set()
    for f in drivers + [f for f in findings if f["key"] == spec["focus"]]:
        if f["column"] and f["key"] not in charted and "error" not in f["result"]:
            charted.add(f["key"])
            yield from session.chart(f["result"], "bar", f"{spec['noun'].capitalize()} by {label(f['column'])}")
    trend = next(f for f in findings if f["key"] == "trend")
    if "error" not in trend["result"]:
        yield from session.chart(trend["result"], "line", f"{spec['noun'].capitalize()} by quarter")

    # 6. Summary, then the number check on everything the user will read
    yield session.emit(events.note("Writing the summary."))
    summary = write_summary(question, qtype, spec, findings, drivers, context, threshold, min_rows)
    yield session.emit(events.summary(summary))

    shown_text = summary + "\n" + "\n".join(f["evidence"] for f in findings)
    flagged = number_check.unsupported_numbers(
        shown_text, list(session.results.values()), question,
        allowed=[threshold * 100, min_rows])
    return {"question_type": qtype, "drivers": len(drivers), "unsupported_numbers": flagged}


def drill_sub_branch(drill, worst, across, metric, min_rows):
    """Sub-branch for a driver: does the effect hold across the drill dimension?"""
    reliable = [g for g in drill["groups"] if g["rows"] >= min_rows] or drill["groups"]
    high, low = reliable[0], reliable[-1]   # groups are sorted highest first
    # "Holds everywhere" if even the best group is still worse than the company baseline
    holds = low["value"] > drill["company_baseline"]
    conclusion = f"holds in every {across}" if holds else f"is concentrated in {high['group']}"
    return {
        "name": f"{worst} across {across}",
        "status": "kept",
        "evidence": (f"From {low['group']} ({fmt(metric, low['value'])}) to {high['group']} "
                     f"({fmt(metric, high['value'])}): the problem {conclusion}."),
    }


# --- The summary template -------------------------------------------------------------
def choose_recommendations(kind, drivers):
    """2-3 actions: driver-specific rules first, topped up with company-wide ones, then
    ordered by impact (high first) and effort (low first)."""
    picks = []
    for driver in drivers:
        group = driver["worst"]["name"]
        for rule in RECOMMENDATIONS.get((kind, driver["key"]), []):
            picks.append({**rule, "action": rule["action"].format(group=group),
                          "first_step": rule["first_step"].format(group=group),
                          "why": f"{label(driver['column'] or 'Time')}: {driver['evidence']}"})
    for rule in RECOMMENDATIONS[(kind, None)]:
        if len(picks) >= 3:
            break
        why = ("A company-wide safeguard alongside the fixes above." if drivers else
               "No single segment explains the problem, so the fix has to be company-wide.")
        picks.append({**rule, "why": why})
    picks = sorted(picks, key=lambda r: (IMPACT_ORDER[r["impact"]], EFFORT_ORDER[r["effort"]]))
    return picks[:3]


def headline(spec, findings, drivers, context, threshold):
    """
    The top of the executive summary (pyramid principle: the answer first).
    Answers the question's own dimension (the focus) first, then names the real driver.
    """
    metric, noun = spec["metric"], spec["noun"]
    focus = next(f for f in findings if f["key"] == spec["focus"])
    lines = []

    if focus["status"] == "dropped":
        # The dimension the question asks about is NOT the lever: say so first
        lines.append(f"**{PLURALS[focus['column']]} barely differ on the {noun}: from "
                     f"{fmt(metric, focus['best']['value'])} ({focus['best']['name']}) to "
                     f"{fmt(metric, focus['worst']['value'])} ({focus['worst']['name']}), "
                     f"so {label(focus['column'])} is not the lever.**")
        # Who is biggest on that dimension (e.g. which market makes the most profit)
        lines += [context_sentence(r) for r in context if r["group_by"] == focus["column"]]

    if drivers:
        top = drivers[0]
        start = "The real driver is" if lines else f"The {noun} is driven by"
        text = (f"{start} {label(top['column'] or 'time')}: {top['worst']['name']} runs at "
                f"{fmt(metric, top['worst']['value'])} against {fmt(metric, top['baseline'])} "
                f"overall ({pp(top['worst']['gap_vs_baseline'])}).")
        others = [label(d["column"] or "time") for d in drivers[1:]]
        if others:
            text += f" Also above the threshold: {', '.join(others)}."
        lines.append(text if lines else f"**{text}**")
    else:
        baseline = next((f["baseline"] for f in findings if "baseline" in f), None)
        base_text = f" of {fmt(metric, baseline)}" if baseline is not None else ""
        start = "No other branch passes the rule either" if lines else f"No single driver explains the {noun}"
        text = (f"{start}: no segment with enough volume is more than {threshold * 100:g} pp worse "
                f"than the baseline{base_text}, so this is a company-wide issue rather than a "
                f"problem in one pocket.")
        lines.append(text if lines else f"**{text}**")
    return lines


def context_sentence(result):
    """One sentence per context analysis, e.g. which market contributes the most profit."""
    metric, column, groups = result["metric"], label(result["group_by"]), result["groups"]
    high, low = groups[0], groups[-1]
    if metric in ("sales", "profit"):
        return (f"By total {metric}, {high['group']} contributes the most ({money(high['value'])}, "
                f"{pct(high['share_of_total'])} of the total) and {low['group']} the least "
                f"({money(low['value'])}, {pct(low['share_of_total'])}).")
    return (f"{tools.METRICS[metric]['definition'].split(' (')[0].split(',')[0]} by {column}: "
            f"from {fmt(metric, low['value'])} ({low['group']}) to {fmt(metric, high['value'])} "
            f"({high['group']}).")


def impact_sentence(driver, kind):
    impact = driver.get("impact")
    if not impact or "error" in impact:
        return None
    group = driver["worst"]["name"]
    if kind == "service":
        return (f"**{group}:** {impact['late_rows']:,} of {impact['rows']:,} order lines were late, "
                f"carrying {money(impact['late_sales'])} in sales "
                f"({pct(impact['late_sales_share_of_company_sales'])} of company sales) and "
                f"{money(impact['late_profit'])} in profit.")
    return (f"**{group}:** {money(impact['segment_sales'])} in sales and "
            f"{money(impact['segment_profit'])} in profit.")


def write_summary(question, qtype, spec, findings, drivers, context, threshold, min_rows):
    metric, kind = spec["metric"], spec["kind"]
    recommendations = choose_recommendations(kind, drivers)
    lines = ["## Executive summary", ""]
    for sentence in headline(spec, findings, drivers, context, threshold):
        lines += [sentence, ""]

    lines += ["**Key drivers**"]
    if drivers:
        for d in drivers:
            text = f"- **{label(d['column'] or 'Time')}:** {d['evidence']}"
            if d.get("sub_branches"):
                text += f" {d['sub_branches'][0]['evidence']}"
            lines.append(text)
    else:
        lines.append(f"- None passed the rule; the {spec['noun']} is spread evenly across the business.")
    lines += ["", "**Recommendations**"]
    for n, r in enumerate(recommendations, 1):
        lines.append(f"{n}. **{r['action']}.** Impact: {r['impact']}, effort: {r['effort']}.")

    lines += ["", "## Question", "",
              f"{spec['root']} Measured by the {spec['noun']}: "
              f"{tools.METRICS[metric]['definition']}"]
    if classify_question(question) is None:
        lines += ["", "_The question didn't match a supported type, so the closest one was answered._"]

    lines += ["", "## Issue tree and findings", "",
              "| Branch | Hypothesis | Verdict | Evidence |", "|---|---|---|---|"]
    verdicts = {"kept": "**Driver**", "dropped": "Ruled out", "untested": "Not tested"}
    for f in findings:
        lines.append(f"| {f['name']} | {spec['hypotheses'][f['key']]} | {verdicts[f['status']]} "
                     f"| {f['evidence']} |")

    lines += ["", "## Impact", ""]
    impacts = [s for s in (impact_sentence(d, kind) for d in drivers) if s]
    lines += [f"- {s}" for s in impacts] or ["- No segment-level driver to size."]
    lines += [f"- {context_sentence(r)}" for r in context]
    if kind == "service" and impacts:
        lines.append("- These are sales and profit exposed to late delivery, not measured losses.")

    lines += ["", "## Recommendations", "",
              "| # | Action | Why | Impact | Effort | First step |", "|---|---|---|---|---|---|"]
    for n, r in enumerate(recommendations, 1):
        lines.append(f"| {n} | {r['action']} | {r['why']} | {r['impact']} | {r['effort']} | {r['first_step']} |")

    lines += ["", "## Method and caveats", "",
              f"- Rule: a branch is a driver if its worst segment with at least {min_rows:,} order lines "
              f"is more than {threshold * 100:g} pp worse than the baseline. Smaller segments are "
              f"too noisy to act on.",
              "- Counts are order lines (a product within an order), not whole orders."]
    trend = next(f for f in findings if f["key"] == "trend")
    if "error" not in trend["result"]:
        low = [p["period"] for p in trend["result"]["periods"] if p["low_volume"]]
        if low:
            lines.append(f"- Periods with unusually low volume were left out of the trend test: "
                         f"{', '.join(low)}.")
    lines.append("- Produced by the scripted brain: fixed rules, no AI model. "
                 "Every number comes from a tool result.")
    return "\n".join(lines)
