"""
Tool definitions shared by every LLM brain (Claude, local Ollama model).

TOOL_DEFINITIONS describes each tool in src/tools.py to a language model:
its name, what it does, and a JSON schema for its inputs. The format is
{name, description, input_schema}; each brain converts it to its provider's
format if needed (e.g. OpenAI-style "function" tools for Ollama).

execute_tool() runs a tool call the model asked for. The model never touches
the data: it names a tool and its arguments, Python runs it, and the result
(with a result_id like "R3") goes back to the model.
"""

from src import tools

METRIC_SCHEMA = {
    "type": "string",
    "enum": list(tools.METRICS),
    "description": (
        "late_rate = share of order lines delivered late (0-1); loss_rate = share of order lines "
        "that lose money (0-1); avg_delay = average of actual minus scheduled shipping days; "
        "sales and profit = totals in dollars."
    ),
}
COLUMN_SCHEMA = {"type": "string", "enum": tools.GROUPABLE_COLUMNS}

# One node of the issue tree (used for branches and, one level down, sub-branches)
BRANCH_FIELDS = {
    "name": {"type": "string", "description": "Short branch name, e.g. 'Logistics: Shipping Mode'."},
    "hypothesis": {"type": "string", "description": "What would be true if this branch drives the problem."},
    "status": {"type": "string", "enum": ["untested", "kept", "dropped"]},
    "evidence": {"type": "string",
                 "description": "One line citing tool results. Leave empty while untested."},
}

TOOL_DEFINITIONS = [
    {
        "name": "record_issue_tree",
        "description": (
            "Record the issue tree so it can be shown to the user as a diagram. It does not analyse "
            "data. Call it after building the tree (all branches 'untested') and again once every "
            "branch is tested, marking each 'kept' (it drives the problem) or 'dropped' (it doesn't) "
            "with one line of evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root": {"type": "string", "description": "The business question, restated."},
                "branches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            **BRANCH_FIELDS,
                            "sub_branches": {
                                "type": "array",
                                "items": {"type": "object", "properties": BRANCH_FIELDS,
                                          "required": ["name", "status"]},
                            },
                        },
                        "required": ["name", "hypothesis", "status"],
                    },
                },
            },
            "required": ["root", "branches"],
        },
    },
    {
        "name": "segment_analysis",
        "description": (
            "Break one metric down by one column across the whole dataset. Returns each group's "
            "value (sorted highest first), its row count, its gap vs the overall baseline "
            "(late_rate, loss_rate, avg_delay) or its share of the total (sales, profit), and a "
            "small_sample flag for groups under 100 rows. Use it to test one branch of the issue tree."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": METRIC_SCHEMA,
                "group_by": {**COLUMN_SCHEMA, "description": "Column to break the metric down by."},
            },
            "required": ["metric", "group_by"],
        },
    },
    {
        "name": "trend_over_time",
        "description": (
            "One metric per month or quarter, in date order, with the row count per period and its "
            "gap vs the overall baseline. Periods with under half the usual volume are flagged "
            "low_volume (possibly incomplete data). Use it to test whether a problem is new, "
            "growing or stable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": METRIC_SCHEMA,
                "freq": {"type": "string", "enum": list(tools.FREQUENCIES)},
            },
            "required": ["metric", "freq"],
        },
    },
    {
        "name": "impact_estimate",
        "description": (
            "For one segment (e.g. Shipping Mode = First Class): how many order lines were late, "
            "and the sales and profit on those late lines, as amounts and as shares of the segment "
            "and company totals. Also the segment's total sales and profit. Use it to size a driver "
            "once you know it matters. Late amounts are exposed to late delivery, not measured losses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filter_column": {**COLUMN_SCHEMA, "description": "Column that defines the segment."},
                "filter_value": {
                    "type": "string",
                    "description": "Exact value in that column, e.g. 'First Class'. "
                                   "segment_analysis shows the valid values.",
                },
            },
            "required": ["filter_column", "filter_value"],
        },
    },
    {
        "name": "drill_down",
        "description": (
            "The same breakdown as segment_analysis, but only inside a subset of rows. Returns the "
            "subset's own baseline and the company-wide baseline. Use it to test a hypothesis inside "
            "one branch, e.g. whether First Class is late in every Market or only in some."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "object",
                    "description": (
                        'Rows to keep, as {column: value} or {column: [values]}, e.g. '
                        '{"Shipping Mode": "First Class"}. Allowed columns: '
                        + ", ".join(tools.GROUPABLE_COLUMNS) + "."
                    ),
                    "additionalProperties": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                },
                "group_by": {**COLUMN_SCHEMA, "description": "Column to break the subset down by."},
                "metric": METRIC_SCHEMA,
            },
            "required": ["filters", "group_by", "metric"],
        },
    },
    {
        "name": "make_chart",
        "description": (
            "Save an interactive chart of an earlier tool result to the outputs folder and return "
            "its file path. Refer to the result by its result_id (e.g. 'R3'); results from "
            "segment_analysis, drill_down and trend_over_time can be charted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "result_id of an earlier tool result."},
                "chart_type": {"type": "string", "enum": tools.CHART_TYPES,
                               "description": "bar for groups, line for trends over time."},
                "title": {"type": "string", "description": "Chart title stating the finding."},
            },
            "required": ["result_id", "chart_type", "title"],
        },
    },
]

# Tool name -> Python function (make_chart and record_issue_tree are handled in execute_tool)
ANALYSIS_TOOLS = {
    "segment_analysis": tools.segment_analysis,
    "trend_over_time": tools.trend_over_time,
    "impact_estimate": tools.impact_estimate,
    "drill_down": tools.drill_down,
}


def execute_tool(name, tool_input, saved_results):
    """
    Run one tool call requested by a model and return the result dict.
    Successful analysis results get a result_id (R1, R2, ...) and are kept in
    saved_results, so make_chart can chart them without the model copying numbers.
    """
    try:
        if not isinstance(tool_input, dict):
            return {"error": "Tool arguments must be a JSON object."}

        if name == "record_issue_tree":
            # Nothing to compute: the tree itself is the tool input
            if not isinstance(tool_input.get("branches"), list) or not tool_input["branches"]:
                return {"error": "branches must be a non-empty list."}
            n = len(tool_input["branches"])
            return {"recorded": True, "description": f"Issue tree recorded with {n} branches."}

        if name == "make_chart":
            result_id = tool_input.get("result_id")
            if result_id not in saved_results:
                available = ", ".join(saved_results) or "none yet"
                return {"error": f"Unknown result_id '{result_id}'. Available: {available}."}
            return tools.make_chart(saved_results[result_id],
                                    tool_input.get("chart_type"), tool_input.get("title"))

        if name not in ANALYSIS_TOOLS:
            return {"error": f"Unknown tool '{name}'. Available: "
                             + ", ".join(t["name"] for t in TOOL_DEFINITIONS) + "."}
        result = ANALYSIS_TOOLS[name](**tool_input)
    except Exception as error:
        # A bug or bad argument in one tool shouldn't crash the whole run;
        # the model sees the error and can try something else.
        return {"error": f"{type(error).__name__}: {error}"}

    if "error" not in result:
        result_id = f"R{len(saved_results) + 1}"
        result = {"result_id": result_id, **result}
        saved_results[result_id] = result
    return result
