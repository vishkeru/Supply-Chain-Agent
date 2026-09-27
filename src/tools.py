"""
Analysis tools for the Supply Chain Diagnostic Agent.

Core rule: the LLM never computes numbers. It calls these functions and explains
what they return.

Every public tool returns a JSON-serializable dict:
- on success: the numbers plus a short plain-English "description"
- on bad input: {"error": "<message>"}. Tools don't raise on bad input, so the
  agent can read the message, fix its call, and try again.

Note: one row in the dataset is one ORDER LINE (a product inside an order),
not a whole order. All counts and rates below are per order line.
"""

import difflib
import functools
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "DataCoSupplyChainDataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --- Settings ----------------------------------------------------------------
# Columns the agent may group or filter by. High-cardinality columns such as
# Order Id are left out on purpose: grouping by them returns thousands of rows.
GROUPABLE_COLUMNS = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Customer Segment",
    "Category Name",
    "Department Name",
    "Delivery Status",
    "Order Status",
]

# Each metric = which column to use + how to aggregate it.
# "mean" metrics are compared with the overall average (gap);
# "sum" metrics are compared with the grand total (share).
METRICS = {
    "late_rate": {
        "column": "Late_delivery_risk",
        "how": "mean",
        "definition": "Share of order lines delivered late (0 to 1). Cancelled lines count as not late.",
    },
    "avg_delay": {
        "column": "delay_days",
        "how": "mean",
        "definition": "Average of actual minus scheduled shipping days. Negative means early.",
    },
    "sales": {
        "column": "Sales",
        "how": "sum",
        "definition": "Total sales in dollars.",
    },
    "profit": {
        "column": "Order Profit Per Order",
        "how": "sum",
        "definition": "Total profit in dollars (recorded per order line despite the column name).",
    },
}

# Friendly names for time periods -> pandas period codes
FREQUENCIES = {"month": "M", "quarter": "Q"}

CHART_TYPES = ["bar", "line"]

# Groups with fewer order lines than this are flagged as unreliable
SMALL_SAMPLE_ROWS = 100

# Chart colours (same as the EDA notebook)
BLUE = "#2a78d6"
GREY = "#898781"


# --- Loading data ------------------------------------------------------------
def prepare_data(df):
    """Add the derived columns the tools need. Used for both real and test data."""
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
    df["delay_days"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
    return df


@functools.lru_cache(maxsize=1)
def load_data(path=DATA_PATH):
    """Read the CSV once and keep it in memory (lru_cache) for later calls."""
    # The file has some non-UTF-8 characters, so it must be read as latin-1
    raw = pd.read_csv(path, encoding="latin-1")
    return prepare_data(raw)


# --- Validation helpers ------------------------------------------------------
class ToolInputError(Exception):
    """Raised when a tool gets a bad argument. Turned into {"error": ...}."""


def returns_error_dict(func):
    """Decorator: if a tool raises ToolInputError, return {"error": message} instead."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ToolInputError as error:
            return {"error": str(error)}

    return wrapper


def check_choice(value, valid_options, what):
    """Raise a helpful error if `value` isn't one of `valid_options`."""
    if value in valid_options:
        return
    message = f"Unknown {what}: '{value}'."
    options = [str(v) for v in valid_options]
    # First try a partial match ('Asia' -> 'Pacific Asia'), then the closest
    # spelling (catches typos like 'Shiping Mode')
    partial = [o for o in options if str(value).lower() in o.lower()]
    close = partial or difflib.get_close_matches(str(value), options, n=1, cutoff=0.6)
    if close:
        message += f" Did you mean '{close[0]}'?"
    message += " Valid options: " + ", ".join(str(v) for v in valid_options) + "."
    raise ToolInputError(message)


def check_metric(metric):
    check_choice(metric, list(METRICS), "metric")


def check_column(df, column):
    # Only allow groupable columns that actually exist in this DataFrame
    valid = [c for c in GROUPABLE_COLUMNS if c in df.columns]
    check_choice(column, valid, "column")


def apply_filters(df, filters):
    """
    Keep only rows matching every filter, e.g. {"Shipping Mode": "First Class"}.
    A filter value can also be a list, e.g. {"Market": ["Europe", "LATAM"]}.
    """
    if not isinstance(filters, dict) or not filters:
        raise ToolInputError(
            'filters must be a non-empty object like {"Shipping Mode": "First Class"}.'
        )

    subset = df
    for column, value in filters.items():
        check_column(df, column)
        values = value if isinstance(value, list) else [value]
        valid_values = sorted(df[column].dropna().unique().tolist())
        for v in values:
            check_choice(v, valid_values, f"value for '{column}'")
        subset = subset[subset[column].isin(values)]

    if subset.empty:
        raise ToolInputError(f"No order lines match the filters {filters}.")
    return subset


# --- Calculation helpers -----------------------------------------------------
def to_number(value):
    """Convert numpy numbers to plain Python floats (JSON-safe), rounded."""
    return round(float(value), 4)


def format_value(metric, value):
    """Human-readable version of a metric value, used in descriptions."""
    if metric == "late_rate":
        return f"{value:.1%}"
    if metric == "avg_delay":
        return f"{value:+.2f} days"
    return f"${value:,.0f}"


def metric_value(df, metric):
    """The metric for a whole DataFrame (mean or sum of its column)."""
    spec = METRICS[metric]
    column = df[spec["column"]]
    return column.mean() if spec["how"] == "mean" else column.sum()


def breakdown(df, metric, group_by):
    """
    The metric for each value of `group_by`, sorted highest first.
    Returns (baseline, list of group dicts).
    """
    spec = METRICS[metric]
    baseline = metric_value(df, metric)

    grouped = df.groupby(group_by)[spec["column"]].agg(value=spec["how"], rows="size")
    # "stable" keeps ties in alphabetical order, so results are repeatable
    grouped = grouped.sort_values("value", ascending=False, kind="stable")

    groups = []
    for name, row in grouped.iterrows():
        rows = int(row["rows"])
        entry = {"group": str(name), "value": to_number(row["value"]), "rows": rows}
        if spec["how"] == "mean":
            # How far above (+) or below (-) the overall average this group is
            entry["gap_vs_baseline"] = to_number(row["value"] - baseline)
        else:
            # What fraction of the total this group accounts for
            entry["share_of_total"] = to_number(row["value"] / baseline) if baseline else None
        entry["small_sample"] = rows < SMALL_SAMPLE_ROWS
        groups.append(entry)

    return baseline, groups


def describe_breakdown(metric, group_by, baseline, groups, n_rows):
    """One-line summary of a breakdown, built only from computed numbers."""
    top, bottom = groups[0], groups[-1]
    return (
        f"{metric} by {group_by} across {n_rows:,} order lines, highest first. "
        f"Overall: {format_value(metric, baseline)}. "
        f"Highest: {top['group']} ({format_value(metric, top['value'])}); "
        f"lowest: {bottom['group']} ({format_value(metric, bottom['value'])})."
    )


# --- Tools -------------------------------------------------------------------
@returns_error_dict
def segment_analysis(metric, group_by, df=None):
    """A metric broken down by one column, sorted, with the overall baseline."""
    if df is None:
        df = load_data()
    check_metric(metric)
    check_column(df, group_by)

    baseline, groups = breakdown(df, metric, group_by)
    return {
        "metric": metric,
        "metric_definition": METRICS[metric]["definition"],
        "group_by": group_by,
        "rows_analysed": len(df),
        "baseline": to_number(baseline),
        "groups": groups,
        "description": describe_breakdown(metric, group_by, baseline, groups, len(df)),
    }


@returns_error_dict
def trend_over_time(metric, freq="month", df=None):
    """A metric per month or quarter, in date order."""
    if df is None:
        df = load_data()
    check_metric(metric)
    check_choice(freq, list(FREQUENCIES), "freq")

    spec = METRICS[metric]
    period = df["order_date"].dt.to_period(FREQUENCIES[freq])
    grouped = df.groupby(period)[spec["column"]].agg(value=spec["how"], rows="size")

    # Flag periods with far fewer order lines than usual: they may be incomplete data
    low_volume_cutoff = grouped["rows"].median() / 2

    periods = []
    for name, row in grouped.iterrows():
        periods.append({
            "period": str(name),
            "value": to_number(row["value"]),
            "rows": int(row["rows"]),
            "low_volume": bool(row["rows"] < low_volume_cutoff),
        })

    first, last = periods[0], periods[-1]
    n_low = sum(p["low_volume"] for p in periods)
    description = (
        f"{metric} per {freq} from {first['period']} to {last['period']} ({len(periods)} periods). "
        f"First: {format_value(metric, first['value'])}; last: {format_value(metric, last['value'])}."
    )
    if n_low:
        description += (
            f" {n_low} period(s) have under half the usual order volume, so treat them with caution."
        )

    return {
        "metric": metric,
        "metric_definition": spec["definition"],
        "freq": freq,
        "periods": periods,
        "description": description,
    }


@returns_error_dict
def impact_estimate(filter_column, filter_value, df=None):
    """Sales and profit on late order lines in one segment, vs the segment and company totals."""
    if df is None:
        df = load_data()
    segment = apply_filters(df, {filter_column: filter_value})
    late = segment[segment["Late_delivery_risk"] == 1]

    segment_sales = segment["Sales"].sum()
    segment_profit = segment["Order Profit Per Order"].sum()
    late_sales = late["Sales"].sum()
    late_profit = late["Order Profit Per Order"].sum()
    company_sales = df["Sales"].sum()
    company_profit = df["Order Profit Per Order"].sum()

    def share(part, whole):
        # Avoid dividing by zero
        return to_number(part / whole) if whole else None

    return {
        "segment": {"column": filter_column, "value": filter_value},
        "rows": len(segment),
        "late_rows": len(late),
        "late_rate": share(len(late), len(segment)),
        "segment_sales": to_number(segment_sales),
        "segment_profit": to_number(segment_profit),
        "late_sales": to_number(late_sales),
        "late_profit": to_number(late_profit),
        "late_share_of_segment_sales": share(late_sales, segment_sales),
        "late_sales_share_of_company_sales": share(late_sales, company_sales),
        "late_profit_share_of_company_profit": share(late_profit, company_profit),
        "description": (
            f"In {filter_column} = {filter_value}, {len(late):,} of {len(segment):,} order lines "
            f"were late, carrying ${late_sales:,.0f} in sales and ${late_profit:,.0f} in profit. "
            f"That is {late_sales / company_sales:.1%} of all company sales. "
            "These are the amounts exposed to late delivery, not a measured loss."
        ),
    }


@returns_error_dict
def drill_down(filters, group_by, metric="late_rate", df=None):
    """Like segment_analysis, but only inside the rows matching `filters`."""
    if df is None:
        df = load_data()
    check_metric(metric)
    check_column(df, group_by)
    subset = apply_filters(df, filters)

    subset_baseline, groups = breakdown(subset, metric, group_by)
    company_baseline = metric_value(df, metric)
    return {
        "metric": metric,
        "metric_definition": METRICS[metric]["definition"],
        "filters": filters,
        "group_by": group_by,
        "rows_analysed": len(subset),
        "baseline": to_number(subset_baseline),
        "company_baseline": to_number(company_baseline),
        "groups": groups,
        "description": (
            f"Within {filters}: "
            + describe_breakdown(metric, group_by, subset_baseline, groups, len(subset))
            + f" Company-wide: {format_value(metric, company_baseline)}."
        ),
    }


@returns_error_dict
def make_chart(data, chart_type, title, output_dir=None):
    """
    Save a chart of another tool's output as an HTML file and return its path.
    `data` must be the result of segment_analysis, drill_down or trend_over_time.
    """
    check_choice(chart_type, CHART_TYPES, "chart_type")
    if not isinstance(title, str) or not title.strip():
        raise ToolInputError("title must be a non-empty string.")
    if not isinstance(data, dict) or "error" in data:
        raise ToolInputError("data must be a successful result from another tool, not an error.")

    # Work out the labels and values from whichever tool produced the data
    if "groups" in data:
        labels = [g["group"] for g in data["groups"]]
        axis_name = data.get("group_by", "Group")
    elif "periods" in data:
        labels = [p["period"] for p in data["periods"]]
        axis_name = "Period"
    else:
        raise ToolInputError(
            "data must be the output of segment_analysis, drill_down or trend_over_time."
        )
    items = data.get("groups") or data.get("periods")
    values = [item["value"] for item in items]

    metric = data.get("metric", "value")
    axis_labels = {"x": axis_name, "y": metric}

    if chart_type == "line":
        fig = px.line(x=labels, y=values, markers=True, labels=axis_labels, title=title)
        fig.update_traces(line_color=BLUE, line_width=2)
        value_axis = "y"
    elif "groups" in data:
        # Horizontal bars read best for named groups
        fig = px.bar(x=values, y=labels, orientation="h",
                     labels={"x": metric, "y": axis_name}, title=title)
        fig.update_traces(marker_color=BLUE)
        fig.update_yaxes(categoryorder="total ascending")
        value_axis = "x"
        # Dashed line at the overall average, for mean metrics only
        if "baseline" in data and METRICS.get(metric, {}).get("how") == "mean":
            fig.add_vline(x=data["baseline"], line_dash="dash", line_color=GREY,
                          annotation_text=f"Overall {format_value(metric, data['baseline'])}")
    else:
        fig = px.bar(x=labels, y=values, labels=axis_labels, title=title)
        fig.update_traces(marker_color=BLUE)
        value_axis = "y"

    if metric == "late_rate":
        # Show rates as percentages on the value axis (xaxis or yaxis)
        fig.update_layout(**{f"{value_axis}axis_tickformat": ".0%"})
    fig.update_layout(template="plotly_white", title_x=0, showlegend=False)

    # File name: slug of the title + timestamp, e.g. late-rate-by-mode_20260926_142501_123456.html
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "chart"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = Path(output_dir) if output_dir else OUTPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slug}_{timestamp}.html"

    # include_plotlyjs="cdn" keeps files small; viewing them needs an internet connection
    fig.write_html(path, include_plotlyjs="cdn")

    return {
        "path": str(path),
        "chart_type": chart_type,
        "title": title,
        "description": f"Saved a {chart_type} chart titled '{title}' to {path.name}.",
    }
