"""
Saving and replaying runs.

A run is saved as JSON in demo_runs/: the question, which brain answered it,
when, how long it took, and every event. Because the events include every
tool result, a saved run can be replayed (and its charts redrawn) without the
dataset or any AI model - handy for demos.
"""

import json
import re
import time
from datetime import datetime

from src import tools

DEMO_RUNS_DIR = tools.PROJECT_ROOT / "demo_runs"
REPLAY_DELAY = 0.4    # seconds between events when replaying at 1x (2x and 4x divide it)
SAVED_FIELDS = ("question", "brain", "timestamp", "seconds", "events")


def save_run(run):
    """Save a finished run to demo_runs/ as JSON. Returns the file path."""
    DEMO_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", run["question"].lower()).strip("-")[:40] or "run"
    path = DEMO_RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}.json"
    path.write_text(json.dumps({k: run[k] for k in SAVED_FIELDS}, indent=2), encoding="utf-8")
    return path


def load_run(path):
    return json.loads(path.read_text(encoding="utf-8"))


def saved_runs():
    """Saved run files, newest first."""
    if not DEMO_RUNS_DIR.exists():
        return []
    return sorted(DEMO_RUNS_DIR.glob("*.json"), reverse=True)


def describe(path):
    """Short label for a saved run, e.g. for a dropdown."""
    try:
        data = load_run(path)
        return f"{data['timestamp']} · {data['question'][:48]} ({data['brain']})"
    except (OSError, ValueError, KeyError):
        return f"{path.name} (unreadable)"


def replay(saved_events, speed=1):
    """
    Yield saved events with a short pause, so a replay streams like a live run.
    Saved runs don't store how long each step took, so the pace is even:
    REPLAY_DELAY seconds per event at 1x, half that at 2x, a quarter at 4x.
    """
    for event in saved_events:
        if REPLAY_DELAY:
            time.sleep(REPLAY_DELAY / speed)
        yield event
