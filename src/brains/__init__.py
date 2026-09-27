"""
Brains: the planners that decide which analyses to run and write the summary.

Every brain has a run(question) generator that yields events in the standard
format (src/events.py), so the UI works the same whichever brain is used.

BRAINS below is the registry the app reads. A brain appears in the app only
if its available() check passes: the scripted brain always works; the others
need their module to exist and their service to be reachable.
"""

import importlib


def _scripted_run(question):
    from src.brains import scripted
    # Read the settings at call time (not import time), so they can be changed,
    # e.g. by tests: scripted.DEFAULT_DELAY = 0
    return scripted.run(question, delay=scripted.DEFAULT_DELAY,
                        threshold=scripted.DRIVER_THRESHOLD, min_rows=scripted.MIN_ROWS_FOR_DRIVER)


def _module_brain(module_name):
    """(available, run) functions for a brain living in src/brains/<module_name>.py."""
    def load():
        try:
            return importlib.import_module(f"src.brains.{module_name}")
        except ImportError:
            return None   # not built yet, or its package isn't installed

    def available():
        module = load()
        return bool(module) and module.is_available()

    def run(question):
        return load().run(question)

    return available, run


_ollama_available, _ollama_run = _module_brain("ollama_brain")
_claude_available, _claude_run = _module_brain("claude_brain")

# Label shown in the app -> how to check it's usable, and how to run it
BRAINS = {
    "Scripted (no LLM)": {"available": lambda: True, "run": _scripted_run},
    "Local LLM (Ollama)": {"available": _ollama_available, "run": _ollama_run},
    "Claude API": {"available": _claude_available, "run": _claude_run},
}


def available_brains():
    """Labels of the brains that can run right now, in display order."""
    return [label for label, brain in BRAINS.items() if brain["available"]()]


def run_brain(label, question):
    """Start the chosen brain on a question; returns its event generator."""
    return BRAINS[label]["run"](question)
