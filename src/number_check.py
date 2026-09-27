"""
Number check: enforces the core rule that every number shown to the user
comes from a tool result.

Used by every brain. It lists numbers in a piece of text (the summary, the
issue tree's evidence) that don't match any tool result.
"""

import re

# A number not glued to a letter, underscore or dot, e.g. 95.3, 180,519, 5.41
NUMBER_IN_TEXT = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?!\d)")


def numbers_in_text(text):
    """Numbers written in text as (as_written, value, decimals), e.g. '95.3' -> ('95.3', 95.3, 1)."""
    found = []
    for match in NUMBER_IN_TEXT.finditer(text):
        raw = match.group().rstrip(",")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        found.append((raw, float(raw.replace(",", "")), decimals))
    return found


def numbers_in_results(obj):
    """Every number inside tool results, including numbers inside description strings."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [abs(float(obj))]
    if isinstance(obj, str):
        return [value for _, value, _ in numbers_in_text(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in numbers_in_results(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in numbers_in_results(v)]
    return []


def unsupported_numbers(text, results, question="", allowed=()):
    """
    List numbers in `text` that don't match any number in `results` (a list of
    tool result dicts). Reformatting is allowed: 0.9532 -> 95.3 (x100),
    5408069 -> 5.41 (millions), rounded to however many decimals are shown.
    Small whole numbers (0-10) are skipped because they're usually list counts,
    like "2-3 actions". Numbers in the question are allowed too, and so are
    `allowed` numbers (e.g. a brain's own rule settings, like "1,000 order lines").
    """
    known = numbers_in_results(results) + [float(a) for a in allowed]
    candidates = {v * scale for v in known for scale in (1, 100, 1e-3, 1e-6, 1e-9)}
    question_numbers = {value for _, value, _ in numbers_in_text(question)}

    unsupported = []
    for raw, value, decimals in numbers_in_text(text):
        if (decimals == 0 and value <= 10) or value in question_numbers:
            continue
        tolerance = 0.5 * 10 ** -decimals + 1e-9   # e.g. "95.3" matches 95.25 to 95.35
        if not any(abs(value - c) <= tolerance for c in candidates):
            unsupported.append(raw)
    return list(dict.fromkeys(unsupported))   # unique, in order of appearance
