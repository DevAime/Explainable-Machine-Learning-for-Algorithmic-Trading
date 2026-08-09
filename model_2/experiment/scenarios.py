"""
scenarios.py

Loads the precomputed experiment/scenarios.json (built offline by
generate_scenarios.py) and builds the fixed, counterbalanced trial
sequence for a given group assignment.

The app NEVER calls the model live -- everything here is static-file
loading, per spec.
"""

from __future__ import annotations

import json
from functools import lru_cache

from config import SCENARIOS_PATH


class ScenarioLoadError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_scenarios() -> dict:
    if not SCENARIOS_PATH.exists():
        raise ScenarioLoadError(
            f"scenarios.json not found at {SCENARIOS_PATH}. Run "
            f"generate_scenarios.py first (offline, once) before starting the app."
        )
    data = json.loads(SCENARIOS_PATH.read_text())
    if not data.get("set_a") or not data.get("set_b"):
        raise ScenarioLoadError("scenarios.json is missing set_a or set_b entries.")
    return data


def build_trial_sequence(group: str) -> list[dict]:
    """
    Returns the ordered list of trial dicts for the given group, each
    tagged with a 'condition' field ('explained' or 'unexplained').

    Fixed trial order within each set (spec: same order for all
    participants) -- we preserve set_a / set_b's order exactly as stored
    in scenarios.json; no shuffling here.

    group_1: Set A (explained) -> Set B (unexplained)
    group_2: Set B (explained) -> Set A (unexplained)
    """
    data = load_scenarios()
    set_a = data["set_a"]
    set_b = data["set_b"]

    if group == "group_1":
        first, first_cond = set_a, "explained"
        second, second_cond = set_b, "unexplained"
    elif group == "group_2":
        first, first_cond = set_b, "explained"
        second, second_cond = set_a, "unexplained"
    else:
        raise ValueError(f"Unknown group: {group!r}")

    sequence = []
    for sc in first:
        sc2 = dict(sc)
        sc2["condition"] = first_cond
        sequence.append(sc2)
    for sc in second:
        sc2 = dict(sc)
        sc2["condition"] = second_cond
        sequence.append(sc2)

    return sequence
