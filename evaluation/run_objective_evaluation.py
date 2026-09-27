from __future__ import annotations

r"""
SekAI formal evaluation runner for CM3070 Objectives 1-3.

Place in:
    <SekAI repository>/evaluation/run_objective_evaluation.py

This is NOT a replacement for pytest. It is a controlled evaluation harness
that imports and exercises the real SekAI backend while using a separate
evaluation SQLite database.

Recommended workflow:
    1. Run pytest.
    2. Start Ollama.
    3. Close the normal SekAI/FastAPI app.
    4. Run a smoke test.
    5. Run the formal experiment with --reset-db.

Smoke test:
    .\.venv\Scripts\python.exe .\evaluation\run_objective_evaluation.py ^
        --reset-db --latency-runs 2 --pairs 1 --max-turns 3 ^
        --review-mode interactive

Formal run:
    .\.venv\Scripts\python.exe .\evaluation\run_objective_evaluation.py ^
        --reset-db ^
        --model qwen2.5:1.5b-instruct-q4_K_M ^
        --latency-runs 20 ^
        --pairs 2 ^
        --max-turns 7 ^
        --review-mode interactive

After the run, complete objective2_manual_story_review.csv, then finalise:
    .\.venv\Scripts\python.exe .\evaluation\run_objective_evaluation.py ^
        --finalize-manual-review
"""

import argparse
import asyncio
import csv
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import httpx


# ---------------------------------------------------------------------------
# Import the REAL SekAI implementation.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from backend import ai_service, db, services
    from backend.continuity_service import get_nli_model
    from backend.schemas import (
        GenerateStoryRequest,
        ProjectCreateRequest,
        SaveNodeRequest,
        StateDelta,
        StoryChoice,
        StoryMemory,
    )
except ImportError as error:
    raise SystemExit(
        "Could not import SekAI. Put this file in evaluation/ and run it "
        "from the repository root.\n"
        f"Underlying error: {error}"
    ) from error


# ---------------------------------------------------------------------------
# Report targets.
# ---------------------------------------------------------------------------

O1_LATENCY_TARGET = 15.0
O2_FACT_TARGET = 0.85
O2_CRITICAL_TARGET = 0
O2_ERRORS_PER_BRANCH_TARGET = 2
O3_CHANGE_TARGET = 0.80

DEFAULT_MODEL = "qwen2.5:1.5b-instruct-q4_K_M"
DEFAULT_TEMPERATURE = 0.5

RESULT_DIR = REPO_ROOT / "evaluation" / "results" / "formal_objectives"
DEFAULT_DB = RESULT_DIR / "sekai_objective_evaluation.db"

O1_LATENCY_CSV = RESULT_DIR / "objective1_latency.csv"
O1_BRANCH_CSV = RESULT_DIR / "objective1_branching.csv"
O2_FACT_CSV = RESULT_DIR / "objective2_continuity.csv"
O2_MANUAL_CSV = RESULT_DIR / "objective2_manual_story_review.csv"
O2_RUNTIME_REVIEW_CSV = RESULT_DIR / "objective2_generation_review.csv"
O3_CSV = RESULT_DIR / "objective3_consequences.csv"
RAW_JSONL = RESULT_DIR / "objective_raw_outputs.jsonl"
SUMMARY_JSON = RESULT_DIR / "objective_summary.json"
SUMMARY_CSV = RESULT_DIR / "objective_summary.csv"


# ---------------------------------------------------------------------------
# Controlled test story.
# ---------------------------------------------------------------------------

WORLD = (
    "A ruined watchtower stands above an old mountain road. Its beacon has "
    "started glowing blue after decades of silence. Guard Harlan waits at the "
    "entrance. The upper floors contain abandoned rooms, old mechanisms and "
    "signs that something has recently moved through the tower."
)

OBJECTIVE = "Reach the beacon chamber and discover why the tower beacon is active."

START_TITLE = "The Entrance Hall"

START_TEXT = (
    "Rain rattles against the broken doorway as the traveller enters the ruined "
    "watchtower. Guard Harlan waits beside a cold brazier, watching the dark "
    "spiral staircase with obvious unease. He says the beacon above started "
    "glowing blue during the night even though nobody has maintained it for "
    "years. The traveller carries a lit Torch and an Iron Key recovered outside. "
    "No attack has begun, but occasional scraping sounds echo from the floors "
    "above. The route upward starts at the staircase beside Harlan."
)

INITIAL_MEMORY = {
    "location": "Entrance Hall",
    "active_characters": ["Guard Harlan"],
    "inventory": ["Torch", "Iron Key"],
    "relationships": {"Guard Harlan": "Wary"},
    "goals": ["Reach the beacon chamber"],
    "unresolved_clues": ["Why the beacon is glowing"],
    "decisions": [],
    "threat": "Medium",
}


@dataclass(frozen=True)
class Turn:
    number: int
    action: str
    gm_instruction: str
    expected: dict[str, Any]
    expected_change: bool
    action_type: str = "do"


BRANCH_A: tuple[Turn, ...] = (
    Turn(
        1,
        "Ask Guard Harlan what he heard when the beacon activated last night.",
        (
            "Harlan must answer, but no tracked story state changes. Keep the "
            "location, inventory, relationship, goal and threat unchanged. "
            "Return an empty applied_delta."
        ),
        {
            "location": "Entrance Hall",
            "inventory_contains": ["Torch", "Iron Key"],
            "relationship": {"Guard Harlan": "Wary"},
            "goals_contains": ["Reach the beacon chamber"],
            "clues_contains": ["Why the beacon is glowing"],
            "threat": "Medium",
        },
        False,
        action_type="ask",
    ),
    Turn(
        2,
        "Climb the spiral staircase with Guard Harlan and arrive at the "
        "Second-Floor Landing.",
        (
            "Guard Harlan accompanies the traveller upstairs. The traveller "
            "and Harlan must visibly arrive at the location named exactly "
            "'Second-Floor Landing'. Set location to exactly that value. "
            "Do not add or remove inventory."
        ),
        {
            "location": "Second-Floor Landing",
            "inventory_contains": ["Torch", "Iron Key"],
            "relationship": {"Guard Harlan": "Wary"},
            "goals_contains": ["Reach the beacon chamber"],
            "threat": "Medium",
        },
        True,
    ),
    Turn(
        3,
        "Open the metal box on the landing and take the Silver Key inside.",
        (
            "The item must be named exactly 'Silver Key'. The story must show "
            "the traveller obtaining it and applied_delta must add 'Silver Key'. "
            "Keep the Torch and Iron Key."
        ),
        {
            "location": "Second-Floor Landing",
            "inventory_contains": ["Torch", "Iron Key", "Silver Key"],
            "inventory_absent": ["Bronze Token"],
            "relationship": {"Guard Harlan": "Wary"},
            "goals_contains": ["Reach the beacon chamber"],
            "threat": "Medium",
        },
        True,
    ),
    Turn(
        4,
        "Pull Guard Harlan out of danger before a falling timber strikes him.",
        (
            "Harlan must recognise that the traveller saved him. Change the "
            "relationship with 'Guard Harlan' to exactly 'Trusting' and visibly "
            "justify that change in story_text."
        ),
        {
            "inventory_contains": ["Torch", "Iron Key", "Silver Key"],
            "relationship": {"Guard Harlan": "Trusting"},
            "goals_contains": ["Reach the beacon chamber"],
            "threat": "Medium",
        },
        True,
    ),
    Turn(
        5,
        "A tower guardian bursts from behind the gears and attacks.",
        (
            "The attack must happen in this scene and meaningfully escalate "
            "danger. Set threat to exactly 'High'. Preserve the existing "
            "inventory and Trusting relationship."
        ),
        {
            "inventory_contains": ["Torch", "Iron Key", "Silver Key"],
            "relationship": {"Guard Harlan": "Trusting"},
            "goals_contains": ["Reach the beacon chamber"],
            "threat": "High",
        },
        True,
        action_type="continue"
    ),
    Turn(
        6,
        "Use the Iron Key in the upper lock; it snaps and can no longer be carried.",
        (
            "The story must explicitly show the 'Iron Key' breaking or becoming "
            "unusable. applied_delta must remove 'Iron Key'. Keep Torch and "
            "Silver Key."
        ),
        {
            "inventory_contains": ["Torch", "Silver Key"],
            "inventory_absent": ["Iron Key", "Bronze Token"],
            "relationship": {"Guard Harlan": "Trusting"},
            "goals_contains": ["Reach the beacon chamber"],
            "threat": "High",
        },
        True,
    ),
    Turn(
        7,
        "Pass through the final doorway and enter the Beacon Chamber.",
        (
            "Reach the location named exactly 'Beacon Chamber'. Set location "
            "to 'Beacon Chamber' and complete the exact goal "
            "'Reach the beacon chamber'. Preserve the Silver Key."
        ),
        {
            "location": "Beacon Chamber",
            "inventory_contains": ["Torch", "Silver Key"],
            "inventory_absent": ["Iron Key", "Bronze Token"],
            "relationship": {"Guard Harlan": "Trusting"},
            "goals_absent": ["Reach the beacon chamber"],
            "threat": "High",
        },
        True,
    ),
)


BRANCH_B: tuple[Turn, ...] = (
    Turn(
        1,
        "Ask Guard Harlan what he heard when the beacon activated last night.",
        (
            "Harlan must answer, but no tracked story state changes. Keep the "
            "location, inventory, relationship, goal and threat unchanged. "
            "Return an empty applied_delta."
        ),
        {
            "location": "Entrance Hall",
            "inventory_contains": ["Torch", "Iron Key"],
            "relationship": {"Guard Harlan": "Wary"},
            "goals_contains": ["Reach the beacon chamber"],
            "clues_contains": ["Why the beacon is glowing"],
            "threat": "Medium",
        },
        False,
        action_type="ask",
    ),
    Turn(
        2,
        "Take the western stairs with Guard Harlan and arrive at the "
        "West Stair Landing.",
        (
            "Guard Harlan accompanies the traveller. They must visibly arrive "
            "at the location named exactly 'West Stair Landing'. Set location "
            "to exactly that value. Do not add or remove inventory."
        ),
        {
            "location": "West Stair Landing",
            "inventory_contains": ["Torch", "Iron Key"],
            "relationship": {"Guard Harlan": "Wary"},
            "goals_contains": ["Reach the beacon chamber"],
            "threat": "Medium",
        },
        True,
    ),
    Turn(
        3,
        "Search a cracked niche and take the Bronze Token found inside.",
        (
            "The item must be named exactly 'Bronze Token'. The story must show "
            "the traveller obtaining it and applied_delta must add 'Bronze Token'. "
            "Do not add a Silver Key."
        ),
        {
            "location": "West Stair Landing",
            "inventory_contains": ["Torch", "Iron Key", "Bronze Token"],
            "inventory_absent": ["Silver Key"],
            "relationship": {"Guard Harlan": "Wary"},
            "goals_contains": ["Reach the beacon chamber"],
            "threat": "Medium",
        },
        True,
    ),
    Turn(
        4,
        "Accuse Guard Harlan of hiding information about the beacon.",
        (
            "Harlan must react negatively. Change the relationship with "
            "'Guard Harlan' to exactly 'Suspicious'. Do not change inventory."
        ),
        {
            "inventory_contains": ["Torch", "Iron Key", "Bronze Token"],
            "inventory_absent": ["Silver Key"],
            "relationship": {"Guard Harlan": "Suspicious"},
            "goals_contains": ["Reach the beacon chamber"],
            "threat": "Medium",
        },
        True,
    ),
    Turn(
        5,
        "A tower guardian discovers the traveller and attacks near the western gears.",
        (
            "The guardian attack must happen and meaningfully escalate danger. "
            "Set threat to exactly 'High'. Preserve the Bronze Token and the "
            "Suspicious relationship."
        ),
        {
            "inventory_contains": ["Torch", "Iron Key", "Bronze Token"],
            "inventory_absent": ["Silver Key"],
            "relationship": {"Guard Harlan": "Suspicious"},
            "goals_contains": ["Reach the beacon chamber"],
            "threat": "High",
        },
        True,
        action_type="continue"
    ),
    Turn(
        6,
        "Use the Iron Key to open a sealed hatch; the mechanism consumes the key.",
        (
            "The story must explicitly show the 'Iron Key' being consumed or "
            "lost. applied_delta must remove 'Iron Key'. Keep Torch and "
            "Bronze Token."
        ),
        {
            "inventory_contains": ["Torch", "Bronze Token"],
            "inventory_absent": ["Iron Key", "Silver Key"],
            "relationship": {"Guard Harlan": "Suspicious"},
            "goals_contains": ["Reach the beacon chamber"],
            "threat": "High",
        },
        True,
    ),
    Turn(
        7,
        "Climb through the hatch and enter the Beacon Chamber.",
        (
            "Reach the location named exactly 'Beacon Chamber'. Set location "
            "to 'Beacon Chamber' and complete the exact goal "
            "'Reach the beacon chamber'. Preserve Bronze Token and do not "
            "invent a Silver Key."
        ),
        {
            "location": "Beacon Chamber",
            "inventory_contains": ["Torch", "Bronze Token"],
            "inventory_absent": ["Iron Key", "Silver Key"],
            "relationship": {"Guard Harlan": "Suspicious"},
            "goals_absent": ["Reach the beacon chamber"],
            "threat": "High",
        },
        True,
    ),
)


# ---------------------------------------------------------------------------
# Generic helpers.
# ---------------------------------------------------------------------------

def now() -> str:
    return datetime.now(UTC).isoformat()


def normalise(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", value.casefold()))


def contains(values: Iterable[str], expected: str) -> bool:
    key = normalise(expected)
    return any(normalise(str(value)) == key for value in values)


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def add_jsonl(record: dict[str, Any]) -> None:
    RAW_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RAW_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def choices(next_turn: Turn | None) -> list[StoryChoice]:
    if next_turn is None:
        primary = "Continue cautiously toward the beacon."
        action_type = "continue"
    else:
        primary = next_turn.action
        action_type = next_turn.action_type

    return [
        StoryChoice(
            action_type=action_type,
            label=primary,
        ),
        StoryChoice(
            action_type="do",
            label="Inspect the immediate surroundings.",
        ),
        StoryChoice(
            action_type="ask",
            label="Ask Guard Harlan for a warning.",
        ),
    ]


def create_project(name: str) -> dict[str, Any]:
    return db.create_project(
        ProjectCreateRequest(
            title=name,
            genre="Fantasy",
            tone="Mysterious adventure",
            world_summary=WORLD,
            main_objective=OBJECTIVE,
            starting_scene_title=START_TITLE,
            starting_scene_text=START_TEXT,
            initial_memory=StoryMemory.model_validate(INITIAL_MEMORY),
            story_element_mode="off",
            story_elements=[],
        )
    )


def set_controlled_choices(
    node: dict[str, Any],
    next_turn: Turn | None,
) -> dict[str, Any]:
    new_choices = choices(next_turn)

    db.update_node_content(
        node["id"],
        node["title"],
        node["story_text"],
        [
            item.model_dump()
            for item in new_choices
        ],
        node.get(
            "state_delta",
            StateDelta().model_dump(),
        ),
    )

    updated = db.get_node(node["id"])
    if not updated:
        raise RuntimeError(
            "Could not reload controlled parent node."
        )

    return updated


def first_choice_id(node: dict[str, Any]) -> str:
    if not node.get("choices"):
        raise RuntimeError("Controlled node has no choices.")
    return str(node["choices"][0]["id"])


def installed_models() -> set[str]:
    response = httpx.get(
        f"{ai_service.OLLAMA_BASE_URL}/api/tags",
        timeout=10.0,
    )
    response.raise_for_status()
    return {
        item["name"]
        for item in response.json().get("models", [])
        if item.get("name")
    }


async def generate(
    *,
    project_id: str,
    branch_id: str,
    parent_id: str,
    choice_id: str | None,
    model: str,
    temperature: float,
    player_input: str,
    gm_instruction: str,
) -> tuple[Any, float]:
    request = GenerateStoryRequest(
        project_id=project_id,
        branch_id=branch_id,
        parent_node_id=parent_id,
        selected_choice_id=choice_id,
        model=model,
        temperature=temperature,
        interaction_type="continue",
        player_input=player_input,
        gm_instruction=gm_instruction,
    )

    started = time.perf_counter()
    response = await ai_service.generate_story(request)
    wall_seconds = time.perf_counter() - started
    return response, wall_seconds


def continuity_issue_count(response: Any) -> tuple[int, int]:
    all_count = 0
    blocking = 0

    for issue in getattr(response, "validation_issues", []) or []:
        if getattr(issue, "stage", "") == "continuity":
            all_count += 1
            if getattr(issue, "severity", "") == "blocking":
                blocking += 1

    return all_count, blocking


def review_issue_rows(response: Any) -> list[dict[str, Any]]:
    """Return validation issues in a serialisable form."""
    return [
        issue.model_dump()
        for issue in (getattr(response, "validation_issues", []) or [])
    ]


def is_human_reviewable_continuity_warning(response: Any) -> bool:
    """
    Permit human adjudication only for non-blocking NLI continuity warnings.

    Deterministic/state/length/repetition failures remain non-overridable in the
    controlled experiment. This prevents the reviewer from repairing the writer's
    output simply to improve the score.
    """
    issues = list(getattr(response, "validation_issues", []) or [])
    if response.status != "review_required" or not issues:
        return False

    return all(
        getattr(issue, "stage", "") == "continuity"
        and getattr(issue, "severity", "") != "blocking"
        and bool(getattr(issue, "can_override", False))
        for issue in issues
    )


def human_review_continuity_warning(
    *,
    args: argparse.Namespace,
    pair: int,
    branch_label: str,
    turn: Turn,
    parent: dict[str, Any],
    response: Any,
) -> tuple[bool, str, str]:
    """
    Ask the human evaluator to adjudicate an NLI-only warning.

    The expected-answer dictionary is deliberately NOT displayed. The reviewer
    sees the selected action, authoritative parent state, generated prose, delta,
    and NLI warning, which mirrors the GM review task without exposing the
    evaluator's scoring labels.
    """
    if args.review_mode != "interactive":
        return False, "strict_reject", "Interactive human review was disabled."

    print("\n" + "-" * 76)
    print("HUMAN CONTINUITY REVIEW REQUIRED")
    print("-" * 76)
    print(f"Pair / branch / turn : {pair} / {branch_label} / {turn.number}")
    print(f"Selected action       : {turn.action}")
    print("\nAuthoritative state BEFORE this scene:")
    print(json.dumps(parent.get("memory_snapshot", {}), ensure_ascii=False, indent=2))
    print("\nGenerated story:")
    print(response.draft.story_text)
    print("\nGenerated applied_delta:")
    print(json.dumps(response.draft.applied_delta.model_dump(), ensure_ascii=False, indent=2))
    print("\nNLI warning(s):")
    for issue in getattr(response, "validation_issues", []) or []:
        print(f"  - {issue.message}")

    while True:
        decision = input(
            "\nDoes the prose remain compatible with the authoritative state "
            "and requested action? [a=accept warning / r=reject scene]: "
        ).strip().casefold()
        if decision in {"a", "accept", "y", "yes"}:
            accepted = True
            label = "accepted_after_human_review"
            break
        if decision in {"r", "reject", "n", "no"}:
            accepted = False
            label = "rejected_after_human_review"
            break
        print("Enter 'a' to accept the warning or 'r' to reject the scene.")

    note = input("Reviewer note (recommended, may be blank): ").strip()
    print("-" * 76 + "\n")
    return accepted, label, note


def expected_fact_rows(
    *,
    memory: dict[str, Any],
    expected: dict[str, Any],
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, wanted: Any, actual: Any, passed: bool) -> None:
        rows.append({
            **base,
            "fact": name,
            "expected": json.dumps(wanted, ensure_ascii=False),
            "actual": json.dumps(actual, ensure_ascii=False),
            "passed": passed,
        })

    if "location" in expected:
        actual = memory.get("location")
        wanted = expected["location"]
        add("location", wanted, actual, normalise(str(actual)) == normalise(str(wanted)))

    for wanted in expected.get("inventory_contains", []):
        actual = memory.get("inventory", [])
        add(f"inventory contains {wanted}", wanted, actual, contains(actual, wanted))

    for wanted in expected.get("inventory_absent", []):
        actual = memory.get("inventory", [])
        add(f"inventory excludes {wanted}", f"not {wanted}", actual, not contains(actual, wanted))

    for character, wanted in expected.get("relationship", {}).items():
        actual = memory.get("relationships", {}).get(character)
        add(
            f"relationship {character}",
            wanted,
            actual,
            normalise(str(actual or "")) == normalise(str(wanted)),
        )

    for wanted in expected.get("goals_contains", []):
        actual = memory.get("goals", [])
        add(f"goals contains {wanted}", wanted, actual, contains(actual, wanted))

    for wanted in expected.get("goals_absent", []):
        actual = memory.get("goals", [])
        add(f"goals excludes {wanted}", f"not {wanted}", actual, not contains(actual, wanted))

    for wanted in expected.get("clues_contains", []):
        actual = memory.get("unresolved_clues", [])
        add(f"clues contains {wanted}", wanted, actual, contains(actual, wanted))

    if "threat" in expected:
        actual = memory.get("threat")
        wanted = expected["threat"]
        add("threat", wanted, actual, normalise(str(actual)) == normalise(str(wanted)))

    return rows


# ---------------------------------------------------------------------------
# Objective 1A/1B.
# ---------------------------------------------------------------------------

async def evaluate_latency(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> None:
    print("\n[O1] Warm-up generation (excluded)...")
    state = create_project("EVAL O1 Warm-up")
    try:
        response, _ = await generate(
            project_id=state["project"]["id"],
            branch_id=state["branches"][0]["id"],
            parent_id=state["nodes"][0]["id"],
            choice_id=None,
            model=args.model,
            temperature=args.temperature,
            player_input="Climb the staircase and continue exploring the tower.",
            gm_instruction=(
                "Write a normal SekAI continuation within the application's "
                "50-150 word requirement. Preserve current branch memory."
            ),
        )
        print(f"     warm-up status={response.status}")
    except Exception as error:
        print(f"     warm-up failed: {type(error).__name__}: {error}")

    # Explicitly load the NLI model outside the measured loop. O1 is therefore
    # a warm-pipeline latency measurement; startup cost should be reported
    # separately if desired, not mixed into one arbitrary timed run.
    print("[O1] Warming NLI specialist...")
    await asyncio.to_thread(get_nli_model)

    for run in range(1, args.latency_runs + 1):
        print(f"[O1] Latency run {run}/{args.latency_runs}")
        state = create_project(f"EVAL O1 Latency {run:02d}")
        started = time.perf_counter()

        try:
            response, wall = await generate(
                project_id=state["project"]["id"],
                branch_id=state["branches"][0]["id"],
                parent_id=state["nodes"][0]["id"],
                choice_id=None,
                model=args.model,
                temperature=args.temperature,
                player_input="Climb the staircase and continue exploring the tower.",
                gm_instruction=(
                    "Write a normal SekAI continuation within the application's "
                    "50-150 word requirement. Preserve current branch memory."
                ),
            )
        except Exception as error:
            wall = time.perf_counter() - started
            row = {
                "timestamp": now(),
                "run": run,
                "status": "exception",
                "returned_draft": False,
                "requires_review": False,
                "validation_pass": False,
                "requested_model": args.model,
                "accepted_model": "",
                "fallback_used": "",
                "attempts": "",
                "pipeline_elapsed_seconds": "",
                "wall_seconds": round(wall, 6),
                "prompt_tokens": "",
                "output_tokens": "",
                "tokens_per_second": "",
                "word_count": "",
                "choice_count": "",
                "semantic_check_completed": False,
                "continuity_check_completed": False,
                "continuity_checked_facts": 0,
                "continuity_max_contradiction": "",
                "continuity_issue_count": 0,
                "blocking_continuity_issue_count": 0,
                "error": f"{type(error).__name__}: {error}",
            }
            rows.append(row)
            save_csv(O1_LATENCY_CSV, rows)
            add_jsonl({"experiment": "O1_latency", **row})
            print(f"     ERROR: {row['error']}")
            continue

        metrics = response.metrics
        continuity_all, continuity_blocking = continuity_issue_count(response)
        words = len(re.findall(r"\b[\w'-]+\b", response.draft.story_text))

        row = {
            "timestamp": now(),
            "run": run,
            "status": response.status,
            "returned_draft": True,
            "requires_review": response.status == "review_required",
            "validation_pass": bool(response.validation_pass),
            "requested_model": metrics.requested_model,
            "accepted_model": metrics.model,
            "fallback_used": bool(metrics.fallback_used),
            "attempts": metrics.attempts,
            "pipeline_elapsed_seconds": metrics.elapsed_seconds,
            "wall_seconds": round(wall, 6),
            "prompt_tokens": metrics.prompt_tokens,
            "output_tokens": metrics.output_tokens,
            "tokens_per_second": metrics.tokens_per_second,
            "word_count": words,
            "choice_count": len(response.draft.choices),
            "semantic_check_completed": metrics.semantic_check_completed,
            "continuity_check_completed": metrics.continuity_check_completed,
            "continuity_checked_facts": metrics.continuity_checked_facts,
            "continuity_max_contradiction": metrics.continuity_max_contradiction,
            "continuity_issue_count": continuity_all,
            "blocking_continuity_issue_count": continuity_blocking,
            "error": "",
        }
        rows.append(row)
        save_csv(O1_LATENCY_CSV, rows)

        add_jsonl({
            "experiment": "O1_latency",
            **row,
            "draft": response.draft.model_dump(),
            "validation_issues": review_issue_rows(response),
        })


# ---------------------------------------------------------------------------
# Objectives 1 branching, 2 continuity, 3 consequences.
# ---------------------------------------------------------------------------

async def run_controlled_branch(
    *,
    pair: int,
    branch_label: str,
    turns: tuple[Turn, ...],
    project_id: str,
    root: dict[str, Any],
    source_branch_id: str,
    force_fork_first: bool,
    args: argparse.Namespace,
    fact_rows: list[dict[str, Any]],
    consequence_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, Any]],
    runtime_review_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    parent = root
    branch_id = source_branch_id
    first_fork: bool | None = None
    error_turns = 0
    critical = 0
    completed = 0
    review_required_turns = 0
    human_overrides_accepted = 0
    human_overrides_rejected = 0
    human_confirmed_continuity_errors = 0

    used = turns[:args.max_turns]

    for index, turn in enumerate(used):
        next_turn = (
            used[index + 1]
            if index + 1 < len(used)
            else None
        )

        selected_choice_id = first_choice_id(parent)

        print(
            f"[O2/O3] pair={pair}, branch={branch_label}, "
            f"turn={turn.number}/{len(used)}"
        )

        generation_started = time.perf_counter()
        try:
            response, wall = await generate(
                project_id=project_id,
                branch_id=branch_id,
                parent_id=parent["id"],
                choice_id=selected_choice_id,
                model=args.model,
                temperature=args.temperature,
                player_input="",
                gm_instruction=turn.gm_instruction,
            )
        except Exception as error:
            wall = time.perf_counter() - generation_started
            error_turns += 1
            error_text = f"{type(error).__name__}: {error}"
            add_jsonl({
                "experiment": "O2_O3_story",
                "timestamp": now(),
                "pair": pair,
                "branch": branch_label,
                "turn": turn.number,
                "action": turn.action,
                "status": "exception",
                "validation_pass": False,
                "wall_seconds": wall,
                "error": error_text,
            })
            fact_rows.append({
                "timestamp": now(),
                "pair": pair,
                "branch": branch_label,
                "turn": turn.number,
                "node_id": "",
                "fact": "generation accepted",
                "expected": "draft returned",
                "actual": "exception",
                "passed": False,
            })
            consequence_rows.append({
                "timestamp": now(),
                "pair": pair,
                "branch": branch_label,
                "turn": turn.number,
                "node_id": "",
                "selected_choice_label": turn.action,
                "expected_change": turn.expected_change,
                "actual_change": "",
                "change_expectation_met": False,
                "memory_expectation_met": False,
                "accepted": False,
                "generation_status": "exception",
                "review_override_used": False,
            })
            print(f"        STOP: generation raised {error_text}")
            break

        continuity_all, continuity_blocking = continuity_issue_count(response)
        critical += continuity_blocking

        review_override = False
        review_decision = "not_required"
        reviewer_note = ""
        can_human_review = is_human_reviewable_continuity_warning(response)

        if response.status == "review_required":
            review_required_turns += 1

            if can_human_review:
                review_override, review_decision, reviewer_note = (
                    human_review_continuity_warning(
                        args=args,
                        pair=pair,
                        branch_label=branch_label,
                        turn=turn,
                        parent=parent,
                        response=response,
                    )
                )
                if review_override:
                    human_overrides_accepted += 1
                else:
                    human_overrides_rejected += 1
                    if review_decision == "rejected_after_human_review":
                        human_confirmed_continuity_errors += 1
            else:
                review_decision = "not_human_overridable"

            runtime_review_rows.append({
                "timestamp": now(),
                "pair": pair,
                "branch": branch_label,
                "turn": turn.number,
                "parent_node_id": parent["id"],
                "selected_choice_label": turn.action,
                "generation_status": response.status,
                "human_reviewable_continuity_only": can_human_review,
                "issue_stages": "; ".join(
                    str(getattr(issue, "stage", ""))
                    for issue in response.validation_issues
                ),
                "issue_messages": " | ".join(
                    str(getattr(issue, "message", ""))
                    for issue in response.validation_issues
                ),
                "story_text": response.draft.story_text,
                "applied_delta": json.dumps(
                    response.draft.applied_delta.model_dump(),
                    ensure_ascii=False,
                ),
                "review_mode": args.review_mode,
                "decision": review_decision,
                "reviewer_note": reviewer_note,
            })
            save_csv(O2_RUNTIME_REVIEW_CSV, runtime_review_rows)

        add_jsonl({
            "experiment": "O2_O3_story",
            "timestamp": now(),
            "pair": pair,
            "branch": branch_label,
            "turn": turn.number,
            "action": turn.action,
            "status": response.status,
            "validation_pass": bool(response.validation_pass),
            "wall_seconds": wall,
            "human_reviewable_continuity_only": can_human_review,
            "review_override_used": review_override,
            "review_decision": review_decision,
            "reviewer_note": reviewer_note,
            "metrics": response.metrics.model_dump(),
            "draft": response.draft.model_dump(),
            "validation_issues": review_issue_rows(response),
        })

        accepted_without_review = (
            response.status == "accepted"
            and bool(response.validation_pass)
        )
        accepted_after_review = (
            response.status == "review_required"
            and can_human_review
            and review_override
        )

        if not (accepted_without_review or accepted_after_review):
            error_turns += 1
            fact_rows.append({
                "timestamp": now(),
                "pair": pair,
                "branch": branch_label,
                "turn": turn.number,
                "node_id": "",
                "fact": "generation accepted",
                "expected": "saveable draft",
                "actual": f"{response.status}; {review_decision}",
                "passed": False,
            })
            consequence_rows.append({
                "timestamp": now(),
                "pair": pair,
                "branch": branch_label,
                "turn": turn.number,
                "node_id": "",
                "selected_choice_label": turn.action,
                "expected_change": turn.expected_change,
                "actual_change": "",
                "change_expectation_met": False,
                "memory_expectation_met": False,
                "accepted": False,
                "generation_status": response.status,
                "review_override_used": False,
                "review_decision": review_decision,
            })
            print(
                "        STOP: non-overridable failure or human rejection recorded."
            )
            break

        reviewed_choices = choices(next_turn)

        try:
            save_result = services.save_story_node(
                SaveNodeRequest(
                    project_id=project_id,
                    source_branch_id=branch_id,
                    parent_node_id=parent["id"],
                    selected_choice_id=selected_choice_id,
                    branch_name=f"Evaluation {branch_label}",
                    force_fork=(force_fork_first and index == 0),
                    title=response.draft.title,
                    story_text=response.draft.story_text,
                    applied_delta=response.draft.applied_delta,
                    choices=reviewed_choices,
                    authoring_mode="ai",
                    interaction_type="continue",
                    interaction_text=turn.action,
                    generated_by_model=response.metrics.model,
                    generation_log_id=response.generation_log_id,
                    review_override=review_override,
                    scene_image_log_id=None,
                )
            )
        except ValueError as error:
            # The save service performs its own deterministic preflight checks.
            # A human NLI override must never bypass a real blocking state error.
            error_turns += 1
            consequence_rows.append({
                "timestamp": now(),
                "pair": pair,
                "branch": branch_label,
                "turn": turn.number,
                "node_id": "",
                "selected_choice_label": turn.action,
                "expected_change": turn.expected_change,
                "actual_change": "",
                "change_expectation_met": False,
                "memory_expectation_met": False,
                "accepted": False,
                "generation_status": response.status,
                "review_override_used": review_override,
                "review_decision": "save_preflight_rejected",
                "save_error": str(error),
            })
            add_jsonl({
                "experiment": "O2_O3_save_rejection",
                "timestamp": now(),
                "pair": pair,
                "branch": branch_label,
                "turn": turn.number,
                "error": str(error),
                "review_override_used": review_override,
            })
            print(f"        STOP: save preflight rejected draft: {error}")
            break

        if first_fork is None:
            first_fork = bool(save_result["fork_created"])

        branch_id = save_result["branch"]["id"]
        node = save_result["node"]
        memory = node["memory_snapshot"]
        completed += 1

        current_facts = expected_fact_rows(
            memory=memory,
            expected=turn.expected,
            base={
                "timestamp": now(),
                "pair": pair,
                "branch": branch_label,
                "turn": turn.number,
                "node_id": node["id"],
            },
        )
        fact_rows.extend(current_facts)

        memory_ok = all(bool(row["passed"]) for row in current_facts)
        if not memory_ok:
            error_turns += 1

        actual_change = StateDelta.model_validate(
            node.get("state_delta", {})
        ).has_meaningful_change()

        consequence_rows.append({
            "timestamp": now(),
            "pair": pair,
            "branch": branch_label,
            "turn": turn.number,
            "node_id": node["id"],
            "selected_choice_id": selected_choice_id,
            "selected_choice_label": turn.action,
            "expected_change": turn.expected_change,
            "actual_change": actual_change,
            "change_expectation_met": actual_change == turn.expected_change,
            "memory_expectation_met": memory_ok,
            "accepted": True,
            "generation_status": response.status,
            "review_override_used": review_override,
            "review_decision": review_decision,
            "state_delta": json.dumps(
                node.get("state_delta", {}),
                ensure_ascii=False,
            ),
        })

        manual_rows.append({
            "pair": pair,
            "branch": branch_label,
            "turn": turn.number,
            "node_id": node["id"],
            "selected_choice_label": turn.action,
            "generation_status": response.status,
            "review_override_used": review_override,
            "runtime_review_decision": review_decision,
            "runtime_reviewer_note": reviewer_note,
            "story_text": node["story_text"],
            "manual_critical_contradiction_0_or_1": "",
            "manual_other_continuity_error_0_or_1": "",
            "reviewer_notes": "",
        })

        parent = node

        save_csv(O2_FACT_CSV, fact_rows)
        save_csv(O2_MANUAL_CSV, manual_rows)
        save_csv(O2_RUNTIME_REVIEW_CSV, runtime_review_rows)
        save_csv(O3_CSV, consequence_rows)

    return {
        "branch": branch_label,
        "branch_id": branch_id,
        "head_node_id": parent["id"],
        "completed_turns": completed,
        "continuity_error_turns": error_turns,
        "critical_contradictions": critical,
        "review_required_turns": review_required_turns,
        "human_overrides_accepted": human_overrides_accepted,
        "human_overrides_rejected": human_overrides_rejected,
        "human_confirmed_continuity_errors": human_confirmed_continuity_errors,
        "fork_created_first_turn": first_fork,
    }


async def evaluate_pair(
    pair: int,
    args: argparse.Namespace,
    branching_rows: list[dict[str, Any]],
    fact_rows: list[dict[str, Any]],
    consequence_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, Any]],
    runtime_review_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    state = create_project(f"EVAL O2-O3 Pair {pair}")
    project = state["project"]
    main_branch = state["branches"][0]
    root = state["nodes"][0]

    # Both controlled branches intentionally begin from the same historical
    # root and the same first choice. Set it once so the source choice ID stays
    # stable when Branch B later forks from that historical root.
    root = set_controlled_choices(
        root,
        BRANCH_A[0],
    )

    branch_a = await run_controlled_branch(
        pair=pair,
        branch_label="A",
        turns=BRANCH_A,
        project_id=project["id"],
        root=root,
        source_branch_id=main_branch["id"],
        force_fork_first=False,
        args=args,
        fact_rows=fact_rows,
        consequence_rows=consequence_rows,
        manual_rows=manual_rows,
        runtime_review_rows=runtime_review_rows,
    )

    # Start again from the same historical root and force a fork.
    branch_b = await run_controlled_branch(
        pair=pair,
        branch_label="B",
        turns=BRANCH_B,
        project_id=project["id"],
        root=root,
        source_branch_id=main_branch["id"],
        force_fork_first=True,
        args=args,
        fact_rows=fact_rows,
        consequence_rows=consequence_rows,
        manual_rows=manual_rows,
        runtime_review_rows=runtime_review_rows,
    )

    compare_error = ""
    comparison: dict[str, Any] | None = None

    try:
        comparison = services.compare_branches(
            branch_a["branch_id"],
            branch_b["branch_id"],
        )
    except Exception as error:
        compare_error = str(error)

    common_id = (
        comparison["common_ancestor"]["id"]
        if comparison and comparison.get("common_ancestor")
        else None
    )

    state_after = db.get_state(project["id"])
    evidence = services.objective_evidence(project["id"])

    row = {
        "timestamp": now(),
        "pair": pair,
        "project_id": project["id"],
        "root_node_id": root["id"],
        "branch_a_id": branch_a["branch_id"],
        "branch_b_id": branch_b["branch_id"],
        "branch_count": len(state_after["branches"]),
        "two_or_more_branches": len(state_after["branches"]) >= 2,
        "common_ancestor_id": common_id,
        "shared_root_confirmed": common_id == root["id"],
        "branch_b_fork_created": branch_b["fork_created_first_turn"],
        "branch_a_completed_turns": branch_a["completed_turns"],
        "branch_b_completed_turns": branch_b["completed_turns"],
        "all_ai_nodes_have_three_choices": evidence["objective_1"][
            "all_ai_nodes_have_three_choices"
        ],
        "compare_error": compare_error,
    }
    branching_rows.append(row)
    save_csv(O1_BRANCH_CSV, branching_rows)

    add_jsonl({
        "experiment": "O1_branching_summary",
        **row,
        "objective_evidence": evidence,
        "comparison": comparison,
    })

    return {
        "project_id": project["id"],
        "branch_a": branch_a,
        "branch_b": branch_b,
        "branching": row,
    }


# ---------------------------------------------------------------------------
# Summaries.
# ---------------------------------------------------------------------------

def build_summary(
    args: argparse.Namespace,
    eval_db: Path,
    latency: list[dict[str, Any]],
    branching: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    consequences: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    returned_rows = [
        row for row in latency
        if bool(row.get("returned_draft"))
    ]
    returned_times = [
        float(row["wall_seconds"])
        for row in returned_rows
    ]
    accepted_rows = [
        row for row in returned_rows
        if row.get("status") == "accepted"
        and bool(row.get("validation_pass"))
    ]
    accepted_times = [
        float(row["wall_seconds"])
        for row in accepted_rows
    ]
    review_required_rows = [
        row for row in returned_rows
        if row.get("status") == "review_required"
    ]
    fallback_rows = [
        row for row in returned_rows
        if bool(row.get("fallback_used"))
    ]
    all_runs_returned = (
        bool(latency)
        and len(returned_rows) == len(latency)
    )

    o1 = {
        "latency_target_seconds": O1_LATENCY_TARGET,
        "latency_definition": (
            "Wall-clock time from generation request to a returned draft, "
            "including drafts that require human review."
        ),
        "measured_latency_runs": len(latency),
        "returned_draft_runs": len(returned_rows),
        "all_latency_runs_returned_draft": all_runs_returned,
        "mean_returned_seconds": (
            statistics.mean(returned_times)
            if returned_times else None
        ),
        "median_returned_seconds": (
            statistics.median(returned_times)
            if returned_times else None
        ),
        "stdev_returned_seconds": (
            statistics.stdev(returned_times)
            if len(returned_times) >= 2 else None
        ),
        "p95_returned_seconds": p95(returned_times),
        "min_returned_seconds": min(returned_times) if returned_times else None,
        "max_returned_seconds": max(returned_times) if returned_times else None,
        "accepted_without_review_runs": len(accepted_rows),
        "review_required_runs": len(review_required_rows),
        "accepted_without_review_rate": (
            len(accepted_rows) / len(returned_rows)
            if returned_rows else 0.0
        ),
        "review_required_rate": (
            len(review_required_rows) / len(returned_rows)
            if returned_rows else 0.0
        ),
        "fallback_runs": len(fallback_rows),
        "fallback_rate": (
            len(fallback_rows) / len(returned_rows)
            if returned_rows else 0.0
        ),
        "mean_accepted_seconds": (
            statistics.mean(accepted_times)
            if accepted_times else None
        ),
        "latency_target_met": (
            all_runs_returned
            and bool(returned_times)
            and statistics.mean(returned_times) < O1_LATENCY_TARGET
        ),
        "all_pairs_have_two_or_more_branches": (
            bool(branching)
            and all(row["two_or_more_branches"] for row in branching)
        ),
        "all_pairs_share_the_controlled_root": (
            bool(branching)
            and all(row["shared_root_confirmed"] for row in branching)
        ),
        "all_saved_ai_nodes_have_three_choices": (
            bool(branching)
            and all(row["all_ai_nodes_have_three_choices"] for row in branching)
        ),
    }

    scored_facts = [
        row for row in facts
        if row.get("fact") != "generation accepted"
    ]
    passed = sum(bool(row.get("passed")) for row in scored_facts)
    fact_rate = passed / len(scored_facts) if scored_facts else 0.0

    branch_errors: dict[str, int] = {}
    critical_total = 0
    for pair_index, pair in enumerate(pairs, start=1):
        for key in ("branch_a", "branch_b"):
            branch = pair[key]
            branch_errors[
                f"pair_{pair_index}_{branch['branch']}"
            ] = branch["continuity_error_turns"]
            critical_total += branch["critical_contradictions"]

    runtime_review_required = sum(
        pair[key]["review_required_turns"]
        for pair in pairs
        for key in ("branch_a", "branch_b")
    )
    human_overrides_accepted = sum(
        pair[key]["human_overrides_accepted"]
        for pair in pairs
        for key in ("branch_a", "branch_b")
    )
    human_overrides_rejected = sum(
        pair[key]["human_overrides_rejected"]
        for pair in pairs
        for key in ("branch_a", "branch_b")
    )
    human_confirmed_continuity_errors = sum(
        pair[key]["human_confirmed_continuity_errors"]
        for pair in pairs
        for key in ("branch_a", "branch_b")
    )

    planned_branch_turns = (
        args.pairs
        * 2
        * args.max_turns
    )
    completed_branch_turns = sum(
        pair["branch_a"]["completed_turns"]
        + pair["branch_b"]["completed_turns"]
        for pair in pairs
    )
    coverage_rate = (
        completed_branch_turns / planned_branch_turns
        if planned_branch_turns
        else 0.0
    )
    coverage_met = (
        completed_branch_turns
        == planned_branch_turns
    )

    o2 = {
        "fact_preservation_target": O2_FACT_TARGET,
        "critical_contradiction_target": O2_CRITICAL_TARGET,
        "max_continuity_errors_per_branch_target": O2_ERRORS_PER_BRANCH_TARGET,
        "fact_checks": len(scored_facts),
        "fact_checks_passed": passed,
        "fact_preservation_rate": fact_rate,
        "planned_branch_turns": planned_branch_turns,
        "completed_branch_turns": completed_branch_turns,
        "coverage_rate": coverage_rate,
        "coverage_target_met": coverage_met,
        "automatically_detected_critical_contradictions": critical_total,
        "automatic_error_turns_by_branch": branch_errors,
        "fact_preservation_target_met": (
            coverage_met
            and fact_rate >= O2_FACT_TARGET
        ),
        "automatic_critical_contradiction_target_met": critical_total == 0,
        "manual_review_complete": False,
        "manual_scenes_reviewed": 0,
        "manual_critical_contradictions": None,
        "manual_continuity_error_turns_by_branch": {},
        "critical_contradiction_target_met": None,
        "continuity_error_target_met": None,
        "objective_2_target_met": None,
        "manual_semantic_review_still_required": True,
        "manual_review_file": str(O2_MANUAL_CSV),
        "generation_review_file": str(O2_RUNTIME_REVIEW_CSV),
        "review_mode": args.review_mode,
        "runtime_review_required_turns": runtime_review_required,
        "human_overrides_accepted": human_overrides_accepted,
        "human_overrides_rejected": human_overrides_rejected,
        "human_confirmed_continuity_errors": human_confirmed_continuity_errors,
    }

    accepted = [row for row in consequences if row.get("accepted")]
    changed = sum(bool(row.get("actual_change")) for row in accepted)
    change_ratio = changed / len(accepted) if accepted else 0.0

    required = [row for row in accepted if row.get("expected_change")]
    required_correct = sum(
        bool(row.get("actual_change"))
        and bool(row.get("memory_expectation_met"))
        for row in required
    )
    required_rate = (
        required_correct / len(required)
        if required else 0.0
    )

    no_change = [row for row in accepted if not row.get("expected_change")]
    no_change_correct = sum(
        not bool(row.get("actual_change"))
        for row in no_change
    )
    no_change_rate = (
        no_change_correct / len(no_change)
        if no_change else None
    )

    planned_expected_change_cases = args.pairs * (
        sum(
            turn.expected_change
            for turn in BRANCH_A[:args.max_turns]
        )
        +
        sum(
            turn.expected_change
            for turn in BRANCH_B[:args.max_turns]
        )
    )
    planned_consequence_cases = (
        args.pairs
        * 2
        * args.max_turns
    )
    expected_change_coverage_met = (
        len(required)
        == planned_expected_change_cases
    )
    o3_coverage_met = (
        len(accepted)
        == planned_consequence_cases
    )

    o3 = {
        "state_change_target": O3_CHANGE_TARGET,
        "accepted_selected_choice_nodes": len(accepted),
        "selected_choice_nodes_with_state_change": changed,
        "state_changing_choice_ratio": change_ratio,
        "planned_consequence_cases": planned_consequence_cases,
        "planned_expected_change_cases": planned_expected_change_cases,
        "expected_change_coverage_met": expected_change_coverage_met,
        "coverage_target_met": o3_coverage_met,
        "target_met": (
            o3_coverage_met
            and change_ratio >= O3_CHANGE_TARGET
        ),
        "expected_change_cases": len(required),
        "expected_change_cases_correct": required_correct,
        "expected_change_correctness_rate": required_rate,
        "no_change_control_cases": len(no_change),
        "no_change_control_accuracy": no_change_rate,
        "note": (
            "The original >=80% ratio is retained as the structural proxy. "
            "Expected-change correctness and no-change controls are reported "
            "separately so arbitrary state mutation is not rewarded."
        ),
    }

    return {
        "generated_at": now(),
        "evaluation_database": str(eval_db),
        "writer_model": args.model,
        "temperature": args.temperature,
        "evaluation_mode": bool(getattr(ai_service, "EVALUATION_MODE", False)),
        "objective_1": o1,
        "objective_2": o2,
        "objective_3": o3,
    }


def summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for objective in ("objective_1", "objective_2", "objective_3"):
        for metric, value in summary[objective].items():
            rows.append({
                "objective": objective,
                "metric": metric,
                "value": (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                ),
            })
    return rows


def _manual_binary(value: str, *, field: str, row_number: int) -> int | None:
    clean = str(value or "").strip()
    if clean == "":
        return None
    if clean not in {"0", "1"}:
        raise SystemExit(
            f"Invalid value '{clean}' in {field} at CSV row {row_number}. "
            "Use only 0 or 1."
        )
    return int(clean)


def finalize_manual_review() -> None:
    """
    Merge the completed human prose labels into objective_summary.json/csv.

    This command performs no generation. It is intended to be run only after the
    evaluator has finished and every row in objective2_manual_story_review.csv has
    been independently labelled 0/1.
    """
    if not SUMMARY_JSON.exists():
        raise SystemExit(
            f"Cannot finalise manual review because {SUMMARY_JSON} does not exist."
        )
    if not O2_MANUAL_CSV.exists():
        raise SystemExit(
            f"Cannot finalise manual review because {O2_MANUAL_CSV} does not exist."
        )

    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    with O2_MANUAL_CSV.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("The manual review CSV contains no saved story scenes.")

    pending = 0
    critical_total = 0
    error_turns_by_branch: dict[str, int] = {}

    for row_number, row in enumerate(rows, start=2):
        critical = _manual_binary(
            row.get("manual_critical_contradiction_0_or_1", ""),
            field="manual_critical_contradiction_0_or_1",
            row_number=row_number,
        )
        other = _manual_binary(
            row.get("manual_other_continuity_error_0_or_1", ""),
            field="manual_other_continuity_error_0_or_1",
            row_number=row_number,
        )

        if critical is None or other is None:
            pending += 1
            continue

        critical_total += critical
        key = f"pair_{row.get('pair')}_{row.get('branch')}"
        error_turns_by_branch.setdefault(key, 0)
        if critical or other:
            error_turns_by_branch[key] += 1

    review_complete = pending == 0
    o2 = summary["objective_2"]

    # Ensure branches with zero human-labelled errors are present in the table.
    automatic_branches = o2.get("automatic_error_turns_by_branch", {})
    for key in automatic_branches:
        error_turns_by_branch.setdefault(key, 0)

    critical_target_met = (
        review_complete
        and critical_total <= O2_CRITICAL_TARGET
    )
    error_target_met = (
        review_complete
        and bool(error_turns_by_branch)
        and all(
            value <= O2_ERRORS_PER_BRANCH_TARGET
            for value in error_turns_by_branch.values()
        )
    )

    o2.update({
        "manual_review_complete": review_complete,
        "manual_scenes_reviewed": len(rows) - pending,
        "manual_scenes_pending": pending,
        "manual_critical_contradictions": critical_total if review_complete else None,
        "manual_continuity_error_turns_by_branch": (
            error_turns_by_branch if review_complete else {}
        ),
        "critical_contradiction_target_met": (
            critical_target_met if review_complete else None
        ),
        "continuity_error_target_met": (
            error_target_met if review_complete else None
        ),
        "objective_2_target_met": (
            bool(o2.get("fact_preservation_target_met"))
            and critical_target_met
            and error_target_met
            if review_complete
            else None
        ),
        "manual_semantic_review_still_required": not review_complete,
        "manual_review_finalized_at": now() if review_complete else None,
    })

    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_csv(SUMMARY_CSV, summary_rows(summary))

    print("=" * 76)
    print("OBJECTIVE 2 MANUAL REVIEW FINALISATION")
    print("=" * 76)
    print(f"Scenes reviewed          : {len(rows) - pending}/{len(rows)}")
    print(f"Review complete          : {review_complete}")
    if review_complete:
        print(f"Critical contradictions  : {critical_total}")
        print(f"Errors by branch          : {error_turns_by_branch}")
        print(f"Critical target met       : {critical_target_met}")
        print(f"<=2 errors/branch met     : {error_target_met}")
        print(f"Objective 2 target met    : {o2['objective_2_target_met']}")
    else:
        print(f"Rows still requiring 0/1 labels: {pending}")
        print("Complete every manual review row and run this command again.")


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    eval_db = Path(args.db).resolve()
    eval_db.parent.mkdir(parents=True, exist_ok=True)

    if eval_db == Path(db.DB_PATH).resolve():
        raise SystemExit(
            "Refusing to use the current development database. "
            "Use a separate --db path for formal evaluation."
        )

    if args.reset_db:
        for candidate in (
            eval_db,
            Path(f"{eval_db}-wal"),
            Path(f"{eval_db}-shm"),
        ):
            if candidate.exists():
                candidate.unlink()

    # Redirect the real backend to the clean evaluation database.
    db.DB_PATH = eval_db

    # Formal evaluation should fail visibly if semantic retrieval is unavailable.
    if hasattr(ai_service, "EVALUATION_MODE"):
        ai_service.EVALUATION_MODE = True

    db.init_db()

    available = installed_models()
    if args.model not in available:
        print("Installed Ollama models:")
        for name in sorted(available):
            print(f"  - {name}")
        raise SystemExit(
            f"\nModel '{args.model}' is not installed. "
            "Use the exact tag shown by 'ollama list'."
        )

    if args.reset_db:
        for path in (
            O1_LATENCY_CSV,
            O1_BRANCH_CSV,
            O2_FACT_CSV,
            O2_MANUAL_CSV,
            O2_RUNTIME_REVIEW_CSV,
            O3_CSV,
            RAW_JSONL,
            SUMMARY_JSON,
            SUMMARY_CSV,
        ):
            if path.exists():
                path.unlink()

    latency: list[dict[str, Any]] = []
    branching: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    consequences: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    runtime_reviews: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []

    print("=" * 76)
    print("SekAI FORMAL OBJECTIVE EVALUATION")
    print("=" * 76)
    print(f"Evaluation DB : {eval_db}")
    print(f"Model         : {args.model}")
    print(f"Temperature   : {args.temperature}")
    print(f"Latency runs  : {args.latency_runs}")
    print(f"Branch pairs  : {args.pairs}")
    print(f"Turns/branch  : {args.max_turns}")
    print(f"Review mode   : {args.review_mode}")

    await evaluate_latency(args, latency)

    print("\n[O1/O2/O3] Controlled branch experiments")
    for pair in range(1, args.pairs + 1):
        pair_results.append(
            await evaluate_pair(
                pair,
                args,
                branching,
                facts,
                consequences,
                manual,
                runtime_reviews,
            )
        )

    save_csv(O1_LATENCY_CSV, latency)
    save_csv(O1_BRANCH_CSV, branching)
    save_csv(O2_FACT_CSV, facts)
    save_csv(O2_MANUAL_CSV, manual)
    save_csv(O2_RUNTIME_REVIEW_CSV, runtime_reviews)
    save_csv(O3_CSV, consequences)

    summary = build_summary(
        args,
        eval_db,
        latency,
        branching,
        facts,
        consequences,
        pair_results,
    )

    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_csv(SUMMARY_CSV, summary_rows(summary))

    print("\n" + "=" * 76)
    print("RESULT SUMMARY")
    print("=" * 76)

    o1 = summary["objective_1"]
    print("\nObjective 1")
    print(f"  mean returned latency : {o1['mean_returned_seconds']}")
    print(f"  median returned       : {o1['median_returned_seconds']}")
    print(f"  p95 returned          : {o1['p95_returned_seconds']}")
    print(
        f"  accepted / review     : "
        f"{o1['accepted_without_review_runs']} / "
        f"{o1['review_required_runs']}"
    )
    print(f"  fallback rate         : {o1['fallback_rate']:.3f}")
    print(f"  <15 s target met      : {o1['latency_target_met']}")
    print(f"  shared-root branches  : {o1['all_pairs_share_the_controlled_root']}")
    print(f"  3 choices per AI node : {o1['all_saved_ai_nodes_have_three_choices']}")

    o2 = summary["objective_2"]
    print("\nObjective 2")
    print(f"  fact preservation     : {o2['fact_preservation_rate']:.3f}")
    print(
        f"  branch coverage       : "
        f"{o2['completed_branch_turns']}/"
        f"{o2['planned_branch_turns']}"
    )
    print(f"  coverage complete     : {o2['coverage_target_met']}")
    print(f"  >=85% fact target     : {o2['fact_preservation_target_met']}")
    print(
        "  manual semantic result: PENDING - complete "
        f"{O2_MANUAL_CSV.name} and run --finalize-manual-review"
    )

    o3 = summary["objective_3"]
    print("\nObjective 3")
    print(f"  state-change ratio    : {o3['state_changing_choice_ratio']:.3f}")
    print(
        f"  consequence coverage : "
        f"{o3['accepted_selected_choice_nodes']}/"
        f"{o3['planned_consequence_cases']}"
    )
    print(
        f"  expected-change cases: "
        f"{o3['expected_change_cases']}/"
        f"{o3['planned_expected_change_cases']}"
    )
    print(f"  O3 coverage complete  : {o3['coverage_target_met']}")
    print(f"  >=80% target met      : {o3['target_met']}")
    print(f"  expected-change score : {o3['expected_change_correctness_rate']:.3f}")
    print(f"  no-change accuracy    : {o3['no_change_control_accuracy']}")

    print("\nFiles:")
    for path in (
        O1_LATENCY_CSV,
        O1_BRANCH_CSV,
        O2_FACT_CSV,
        O2_MANUAL_CSV,
        O2_RUNTIME_REVIEW_CSV,
        O3_CSV,
        RAW_JSONL,
        SUMMARY_JSON,
        SUMMARY_CSV,
        eval_db,
    ):
        print(f"  {path}")

    print(
        "\nObjective 2 finalisation: open "
        f"{O2_MANUAL_CSV.name}, read every saved story_text, fill both 0/1 "
        "error columns plus reviewer_notes, then run:\n"
        "  python .\\evaluation\\run_objective_evaluation.py "
        "--finalize-manual-review"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled formal evaluator for SekAI Objectives 1-3."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--latency-runs", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=7,
        choices=range(1, 8),
        metavar="1-7",
    )
    parser.add_argument(
        "--review-mode",
        choices=("interactive", "strict"),
        default="interactive",
        help=(
            "interactive: allow explicit human adjudication of NLI-only "
            "continuity warnings; strict: reject every review_required draft."
        ),
    )
    parser.add_argument(
        "--finalize-manual-review",
        action="store_true",
        help=(
            "Do not generate stories. Read the completed Objective 2 manual "
            "review CSV and merge its labels into the summary files."
        ),
    )
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--reset-db", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.finalize_manual_review:
        finalize_manual_review()
        return
    if args.latency_runs < 1:
        raise SystemExit("--latency-runs must be >= 1.")
    if args.pairs < 1:
        raise SystemExit("--pairs must be >= 1.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
