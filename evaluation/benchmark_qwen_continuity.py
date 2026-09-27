from __future__ import annotations

"""
Controlled SekAI writer/continuity benchmark.

Purpose
-------
This benchmark evaluates three progressively more application-realistic stages
without using SekAI's cross-model fallback:

Experiment 2A - structured writer compatibility
    Qwen + the real SekAI system prompt + Ollama JSON schema.
    Measures schema validity, 50-150 word compliance, three-choice validity,
    StateDelta validity/grounding, and simple instruction-adherence labels.

Experiment 2B - continuity compatibility
    Adds the real deterministic validation, lexical repetition check,
    EmbeddingGemma semantic retrieval/repetition check, StateDelta application,
    and DeBERTa-v3 NLI review.

Experiment 2C - retry behaviour
    Gives the SAME quantized writer at most two attempts using the same
    failure-specific retry temperatures/feedback used by SekAI. Cross-model
    fallback is deliberately disabled so one quantization cannot be rescued by
    another model.

Run from the SekAI repository root, for example:

    .\\.venv\\Scripts\\python.exe .\\evaluation\\benchmark_qwen_continuity.py

Useful shorter runs:

    .\\.venv\\Scripts\\python.exe .\\evaluation\\benchmark_qwen_continuity.py --seeds 101

    .\\.venv\\Scripts\\python.exe .\\evaluation\\benchmark_qwen_continuity.py \
        --cases location_change inventory_add no_change --seeds 101 202

The benchmark expects Ollama, EmbeddingGemma, and the local DeBERTa NLI model
used by SekAI to be installed. It does not start FastAPI and does not modify the
SekAI database.
"""

import argparse
import asyncio
import csv
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Import the real SekAI logic from the repository rather than copying it.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from backend import ai_service
    from backend.continuity_service import (
        ContinuityModelUnavailableError,
        review_story_continuity,
    )
    from backend.schemas import AIGeneratedDraft, StoryMemory
    from backend.similarity import (
        retrieve_relevant_nodes,
        semantic_repetition_reason,
    )
    from backend.state import apply_state_delta
    from backend.validation_service import (
        blocking_issues,
        clean_redundant_delta,
        format_issues_for_retry,
        story_word_count,
        validate_reviewed_draft,
    )
except ImportError as error:
    raise SystemExit(
        "Could not import the SekAI backend. Run this file from the SekAI "
        "repository and keep it inside evaluation/. Underlying error: "
        f"{error}"
    ) from error


# ---------------------------------------------------------------------------
# Experiment configuration.
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_0",
    "qwen2.5:1.5b-instruct-q3_K_M",
]

DEFAULT_SEEDS = [101, 202, 303]
MAX_ATTEMPTS = 2
BASE_TEMPERATURE = 0.5
NUM_CTX = 4096
NUM_PREDICT = 600
KEEP_ALIVE = "30m"
OLLAMA_TIMEOUT_SECONDS = 180.0

RESULT_DIR = Path(__file__).resolve().parent / "results"
ATTEMPT_CSV = RESULT_DIR / "qwen_continuity_attempts.csv"
SUMMARY_CSV = RESULT_DIR / "qwen_continuity_summary.csv"
OUTPUT_JSONL = RESULT_DIR / "qwen_continuity_outputs.jsonl"

WORLD_SUMMARY = (
    "A ruined watchtower stands above an old mountain road. Its beacon has "
    "started glowing again after decades of silence. The tower contains old "
    "mechanisms, abandoned rooms and signs that something has recently moved "
    "through the upper floors."
)
MAIN_OBJECTIVE = "Reach the beacon chamber and discover why the tower beacon is active."


@dataclass(frozen=True)
class ContinuityCase:
    case_id: str
    description: str
    current_title: str
    current_story: str
    parent_memory: dict[str, Any]
    interaction_instruction: str
    gm_instruction: str
    expected: dict[str, Any]
    required_story_terms: tuple[tuple[str, ...], ...] = ()
    prior_nodes: tuple[dict[str, Any], ...] = ()
    retrieval_query: str = ""


BASE_PRIOR_NODES: tuple[dict[str, Any], ...] = (
    {
        "id": "history_road",
        "title": "The Road to the Tower",
        "story_text": (
            "The traveller followed a rain-soaked mountain road toward the ruined "
            "tower. A merchant warned that the old beacon had begun shining blue at "
            "night, although nobody had tended it for years."
        ),
        "source_choice_label": "Follow the mountain road",
    },
    {
        "id": "history_gate",
        "title": "At the Broken Gate",
        "story_text": (
            "At the broken outer gate the traveller found an iron key beside an "
            "abandoned pack and kept it. Beyond the gate, a narrow passage led into "
            "the tower entrance hall."
        ),
        "source_choice_label": "Search the broken gate",
    },
)


TEST_CASES: tuple[ContinuityCase, ...] = (
    ContinuityCase(
        case_id="location_change",
        description="Move from the Entrance Hall to the Second-Floor Landing.",
        current_title="The Entrance Hall",
        current_story=(
            "The traveller stands inside the tower entrance hall with a lit torch. "
            "Guard Harlan remains beside a cold brazier while a spiral staircase "
            "climbs into darkness. Scraping sounds echo from somewhere above."
        ),
        parent_memory={
            "location": "Entrance Hall",
            "active_characters": ["Guard Harlan"],
            "inventory": ["Torch", "Iron Key"],
            "relationships": {"Guard Harlan": "Wary"},
            "goals": ["Reach the beacon chamber"],
            "unresolved_clues": ["Why the beacon is glowing"],
            "decisions": [],
            "threat": "Medium",
        },
        interaction_instruction=(
            "CONTINUE: The traveller climbs the spiral staircase and reaches the "
            "Second-Floor Landing."
        ),
        gm_instruction=(
            "This scene must visibly show the traveller arriving at the location "
            "named exactly 'Second-Floor Landing'. Record that location change in "
            "applied_delta. Do not add or remove inventory."
        ),
        expected={"location": "Second-Floor Landing"},
        required_story_terms=(("Second-Floor Landing", "second-floor landing", "second floor landing"),),
        prior_nodes=BASE_PRIOR_NODES,
        retrieval_query="climb the staircase to the second floor of the tower",
    ),
    ContinuityCase(
        case_id="inventory_add",
        description="Acquire a Silver Key that was not previously owned.",
        current_title="A Locked Alcove",
        current_story=(
            "On the second floor, the traveller discovers a narrow alcove behind a "
            "cracked tapestry. A small metal box rests on a stone shelf. The box is "
            "unlocked, but its contents have not yet been examined."
        ),
        parent_memory={
            "location": "Second-Floor Landing",
            "active_characters": [],
            "inventory": ["Torch", "Iron Key"],
            "relationships": {"Guard Harlan": "Wary"},
            "goals": ["Reach the beacon chamber"],
            "unresolved_clues": ["Why the beacon is glowing"],
            "decisions": [],
            "threat": "Medium",
        },
        interaction_instruction=(
            "DO: Open the metal box and take the Silver Key found inside."
        ),
        gm_instruction=(
            "The item must be named exactly 'Silver Key'. The story must explicitly "
            "show the traveller obtaining it, and applied_delta must add 'Silver Key' "
            "to inventory. Do not remove the Iron Key."
        ),
        expected={
            "inventory_contains": ["Silver Key", "Iron Key"],
        },
        required_story_terms=(("Silver Key", "silver key"),),
        prior_nodes=BASE_PRIOR_NODES,
        retrieval_query="find and take the Silver Key in the tower",
    ),
    ContinuityCase(
        case_id="inventory_remove",
        description="Remove the Iron Key after it is destroyed.",
        current_title="The Rusted Door",
        current_story=(
            "A rusted iron door blocks the staircase to the third floor. The lock is "
            "old and stiff. The traveller still carries the Iron Key found outside "
            "the tower and a lit Torch."
        ),
        parent_memory={
            "location": "Second-Floor Landing",
            "active_characters": [],
            "inventory": ["Torch", "Iron Key"],
            "relationships": {"Guard Harlan": "Wary"},
            "goals": ["Reach the beacon chamber"],
            "unresolved_clues": ["Why the beacon is glowing"],
            "decisions": [],
            "threat": "Medium",
        },
        interaction_instruction=(
            "DO: Use the Iron Key on the rusted lock. The key snaps inside the lock "
            "and can no longer be carried or used."
        ),
        gm_instruction=(
            "The story must explicitly show the 'Iron Key' breaking or becoming lost, "
            "and applied_delta must remove 'Iron Key' from inventory. Keep the Torch."
        ),
        expected={
            "inventory_absent": ["Iron Key"],
            "inventory_contains": ["Torch"],
        },
        required_story_terms=(("Iron Key", "iron key"),),
        prior_nodes=BASE_PRIOR_NODES,
        retrieval_query="use the Iron Key on the rusted door",
    ),
    ContinuityCase(
        case_id="relationship_change",
        description="Change Guard Harlan's relationship from Wary to Trusting.",
        current_title="Harlan in Danger",
        current_story=(
            "Guard Harlan has followed the traveller upstairs despite his distrust. "
            "Loose masonry suddenly collapses above him, leaving him exposed beneath "
            "a falling timber while the traveller is close enough to intervene."
        ),
        parent_memory={
            "location": "Second-Floor Landing",
            "active_characters": ["Guard Harlan"],
            "inventory": ["Torch", "Iron Key"],
            "relationships": {"Guard Harlan": "Wary"},
            "goals": ["Reach the beacon chamber"],
            "unresolved_clues": ["Why the beacon is glowing"],
            "decisions": [],
            "threat": "Medium",
        },
        interaction_instruction=(
            "DO: Pull Guard Harlan out of danger before the timber falls."
        ),
        gm_instruction=(
            "Harlan must recognise that the traveller saved him. Change the relationship "
            "with 'Guard Harlan' to exactly 'Trusting' in applied_delta and visibly "
            "justify that change in story_text."
        ),
        expected={"relationships": {"Guard Harlan": "Trusting"}},
        required_story_terms=(("Harlan", "Guard Harlan"),),
        prior_nodes=BASE_PRIOR_NODES,
        retrieval_query="save Guard Harlan from danger and gain his trust",
    ),
    ContinuityCase(
        case_id="goal_complete",
        description="Complete the active goal after reaching the beacon chamber.",
        current_title="The Final Stair",
        current_story=(
            "The traveller reaches the final staircase beneath the tower roof. A blue "
            "glow spills through a doorway above. The active objective remains to "
            "Reach the beacon chamber."
        ),
        parent_memory={
            "location": "Upper Stairwell",
            "active_characters": [],
            "inventory": ["Torch", "Iron Key"],
            "relationships": {"Guard Harlan": "Wary"},
            "goals": ["Reach the beacon chamber"],
            "unresolved_clues": ["Why the beacon is glowing"],
            "decisions": [],
            "threat": "Medium",
        },
        interaction_instruction=(
            "CONTINUE: Climb through the doorway and enter the Beacon Chamber."
        ),
        gm_instruction=(
            "The traveller must reach the location named exactly 'Beacon Chamber'. "
            "Because the goal 'Reach the beacon chamber' is achieved in this scene, "
            "applied_delta must complete that exact goal."
        ),
        expected={
            "location": "Beacon Chamber",
            "goals_absent": ["Reach the beacon chamber"],
        },
        required_story_terms=(("Beacon Chamber", "beacon chamber"),),
        prior_nodes=BASE_PRIOR_NODES,
        retrieval_query="reach the beacon chamber at the top of the tower",
    ),
    ContinuityCase(
        case_id="threat_change",
        description="Escalate the threat from Medium to High after an attack begins.",
        current_title="The Silent Mechanism",
        current_story=(
            "Inside the upper mechanism room, the traveller hears metal scraping behind "
            "a bank of gears. Nothing has attacked yet, but the sound is moving closer."
        ),
        parent_memory={
            "location": "Mechanism Room",
            "active_characters": [],
            "inventory": ["Torch", "Iron Key"],
            "relationships": {"Guard Harlan": "Wary"},
            "goals": ["Reach the beacon chamber"],
            "unresolved_clues": ["Why the beacon is glowing"],
            "decisions": [],
            "threat": "Medium",
        },
        interaction_instruction=(
            "CONTINUE: A tower guardian bursts from behind the gears and attacks the "
            "traveller."
        ),
        gm_instruction=(
            "The attack must happen in this scene and must meaningfully escalate danger. "
            "Set threat to exactly 'High' in applied_delta."
        ),
        expected={"threat": "High"},
        required_story_terms=(("guardian", "tower guardian"), ("attack", "attacks", "strikes", "lunges")),
        prior_nodes=BASE_PRIOR_NODES,
        retrieval_query="tower guardian attacks in the mechanism room",
    ),
    ContinuityCase(
        case_id="no_change",
        description="Conversation-only scene where no tracked state should change.",
        current_title="Questioning Harlan",
        current_story=(
            "The traveller pauses beside Guard Harlan in the entrance hall. Harlan is "
            "wary but willing to answer one question. No one moves, gives away an item, "
            "receives an item, completes a goal or faces a new danger."
        ),
        parent_memory={
            "location": "Entrance Hall",
            "active_characters": ["Guard Harlan"],
            "inventory": ["Torch", "Iron Key"],
            "relationships": {"Guard Harlan": "Wary"},
            "goals": ["Reach the beacon chamber"],
            "unresolved_clues": ["Why the beacon is glowing"],
            "decisions": [],
            "threat": "Medium",
        },
        interaction_instruction=(
            "ASK: Ask Guard Harlan whether he heard the beacon activate last night."
        ),
        gm_instruction=(
            "Harlan should answer the question, but no tracked story state changes. "
            "Keep location, inventory, relationships, goals and threat unchanged and "
            "return an empty applied_delta."
        ),
        expected={"delta_empty": True},
        required_story_terms=(("Harlan", "Guard Harlan"), ("beacon",),),
        prior_nodes=BASE_PRIOR_NODES,
        retrieval_query="ask Guard Harlan about the beacon activating",
    ),
)

CASE_BY_ID = {case.case_id: case for case in TEST_CASES}


# ---------------------------------------------------------------------------
# Utility helpers.
# ---------------------------------------------------------------------------


def normalise(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", value.casefold()))


def contains_normalised(values: Iterable[str], expected: str) -> bool:
    key = normalise(expected)
    return any(normalise(value) == key for value in values)


def relationship_value(memory: StoryMemory, character: str) -> str | None:
    target = normalise(character)
    for name, status in memory.relationships.items():
        if normalise(name) == target:
            return status
    return None


def delta_is_empty(draft: AIGeneratedDraft) -> bool:
    return not draft.applied_delta.has_meaningful_change()


def ns_to_seconds(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000


def extract_ollama_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    eval_seconds = ns_to_seconds(raw.get("eval_duration"))
    output_tokens = raw.get("eval_count")
    tokens_per_second = None
    if eval_seconds and output_tokens is not None and eval_seconds > 0:
        tokens_per_second = float(output_tokens) / eval_seconds

    return {
        "ollama_total_seconds": ns_to_seconds(raw.get("total_duration")),
        "load_seconds": ns_to_seconds(raw.get("load_duration")),
        "prompt_eval_seconds": ns_to_seconds(raw.get("prompt_eval_duration")),
        "generation_seconds": eval_seconds,
        "prompt_tokens": raw.get("prompt_eval_count"),
        "output_tokens": output_tokens,
        "tokens_per_second": tokens_per_second,
    }


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


def requested_models_exist(models: list[str]) -> None:
    available = installed_models()
    missing = [model for model in models if model not in available]
    if missing:
        joined = "\n".join(f"  - {model}" for model in missing)
        raise SystemExit(
            "The following benchmark writer models are not installed in Ollama:\n"
            f"{joined}\n\nRun 'ollama list' and pass the exact installed tags with --models."
        )


def build_reference_nodes(case: ContinuityCase) -> list[dict[str, Any]]:
    current = {
        "id": f"current_{case.case_id}",
        "title": case.current_title,
        "story_text": case.current_story,
        "source_choice_label": None,
    }
    return [*case.prior_nodes, current]


async def build_controlled_context(case: ContinuityCase) -> tuple[dict[str, Any], float]:
    reference_nodes = build_reference_nodes(case)

    retrieval_started = time.perf_counter()
    retrieved = await retrieve_relevant_nodes(
        case.retrieval_query or case.interaction_instruction,
        list(case.prior_nodes),
        top_k=2,
    )
    retrieval_seconds = time.perf_counter() - retrieval_started

    recent_nodes = reference_nodes[-3:]
    forbidden_titles = [node["title"] for node in reference_nodes]
    forbidden_openings = [
        " ".join(str(node["story_text"]).split()[:12])
        for node in reference_nodes
    ]

    context = {
        "project": {
            "genre": "Fantasy",
            "tone": "Mysterious adventure",
            "main_objective": MAIN_OBJECTIVE,
            "world_summary": WORLD_SUMMARY,
        },
        "selected_node": {
            "id": f"current_{case.case_id}",
            "title": case.current_title,
            "ending_excerpt": case.current_story,
            "memory_snapshot": case.parent_memory,
        },
        "recent_story_path": [
            {
                "id": node["id"],
                "title": node["title"],
                "ending_excerpt": node["story_text"],
                "source_choice_label": node.get("source_choice_label"),
            }
            for node in recent_nodes
        ],
        "retrieved_branch_context": [
            {
                "title": node["title"],
                "excerpt": node["story_text"],
                "similarity": round(float(node["retrieval_score"]), 4),
            }
            for node in retrieved
        ],
        "story_element_policy": {"mode": "off", "elements": []},
        "interaction": {
            "type": "continue",
            "player_input": case.interaction_instruction,
            "instruction": case.interaction_instruction,
        },
        "game_master_instruction": case.gm_instruction,
        "novelty_constraints": {
            "do_not_reuse_titles": forbidden_titles,
            "do_not_reuse_openings": forbidden_openings,
        },
    }

    return context, retrieval_seconds


def instruction_adherence(story_text: str, case: ContinuityCase) -> tuple[int, int, list[str]]:
    """
    Simple pre-labelled lexical proxy for mandatory-event coverage.

    This is deliberately transparent and conservative. It is useful for model comparison,
    but it is NOT a substitute for human narrative-quality judgement.
    """
    if not case.required_story_terms:
        return 0, 0, []

    story = normalise(story_text)
    passed = 0
    details: list[str] = []

    for alternatives in case.required_story_terms:
        hit = any(normalise(term) in story for term in alternatives)
        if hit:
            passed += 1
        details.append(f"{'PASS' if hit else 'FAIL'}: one of {alternatives}")

    return passed, len(case.required_story_terms), details


def labelled_continuity(
    memory: StoryMemory,
    draft: AIGeneratedDraft,
    case: ContinuityCase,
) -> tuple[int, int, list[str]]:
    """Compare proposed child state with hand-labelled expected facts for each case."""
    expected = case.expected
    passed = 0
    total = 0
    details: list[str] = []

    def record(ok: bool, label: str, actual: Any) -> None:
        nonlocal passed, total
        total += 1
        if ok:
            passed += 1
        details.append(f"{'PASS' if ok else 'FAIL'}: {label}; actual={actual!r}")

    if "location" in expected:
        target = str(expected["location"])
        record(
            normalise(memory.location) == normalise(target),
            f"location == {target!r}",
            memory.location,
        )

    for item in expected.get("inventory_contains", []):
        record(
            contains_normalised(memory.inventory, item),
            f"inventory contains {item!r}",
            memory.inventory,
        )

    for item in expected.get("inventory_absent", []):
        record(
            not contains_normalised(memory.inventory, item),
            f"inventory does not contain {item!r}",
            memory.inventory,
        )

    for character, status in expected.get("relationships", {}).items():
        actual = relationship_value(memory, character)
        record(
            actual is not None and normalise(actual) == normalise(status),
            f"relationship[{character!r}] == {status!r}",
            actual,
        )

    for goal in expected.get("goals_absent", []):
        record(
            not contains_normalised(memory.goals, goal),
            f"goal {goal!r} is complete/absent",
            memory.goals,
        )

    if "threat" in expected:
        target = str(expected["threat"])
        record(
            normalise(memory.threat) == normalise(target),
            f"threat == {target!r}",
            memory.threat,
        )

    if expected.get("delta_empty") is True:
        record(
            delta_is_empty(draft),
            "applied_delta is empty",
            draft.applied_delta.model_dump(),
        )

    return passed, total, details


def retry_temperature(base_temperature: float, retry_kind: str | None) -> float:
    if retry_kind == "repetition":
        return max(base_temperature, 0.65)
    if retry_kind in {"continuity", "length"}:
        return min(base_temperature, 0.4)
    if retry_kind in {"state", "json"}:
        return min(base_temperature, 0.3)
    return min(base_temperature, 0.3)


def json_retry_feedback(error: Exception) -> str:
    return (
        "REGENERATION REQUIRED: the previous output failed SekAI's response contract. "
        f"Validation error: {str(error)[:1200]}\n\n"
        "Generate a completely new valid response that follows the supplied JSON schema, "
        "contains a 50-150 word story_text, a grounded applied_delta and exactly three "
        "distinct unresolved next choices."
    )


def deterministic_retry_feedback(issues: list[Any]) -> tuple[str, str]:
    blocking = blocking_issues(issues)
    failure_stage = "state" if blocking else "length"
    summary = format_issues_for_retry(issues)
    feedback = (
        "REGENERATION REQUIRED: the previous draft failed one or more deterministic "
        "validation checks. Correct every issue below in the next complete draft:\n"
        f"{summary}\n"
        "Keep all applied_delta changes visibly grounded in story_text, do not re-add "
        "existing state, do not remove absent state, and keep story_text between 50 and "
        "150 words."
    )
    return failure_stage, feedback


def repetition_retry_feedback(reason: str, semantic: bool = False) -> str:
    if semantic:
        return (
            "REGENERATION REQUIRED: the draft describes substantially the same event "
            "as an earlier scene. "
            f"{reason} Begin after the final event of the parent scene and introduce a "
            "different action, answer, discovery, location change, relationship change, "
            "or consequence."
        )
    return (
        "REGENERATION REQUIRED: the previous draft repeated an earlier scene. "
        f"{reason} Write a genuinely new scene beginning after the selected node. "
        "Directly perform or answer the supplied interaction. Use a different title, "
        "opening, action, and outcome."
    )


def continuity_retry_feedback(contradictions: list[dict[str, Any]]) -> str:
    contradiction_text = "\n".join(
        f"- {item['fact']} (contradiction={item['contradiction_score']:.2f})"
        for item in contradictions
    )
    return (
        "REGENERATION REQUIRED: a specialised Natural Language Inference model detected "
        "likely contradictions between the draft and the state that should exist after "
        "its applied_delta. Preserve these authoritative facts unless the requested "
        "interaction explicitly changes them:\n"
        f"{contradiction_text}"
    )


async def ollama_chat(payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    timeout = httpx.Timeout(OLLAMA_TIMEOUT_SECONDS, connect=5.0)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{ai_service.OLLAMA_BASE_URL}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        raw = response.json()
    return raw, time.perf_counter() - started


def make_payload(
    *,
    model: str,
    prompt: str,
    retry_feedback: str,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": ai_service.SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if retry_feedback:
        messages.append({"role": "user", "content": retry_feedback})

    return {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "format": ai_service.OLLAMA_RESPONSE_SCHEMA,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "repeat_last_n": 256,
            "repeat_penalty": 1.15,
            "top_k": 40,
            "top_p": 0.90,
            "seed": seed,
        },
    }


async def evaluate_condition(
    *,
    model: str,
    case: ContinuityCase,
    seed: int,
    max_attempts: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    condition_started = time.perf_counter()
    parent_memory = StoryMemory.model_validate(case.parent_memory)
    reference_nodes = build_reference_nodes(case)

    context, retrieval_seconds = await build_controlled_context(case)
    prompt = ai_service.build_prompt(context)

    attempts: list[dict[str, Any]] = []
    output_records: list[dict[str, Any]] = []

    retry_kind: str | None = None
    retry_feedback = ""
    eventual_accepted = False
    final_failure_stage: str | None = None
    final_labelled_rate: float | None = None
    final_instruction_rate: float | None = None
    final_nli_pass: bool | None = None
    final_word_count: int | None = None
    total_writer_seconds = 0.0
    total_semantic_seconds = 0.0
    total_nli_seconds = 0.0

    for attempt_number in range(1, max_attempts + 1):
        if attempt_number == 1:
            temperature = BASE_TEMPERATURE
        else:
            temperature = retry_temperature(BASE_TEMPERATURE, retry_kind)

        attempt_seed = seed + (attempt_number - 1)
        payload = make_payload(
            model=model,
            prompt=prompt,
            retry_feedback=retry_feedback,
            temperature=temperature,
            seed=attempt_seed,
        )

        attempt_started = time.perf_counter()
        failure_stage: str | None = None
        failure_detail = ""
        schema_valid = False
        deterministic_pass = False
        lexical_repetition_pass: bool | None = None
        semantic_repetition_pass: bool | None = None
        nli_pass: bool | None = None
        nli_checked_facts = 0
        nli_max_contradiction: float | None = None
        nli_contradictions: list[dict[str, Any]] = []
        semantic_seconds = 0.0
        nli_seconds = 0.0
        labelled_passed = 0
        labelled_total = 0
        labelled_details: list[str] = []
        instruction_passed = 0
        instruction_total = 0
        instruction_details: list[str] = []
        story_words: int | None = None
        raw: dict[str, Any] = {}
        content = ""
        draft: AIGeneratedDraft | None = None
        proposed_memory: StoryMemory | None = None
        ollama_metrics: dict[str, Any] = {
            "ollama_total_seconds": None,
            "load_seconds": None,
            "prompt_eval_seconds": None,
            "generation_seconds": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "tokens_per_second": None,
        }
        writer_wall_seconds = 0.0

        try:
            raw, writer_wall_seconds = await ollama_chat(payload)
            total_writer_seconds += writer_wall_seconds
            ollama_metrics = extract_ollama_metrics(raw)
            content = str(raw.get("message", {}).get("content", "")).strip()

            # ---------------------------------------------------------------
            # Experiment 2A: structured writer compatibility.
            # ---------------------------------------------------------------
            try:
                draft = AIGeneratedDraft.model_validate_json(content)
                schema_valid = True
            except (
                ValidationError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as error:
                failure_stage = "schema"
                failure_detail = str(error)[:2000]
                retry_kind = "json"
                retry_feedback = json_retry_feedback(error)

            if draft is not None:
                cleaned_delta = clean_redundant_delta(
                    parent_memory,
                    draft.applied_delta,
                )
                draft = draft.model_copy(update={"applied_delta": cleaned_delta})
                story_words = story_word_count(draft.story_text)

                deterministic_issues = validate_reviewed_draft(
                    story_text=draft.story_text,
                    parent_memory=parent_memory,
                    applied_delta=draft.applied_delta,
                    previous_nodes=[],
                    include_duplicate=False,
                )

                if deterministic_issues:
                    failure_stage, retry_feedback = deterministic_retry_feedback(
                        deterministic_issues
                    )
                    retry_kind = failure_stage
                    failure_detail = format_issues_for_retry(deterministic_issues)
                else:
                    deterministic_pass = True

                instruction_passed, instruction_total, instruction_details = (
                    instruction_adherence(draft.story_text, case)
                )

            # ---------------------------------------------------------------
            # Experiment 2B: repetition + continuity checks.
            # Only run later stages if the current production stage passed.
            # ---------------------------------------------------------------
            if draft is not None and deterministic_pass:
                lexical_reason = ai_service.repetition_reason(
                    draft.story_text,
                    reference_nodes,
                )
                lexical_repetition_pass = lexical_reason is None
                if lexical_reason:
                    failure_stage = "repetition"
                    failure_detail = lexical_reason
                    retry_kind = "repetition"
                    retry_feedback = repetition_retry_feedback(lexical_reason)

            if (
                draft is not None
                and deterministic_pass
                and lexical_repetition_pass is True
            ):
                semantic_started = time.perf_counter()
                semantic_reason = await semantic_repetition_reason(
                    draft.story_text,
                    reference_nodes,
                )
                semantic_seconds = time.perf_counter() - semantic_started
                total_semantic_seconds += semantic_seconds
                semantic_repetition_pass = semantic_reason is None

                if semantic_reason:
                    failure_stage = "semantic_repetition"
                    failure_detail = semantic_reason
                    retry_kind = "repetition"
                    retry_feedback = repetition_retry_feedback(
                        semantic_reason,
                        semantic=True,
                    )

            if (
                draft is not None
                and deterministic_pass
                and lexical_repetition_pass is True
                and semantic_repetition_pass is True
            ):
                proposed_memory = apply_state_delta(
                    parent_memory,
                    draft.applied_delta,
                )

                labelled_passed, labelled_total, labelled_details = labelled_continuity(
                    proposed_memory,
                    draft,
                    case,
                )

                nli_started = time.perf_counter()
                review = await asyncio.to_thread(
                    review_story_continuity,
                    draft.story_text,
                    proposed_memory,
                    draft.applied_delta,
                )
                nli_seconds = time.perf_counter() - nli_started
                total_nli_seconds += nli_seconds

                nli_checked_facts = review.checked_facts
                nli_max_contradiction = review.max_contradiction_score
                nli_contradictions = [
                    {
                        "fact": item.fact,
                        "contradiction_score": item.contradiction_score,
                        "entailment_score": item.entailment_score,
                        "neutral_score": item.neutral_score,
                    }
                    for item in review.contradictions
                ]
                nli_pass = not nli_contradictions

                if not nli_pass:
                    failure_stage = "continuity"
                    failure_detail = "; ".join(
                        f"{item['fact']} (contradiction={item['contradiction_score']:.2f})"
                        for item in nli_contradictions
                    )
                    retry_kind = "continuity"
                    retry_feedback = continuity_retry_feedback(nli_contradictions)

            accepted = (
                draft is not None
                and schema_valid
                and deterministic_pass
                and lexical_repetition_pass is True
                and semantic_repetition_pass is True
                and nli_pass is True
            )

            if accepted:
                eventual_accepted = True
                failure_stage = None
                failure_detail = ""

            labelled_rate = (
                labelled_passed / labelled_total
                if labelled_total
                else None
            )
            instruction_rate = (
                instruction_passed / instruction_total
                if instruction_total
                else None
            )

            if labelled_rate is not None:
                final_labelled_rate = labelled_rate
            if instruction_rate is not None:
                final_instruction_rate = instruction_rate
            if nli_pass is not None:
                final_nli_pass = nli_pass
            if story_words is not None:
                final_word_count = story_words

            attempt_row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "case_id": case.case_id,
                "seed": seed,
                "attempt": attempt_number,
                "attempt_seed": attempt_seed,
                "temperature": temperature,
                "accepted": accepted,
                "failure_stage": failure_stage or "",
                "failure_detail": failure_detail,
                "schema_valid": schema_valid,
                "word_count": story_words,
                "word_range_pass": (
                    story_words is not None and 50 <= story_words <= 150
                ),
                "deterministic_pass": deterministic_pass,
                "lexical_repetition_pass": lexical_repetition_pass,
                "semantic_repetition_pass": semantic_repetition_pass,
                "nli_pass": nli_pass,
                "nli_checked_facts": nli_checked_facts,
                "nli_max_contradiction": nli_max_contradiction,
                "labelled_checks_passed": labelled_passed,
                "labelled_checks_total": labelled_total,
                "labelled_continuity_rate": labelled_rate,
                "instruction_checks_passed": instruction_passed,
                "instruction_checks_total": instruction_total,
                "instruction_adherence_rate": instruction_rate,
                "writer_wall_seconds": round(writer_wall_seconds, 4),
                "semantic_seconds": round(semantic_seconds, 4),
                "nli_seconds": round(nli_seconds, 4),
                "attempt_pipeline_seconds": round(
                    time.perf_counter() - attempt_started,
                    4,
                ),
                **ollama_metrics,
            }
            attempts.append(attempt_row)

            output_records.append(
                {
                    **attempt_row,
                    "case_description": case.description,
                    "prompt": prompt,
                    "retry_feedback": retry_feedback if not accepted else "",
                    "raw_model_content": content,
                    "draft": draft.model_dump() if draft is not None else None,
                    "proposed_memory": (
                        proposed_memory.model_dump()
                        if proposed_memory is not None
                        else None
                    ),
                    "nli_contradictions": nli_contradictions,
                    "labelled_details": labelled_details,
                    "instruction_details": instruction_details,
                    "retrieved_context": context["retrieved_branch_context"],
                }
            )

            final_failure_stage = failure_stage
            if accepted:
                break

        except ContinuityModelUnavailableError as error:
            raise SystemExit(
                "The local DeBERTa continuity model is unavailable. Run SekAI's model "
                f"setup before this benchmark. Underlying error: {error}"
            ) from error
        except (httpx.HTTPError, ValueError, RuntimeError) as error:
            attempt_row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "case_id": case.case_id,
                "seed": seed,
                "attempt": attempt_number,
                "attempt_seed": attempt_seed,
                "temperature": temperature,
                "accepted": False,
                "failure_stage": "service_error",
                "failure_detail": str(error)[:2000],
                "schema_valid": False,
                "word_count": None,
                "word_range_pass": False,
                "deterministic_pass": False,
                "lexical_repetition_pass": None,
                "semantic_repetition_pass": None,
                "nli_pass": None,
                "nli_checked_facts": 0,
                "nli_max_contradiction": None,
                "labelled_checks_passed": 0,
                "labelled_checks_total": 0,
                "labelled_continuity_rate": None,
                "instruction_checks_passed": 0,
                "instruction_checks_total": 0,
                "instruction_adherence_rate": None,
                "writer_wall_seconds": round(writer_wall_seconds, 4),
                "semantic_seconds": round(semantic_seconds, 4),
                "nli_seconds": round(nli_seconds, 4),
                "attempt_pipeline_seconds": round(
                    time.perf_counter() - attempt_started,
                    4,
                ),
                **ollama_metrics,
            }
            attempts.append(attempt_row)
            output_records.append(
                {
                    **attempt_row,
                    "case_description": case.description,
                    "prompt": prompt,
                    "retry_feedback": retry_feedback,
                    "raw_model_content": content,
                    "draft": None,
                    "proposed_memory": None,
                    "nli_contradictions": [],
                    "labelled_details": [],
                    "instruction_details": [],
                    "retrieved_context": context["retrieved_branch_context"],
                }
            )
            final_failure_stage = "service_error"
            break

    condition_seconds = time.perf_counter() - condition_started
    first_pass_accepted = bool(attempts and attempts[0]["accepted"])

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "case_id": case.case_id,
        "seed": seed,
        "first_pass_accepted": first_pass_accepted,
        "eventual_accepted": eventual_accepted,
        "attempts": len(attempts),
        "final_failure_stage": "" if eventual_accepted else (final_failure_stage or "unknown"),
        "retrieval_seconds": round(retrieval_seconds, 4),
        "writer_seconds_total": round(total_writer_seconds, 4),
        "semantic_seconds_total": round(total_semantic_seconds, 4),
        "nli_seconds_total": round(total_nli_seconds, 4),
        "effective_pipeline_seconds": round(condition_seconds, 4),
        "final_word_count": final_word_count,
        "final_labelled_continuity_rate": final_labelled_rate,
        "final_instruction_adherence_rate": final_instruction_rate,
        "final_nli_pass": final_nli_pass,
    }

    return attempts, summary, output_records


# ---------------------------------------------------------------------------
# Persistence and summary output.
# ---------------------------------------------------------------------------

ATTEMPT_FIELDS = [
    "timestamp",
    "model",
    "case_id",
    "seed",
    "attempt",
    "attempt_seed",
    "temperature",
    "accepted",
    "failure_stage",
    "failure_detail",
    "schema_valid",
    "word_count",
    "word_range_pass",
    "deterministic_pass",
    "lexical_repetition_pass",
    "semantic_repetition_pass",
    "nli_pass",
    "nli_checked_facts",
    "nli_max_contradiction",
    "labelled_checks_passed",
    "labelled_checks_total",
    "labelled_continuity_rate",
    "instruction_checks_passed",
    "instruction_checks_total",
    "instruction_adherence_rate",
    "writer_wall_seconds",
    "semantic_seconds",
    "nli_seconds",
    "attempt_pipeline_seconds",
    "ollama_total_seconds",
    "load_seconds",
    "prompt_eval_seconds",
    "generation_seconds",
    "prompt_tokens",
    "output_tokens",
    "tokens_per_second",
]

SUMMARY_FIELDS = [
    "timestamp",
    "model",
    "case_id",
    "seed",
    "first_pass_accepted",
    "eventual_accepted",
    "attempts",
    "final_failure_stage",
    "retrieval_seconds",
    "writer_seconds_total",
    "semantic_seconds_total",
    "nli_seconds_total",
    "effective_pipeline_seconds",
    "final_word_count",
    "final_labelled_continuity_rate",
    "final_instruction_adherence_rate",
    "final_nli_pass",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def mean_optional(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.mean(clean) if clean else None


def print_model_summary(
    models: list[str],
    attempts: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 110)
    print("SEKAI QWEN STRUCTURED / CONTINUITY / RETRY BENCHMARK SUMMARY")
    print("=" * 110)

    for model in models:
        model_summaries = [row for row in summaries if row["model"] == model]
        model_attempts = [row for row in attempts if row["model"] == model]
        if not model_summaries:
            continue

        first_pass = sum(bool(row["first_pass_accepted"]) for row in model_summaries)
        eventual = sum(bool(row["eventual_accepted"]) for row in model_summaries)
        n = len(model_summaries)

        mean_attempts = statistics.mean(float(row["attempts"]) for row in model_summaries)
        mean_effective = statistics.mean(
            float(row["effective_pipeline_seconds"]) for row in model_summaries
        )
        mean_labelled = mean_optional(
            row["final_labelled_continuity_rate"] for row in model_summaries
        )
        mean_instruction = mean_optional(
            row["final_instruction_adherence_rate"] for row in model_summaries
        )
        mean_tps = mean_optional(row["tokens_per_second"] for row in model_attempts)

        print(f"\nMODEL: {model}")
        print(f"  Conditions:                 {n}")
        print(f"  First-pass acceptance:      {first_pass}/{n} ({first_pass / n:.1%})")
        print(f"  Eventual acceptance:        {eventual}/{n} ({eventual / n:.1%})")
        print(f"  Mean attempts:              {mean_attempts:.2f}")
        print(f"  Mean effective latency:     {mean_effective:.2f} s")
        if mean_tps is not None:
            print(f"  Mean writer throughput:     {mean_tps:.2f} tok/s")
        if mean_labelled is not None:
            print(f"  Mean labelled continuity:   {mean_labelled:.1%}")
        if mean_instruction is not None:
            print(f"  Mean instruction coverage:  {mean_instruction:.1%}")

        failures: dict[str, int] = {}
        for row in model_attempts:
            stage = str(row.get("failure_stage") or "accepted")
            failures[stage] = failures.get(stage, 0) + 1

        print("  Attempt outcomes by stage:")
        for stage, count in sorted(failures.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {stage:<24} {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled SekAI benchmark for Qwen structured generation, continuity "
            "checking and same-model retry behaviour."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Exact Ollama writer tags to compare.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Deterministic base seeds. The retry uses seed+1.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASE_BY_ID),
        default=[case.case_id for case in TEST_CASES],
        help="Subset of labelled continuity cases to run.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        choices=[1, 2],
        default=MAX_ATTEMPTS,
        help="1 tests first-pass compatibility only; 2 also tests SekAI-style retry.",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    models: list[str] = list(args.models)
    seeds: list[int] = list(args.seeds)
    cases = [CASE_BY_ID[case_id] for case_id in args.cases]

    requested_models_exist(models)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSONL.write_text("", encoding="utf-8")

    all_attempts: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []

    total_conditions = len(models) * len(cases) * len(seeds)
    completed = 0

    print("SekAI controlled continuity benchmark")
    print(f"Models:      {models}")
    print(f"Cases:       {[case.case_id for case in cases]}")
    print(f"Seeds:       {seeds}")
    print(f"Max attempts:{args.attempts}")
    print(f"Conditions:  {total_conditions}")
    print("Cross-model fallback: DISABLED")

    for model in models:
        print("\n" + "=" * 110)
        print(f"WRITER MODEL: {model}")
        print("=" * 110)

        for case in cases:
            for seed in seeds:
                completed += 1
                print(
                    f"[{completed}/{total_conditions}] "
                    f"case={case.case_id} seed={seed}"
                )

                attempts, summary, outputs = await evaluate_condition(
                    model=model,
                    case=case,
                    seed=seed,
                    max_attempts=args.attempts,
                )

                all_attempts.extend(attempts)
                all_summaries.append(summary)
                append_jsonl(OUTPUT_JSONL, outputs)

                # Persist continuously in case a long benchmark is interrupted.
                write_csv(ATTEMPT_CSV, all_attempts, ATTEMPT_FIELDS)
                write_csv(SUMMARY_CSV, all_summaries, SUMMARY_FIELDS)

                status = "ACCEPTED" if summary["eventual_accepted"] else "REJECTED"
                print(
                    f"  {status} | attempts={summary['attempts']} | "
                    f"pipeline={summary['effective_pipeline_seconds']:.2f}s | "
                    f"labelled={summary['final_labelled_continuity_rate']} | "
                    f"instruction={summary['final_instruction_adherence_rate']}"
                )

    print_model_summary(models, all_attempts, all_summaries)

    print("\nFiles written:")
    print(f"  Attempts: {ATTEMPT_CSV}")
    print(f"  Summary:  {SUMMARY_CSV}")
    print(f"  Outputs:  {OUTPUT_JSONL}")


if __name__ == "__main__":
    asyncio.run(async_main())
