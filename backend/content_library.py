from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schemas import StoryElementDefinition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "content" / "story_elements.json"

MAX_GUIDED_ELEMENTS = 3
MAX_STRICT_ELEMENTS = 8
MIN_GUIDED_SCORE = 5

CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "creature": {
        "creature", "monster", "enemy", "guard", "beast", "alien",
        "robot", "drone", "attack", "combat",
    },
    "hazard": {
        "trap", "hazard", "danger", "corridor", "door", "floor",
        "alarm", "detect", "disarm",
    },
    "item": {
        "item", "loot", "reward", "equipment", "weapon", "key",
        "potion", "medkit", "search", "obtain",
    },
    "location": {
        "room", "location", "area", "chamber", "tower", "station",
        "deck", "level",
    },
    "npc": {
        "npc", "person", "character", "merchant", "guard", "captain",
        "villager",
    },
    "encounter": {
        "encounter", "event", "challenge", "scene",
    },
}


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


@lru_cache(maxsize=1)
def load_factory_elements() -> tuple[StoryElementDefinition, ...]:
    """
    Load the immutable factory catalogue.

    The JSON file is no longer the live library. It is used only to seed a
    new database and to provide the original values used by reset operations.
    """
    try:
        raw = json.loads(
            CATALOG_PATH.read_text(encoding="utf-8")
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Story element factory catalogue is missing: {CATALOG_PATH}"
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Story element factory catalogue contains invalid JSON: "
            f"{error}"
        ) from error

    if not isinstance(raw, list):
        raise RuntimeError(
            "Story element factory catalogue must contain a JSON array."
        )

    elements = tuple(
        StoryElementDefinition.model_validate(item)
        for item in raw
    )

    ids = [element.id for element in elements]
    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Story element factory catalogue contains duplicate IDs."
        )

    return elements


def _candidate_score(
    element: dict[str, Any],
    *,
    preference: str,
    query_text: str,
    avoid_text: str,
) -> int:
    query_tokens = _tokens(query_text)

    # Pack/genre names are deliberately excluded from relevance scoring.
    # Otherwise every element in a Science Fiction project receives a positive
    # score simply because both the story and element say "Science Fiction".
    element_tokens = (
        _tokens(str(element.get("name", "")))
        | {
            token
            for tag in element.get("tags", [])
            for token in _tokens(str(tag))
        }
    )

    score = len(query_tokens & element_tokens) * 3

    element_name = str(
        element.get("name", "")
    ).strip().casefold()

    if (
        element_name
        and element_name in query_text.casefold()
    ):
        score += 20

    category = str(element.get("category", ""))
    if (
        query_tokens
        & CATEGORY_KEYWORDS.get(category, set())
    ):
        score += 5

    if preference == "preferred":
        score += 3

    # Discourage immediate repetition of an element used in recent scenes.
    if (
        element_name
        and element_name in avoid_text.casefold()
    ):
        score -= 8

    return score


def select_prompt_elements(
    project: dict[str, Any],
    catalog: list[dict[str, Any]],
    query_text: str,
    *,
    avoid_text: str = "",
) -> list[dict[str, Any]]:
    """
    Rank active database-backed elements enabled for one project.

    The returned records are lower-priority reference knowledge. This function
    never changes the player's instruction, GM instruction, or branch memory.
    """
    mode = str(
        project.get("story_element_mode", "guided")
    )

    if mode == "off":
        return []

    preference_by_id = {
        str(selection.get("element_id", "")):
            str(selection.get("preference", "available"))
        for selection in project.get("story_elements", [])
        if selection.get("element_id")
    }

    candidates: list[dict[str, Any]] = []

    for element in catalog:
        element_id = str(element.get("id", ""))
        preference = preference_by_id.get(element_id)

        if not preference:
            continue

        candidates.append(
            {
                **element,
                "preference": preference,
                "source": (
                    "builtin"
                    if element.get("is_builtin")
                    else "custom"
                ),
            }
        )

    if not candidates:
        return []

    scored = [
        (
            _candidate_score(
                candidate,
                preference=str(
                    candidate.get("preference", "available")
                ),
                query_text=query_text,
                avoid_text=avoid_text,
            ),
            candidate,
        )
        for candidate in candidates
    ]

    scored.sort(
        key=lambda item: (
            item[0],
            item[1].get("preference") == "preferred",
            item[1].get("name", ""),
        ),
        reverse=True,
    )

    if mode == "strict":
        return [
            candidate
            for _score, candidate
            in scored[:MAX_STRICT_ELEMENTS]
        ]

    relevant = [
        candidate
        for score, candidate in scored
        if score >= MIN_GUIDED_SCORE
    ]

    if relevant:
        return relevant[:MAX_GUIDED_ELEMENTS]

    preferred = [
        candidate
        for _score, candidate in scored
        if candidate.get("preference") == "preferred"
    ]

    return preferred[:1]
