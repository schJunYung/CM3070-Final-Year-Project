from __future__ import annotations

from .schemas import StateDelta, StoryMemory


def _normalise(value: str) -> str:
    return value.strip().casefold()

def _matching_key(
    mapping: dict[str, str],
    candidate: str,
) -> str | None:
    candidate_key = _normalise(candidate)

    return next(
        (
            key
            for key in mapping
            if _normalise(key) == candidate_key
        ),
        None,
    )

def add_unique(items: list[str], new_items: list[str]) -> list[str]:
    """Append non-empty values while preserving order and uniqueness."""
    result = list(items)
    known = {_normalise(item) for item in result}

    for item in new_items:
        clean = item.strip()
        if clean and _normalise(clean) not in known:
            result.append(clean)
            known.add(_normalise(clean))

    return result


def remove_values(items: list[str], removed: list[str]) -> list[str]:
    """Remove values using case-insensitive matching."""
    removed_keys = {
        _normalise(item)
        for item in removed
        if item.strip()
    }
    return [
        item
        for item in items
        if _normalise(item) not in removed_keys
    ]


def apply_state_delta(
    memory: StoryMemory,
    delta: StateDelta,
) -> StoryMemory:
    """Apply one validated state delta to a copy of a memory snapshot."""
    next_memory = memory.model_copy(deep=True)

    if delta.location_set:
        next_memory.location = delta.location_set.strip()

    next_memory.active_characters = add_unique(
        next_memory.active_characters,
        delta.active_characters_add,
    )
    next_memory.active_characters = remove_values(
        next_memory.active_characters,
        delta.active_characters_remove,
    )

    next_memory.inventory = add_unique(
        next_memory.inventory,
        delta.inventory_add,
    )
    next_memory.inventory = remove_values(
        next_memory.inventory,
        delta.inventory_remove,
    )

    for character, status in delta.relationships_set.items():
        character_clean = character.strip()
        status_clean = status.strip()

        if character_clean and status_clean:
            existing_key = _matching_key(
                next_memory.relationships,
                character_clean,
            )

            target_key = existing_key or character_clean
            next_memory.relationships[target_key] = status_clean

    next_memory.goals = add_unique(
        next_memory.goals,
        delta.goals_add,
    )
    next_memory.goals = remove_values(
        next_memory.goals,
        delta.goals_complete,
    )

    next_memory.unresolved_clues = add_unique(
        next_memory.unresolved_clues,
        delta.unresolved_clues_add,
    )
    next_memory.unresolved_clues = remove_values(
        next_memory.unresolved_clues,
        delta.unresolved_clues_resolve,
    )

    next_memory.decisions = add_unique(
        next_memory.decisions,
        delta.decisions_add,
    )

    if delta.threat_set:
        next_memory.threat = delta.threat_set

    return next_memory
