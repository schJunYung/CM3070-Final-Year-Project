from __future__ import annotations

import re
from typing import Any

from .schemas import GenerationIssue, StateDelta, StoryChoice, StoryMemory


STORY_MIN_WORDS = 50
STORY_MAX_WORDS = 150


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", value.casefold()))


def story_word_count(story_text: str) -> int:
    """Count words using the same rule for generation, review and saving."""
    return len(re.findall(r"\b[\w'-]+\b", story_text))


def _mentioned(story_text: str, value: str) -> bool:
    """Approximate whether a named state value is represented in scene prose."""
    story_tokens = set(_normalise(story_text).split())
    value_tokens = [
        token
        for token in _normalise(value).split()
        if len(token) >= 3
    ]

    if not value_tokens:
        return True

    matched = sum(token in story_tokens for token in value_tokens)
    return matched / len(value_tokens) >= 0.75


def _issue(
    *,
    stage: str,
    severity: str,
    target: str,
    message: str,
    suggestion: str,
    field: str | None = None,
    value: str | None = None,
    can_override: bool = False,
) -> GenerationIssue:
    return GenerationIssue(
        stage=stage,
        severity=severity,
        target=target,
        field=field,
        value=value,
        message=message,
        suggestion=suggestion,
        can_override=can_override,
    )


def story_length_issues(story_text: str) -> list[GenerationIssue]:
    """Return a non-blocking issue when an AI scene is outside 50–150 words."""
    count = story_word_count(story_text)

    if count < STORY_MIN_WORDS:
        missing = STORY_MIN_WORDS - count
        return [
            _issue(
                stage="length",
                severity="warning",
                target="story_text",
                message=(
                    f"The scene contains {count} words; AI scenes should contain "
                    f"at least {STORY_MIN_WORDS} words."
                ),
                suggestion=(
                    f"Add at least {missing} meaningful "
                    f"word{'s' if missing != 1 else ''}."
                ),
                can_override=True,
            )
        ]

    if count > STORY_MAX_WORDS:
        excess = count - STORY_MAX_WORDS
        return [
            _issue(
                stage="length",
                severity="warning",
                target="story_text",
                message=(
                    f"The scene contains {count} words; AI scenes should contain "
                    f"no more than {STORY_MAX_WORDS} words."
                ),
                suggestion=(
                    f"Remove approximately {excess} word"
                    f"{'s' if excess != 1 else ''} while preserving the "
                    "important action and consequence."
                ),
                can_override=True,
            )
        ]

    return []


def _keys(values: list[str]) -> set[str]:
    return {
        value.strip().casefold()
        for value in values
        if value.strip()
    }


def _original_values(values: list[str]) -> dict[str, str]:
    return {
        value.strip().casefold(): value.strip()
        for value in values
        if value.strip()
    }


def state_transition_issues(
    memory: StoryMemory,
    delta: StateDelta,
) -> list[GenerationIssue]:
    """Return every redundant or impossible state transition in a delta."""
    issues: list[GenerationIssue] = []

    additions = [
        (
            "active_characters_add",
            "active character",
            delta.active_characters_add,
            memory.active_characters,
            "already active",
        ),
        (
            "inventory_add",
            "inventory item",
            delta.inventory_add,
            memory.inventory,
            "already owned",
        ),
        (
            "goals_add",
            "goal",
            delta.goals_add,
            memory.goals,
            "already active",
        ),
        (
            "unresolved_clues_add",
            "clue",
            delta.unresolved_clues_add,
            memory.unresolved_clues,
            "already unresolved",
        ),
    ]

    for field, label, added, existing, reason in additions:
        overlap = _keys(added) & _keys(existing)
        originals = _original_values(added)

        for key in sorted(overlap):
            value = originals.get(key, key)
            issues.append(
                _issue(
                    stage="state_transition",
                    severity="blocking",
                    target="applied_delta",
                    field=field,
                    value=value,
                    message=f"The {label} '{value}' is {reason}.",
                    suggestion=(
                        f"Remove '{value}' from {field}; the parent memory "
                        "already contains it."
                    ),
                )
            )

    removals = [
        (
            "active_characters_remove",
            "active character",
            delta.active_characters_remove,
            memory.active_characters,
        ),
        (
            "inventory_remove",
            "inventory item",
            delta.inventory_remove,
            memory.inventory,
        ),
        (
            "goals_complete",
            "goal",
            delta.goals_complete,
            memory.goals,
        ),
        (
            "unresolved_clues_resolve",
            "clue",
            delta.unresolved_clues_resolve,
            memory.unresolved_clues,
        ),
    ]

    for field, label, removed, existing in removals:
        missing = _keys(removed) - _keys(existing)
        originals = _original_values(removed)

        for key in sorted(missing):
            value = originals.get(key, key)
            issues.append(
                _issue(
                    stage="state_transition",
                    severity="blocking",
                    target="applied_delta",
                    field=field,
                    value=value,
                    message=(
                        f"The {label} '{value}' cannot be removed because it "
                        "is not present in the parent memory."
                    ),
                    suggestion=f"Remove '{value}' from {field}.",
                )
            )

    if (
        delta.location_set
        and delta.location_set.strip().casefold()
        == memory.location.strip().casefold()
    ):
        issues.append(
            _issue(
                stage="state_transition",
                severity="blocking",
                target="applied_delta",
                field="location_set",
                value=delta.location_set,
                message=f"The location is already '{memory.location}'.",
                suggestion="Clear location_set because no location change occurred.",
            )
        )

    if delta.threat_set is not None and delta.threat_set == memory.threat:
        issues.append(
            _issue(
                stage="state_transition",
                severity="blocking",
                target="applied_delta",
                field="threat_set",
                value=delta.threat_set,
                message=f"The threat level is already '{memory.threat}'.",
                suggestion="Clear threat_set because the threat level did not change.",
            )
        )

    relationship_keys = {
        key.strip().casefold(): key
        for key in memory.relationships
    }

    for character, status in delta.relationships_set.items():
        existing_key = relationship_keys.get(character.strip().casefold())
        existing_status = (
            memory.relationships.get(existing_key)
            if existing_key is not None
            else None
        )

        if (
            existing_status is not None
            and existing_status.strip().casefold() == status.strip().casefold()
        ):
            issues.append(
                _issue(
                    stage="state_transition",
                    severity="blocking",
                    target="applied_delta",
                    field="relationships_set",
                    value=character,
                    message=(
                        f"The relationship with '{existing_key}' is already "
                        f"'{existing_status}'."
                    ),
                    suggestion=(
                        f"Remove '{character}' from relationships_set because "
                        "the relationship did not change."
                    ),
                )
            )

    return issues


def clean_redundant_delta(
    memory: StoryMemory,
    delta: StateDelta,
) -> StateDelta:
    """Remove harmless no-op additions produced by the writer before review."""
    data = delta.model_dump()

    active_characters = _keys(memory.active_characters)
    inventory = _keys(memory.inventory)
    goals = _keys(memory.goals)
    clues = _keys(memory.unresolved_clues)

    data["active_characters_add"] = [
        value
        for value in data["active_characters_add"]
        if value.strip().casefold() not in active_characters
    ]
    data["inventory_add"] = [
        value
        for value in data["inventory_add"]
        if value.strip().casefold() not in inventory
    ]
    data["goals_add"] = [
        value
        for value in data["goals_add"]
        if value.strip().casefold() not in goals
    ]
    data["unresolved_clues_add"] = [
        value
        for value in data["unresolved_clues_add"]
        if value.strip().casefold() not in clues
    ]

    if (
        data["location_set"]
        and data["location_set"].strip().casefold()
        == memory.location.strip().casefold()
    ):
        data["location_set"] = None

    if data["threat_set"] == memory.threat:
        data["threat_set"] = None

    cleaned_relationships: dict[str, str] = {}

    for character, status in data["relationships_set"].items():
        existing_key = next(
            (
                key
                for key in memory.relationships
                if key.strip().casefold() == character.strip().casefold()
            ),
            None,
        )
        existing_status = (
            memory.relationships.get(existing_key)
            if existing_key is not None
            else None
        )

        if (
            existing_status is not None
            and existing_status.strip().casefold() == status.strip().casefold()
        ):
            continue

        cleaned_relationships[character] = status

    data["relationships_set"] = cleaned_relationships
    return StateDelta.model_validate(data)


def delta_grounding_issues(
    story_text: str,
    memory: StoryMemory,
    delta: StateDelta,
) -> list[GenerationIssue]:
    """Return every concrete state change not sufficiently supported by prose."""
    issues: list[GenerationIssue] = []

    def append_missing(
        *,
        field: str,
        value: str,
        message: str,
        suggestion: str,
    ) -> None:
        if not _mentioned(story_text, value):
            issues.append(
                _issue(
                    stage="state_grounding",
                    severity="blocking",
                    target="applied_delta",
                    field=field,
                    value=value,
                    message=message,
                    suggestion=suggestion,
                )
            )

    for item in delta.inventory_add:
        append_missing(
            field="inventory_add",
            value=item,
            message=(
                f"The scene does not establish that the item '{item}' was obtained."
            ),
            suggestion=(
                f"Describe the character obtaining '{item}' in the scene text, "
                "or remove it from inventory_add."
            ),
        )

    for item in delta.inventory_remove:
        append_missing(
            field="inventory_remove",
            value=item,
            message=(
                f"The scene does not establish that the item '{item}' was removed."
            ),
            suggestion=(
                f"Describe '{item}' being lost, consumed, destroyed or given away, "
                "or remove it from inventory_remove."
            ),
        )

    if (
        delta.location_set
        and delta.location_set.strip().casefold()
        != memory.location.strip().casefold()
        and not _mentioned(story_text, delta.location_set)
    ):
        issues.append(
            _issue(
                stage="state_grounding",
                severity="blocking",
                target="applied_delta",
                field="location_set",
                value=delta.location_set,
                message=(
                    f"The scene does not establish movement or arrival at "
                    f"'{delta.location_set}'."
                ),
                suggestion=(
                    f"Describe arrival at '{delta.location_set}' in the scene text, "
                    "or clear location_set."
                ),
            )
        )

    for character in delta.active_characters_add:
        append_missing(
            field="active_characters_add",
            value=character,
            message=(
                f"The scene does not establish that '{character}' becomes present."
            ),
            suggestion=(
                f"Introduce '{character}' as present in the scene, or remove the "
                "character from active_characters_add."
            ),
        )

    for character in delta.active_characters_remove:
        append_missing(
            field="active_characters_remove",
            value=character,
            message=(
                f"The scene does not establish that '{character}' leaves or is no "
                "longer active in the scene."
            ),
            suggestion=(
                f"Describe '{character}' leaving, becoming unavailable or being "
                "removed from the scene, or remove the memory change."
            ),
        )

    grounded_array_fields: tuple[tuple[str, list[str], str, str], ...] = (
        (
            "goals_add",
            delta.goals_add,
            "The goal '{value}' is not established as a new objective by the scene.",
            (
                "Establish '{value}' as a new objective in the scene text, or "
                "remove it from goals_add."
            ),
        ),
        (
            "goals_complete",
            delta.goals_complete,
            "The scene does not establish that the goal '{value}' was completed.",
            (
                "Describe how '{value}' is achieved or abandoned, or remove it "
                "from goals_complete."
            ),
        ),
        (
            "unresolved_clues_add",
            delta.unresolved_clues_add,
            "The clue '{value}' is not established or discovered in the scene.",
            (
                "Introduce or discover '{value}' in the scene text, or remove it "
                "from unresolved_clues_add."
            ),
        ),
        (
            "unresolved_clues_resolve",
            delta.unresolved_clues_resolve,
            "The scene does not establish that the clue '{value}' was resolved.",
            (
                "Resolve '{value}' in the scene text, or remove it from "
                "unresolved_clues_resolve."
            ),
        ),
        (
            "decisions_add",
            delta.decisions_add,
            "The decision '{value}' is not established by the scene.",
            (
                "Make '{value}' an explicit decision in the scene text, or remove "
                "it from decisions_add."
            ),
        ),
    )

    for field, values, message_template, suggestion_template in grounded_array_fields:
        for value in values:
            append_missing(
                field=field,
                value=value,
                message=message_template.format(value=value),
                suggestion=suggestion_template.format(value=value),
            )

    for character in delta.relationships_set:
        append_missing(
            field="relationships_set",
            value=character,
            message=(
                f"The scene changes the relationship with '{character}', but does "
                "not mention that character."
            ),
            suggestion=(
                f"Describe an interaction with '{character}' that supports the "
                "relationship change, or remove that relationship entry."
            ),
        )

    return issues


def choice_issues(choices: list[StoryChoice]) -> list[GenerationIssue]:
    """Return structured problems in the three editable AI choices."""
    issues: list[GenerationIssue] = []

    if len(choices) != 3:
        issues.append(
            _issue(
                stage="choices",
                severity="blocking",
                target="choices",
                field="choices",
                message=(
                    f"A reviewed AI scene must contain exactly three choices; "
                    f"{len(choices)} were provided."
                ),
                suggestion="Restore three distinct next-action choices.",
            )
        )
        return issues

    labels = [choice.label.strip().casefold() for choice in choices]
    if len(set(labels)) != 3:
        issues.append(
            _issue(
                stage="choices",
                severity="blocking",
                target="choices",
                field="choices",
                message="The three choice labels must be distinct.",
                suggestion="Edit repeated choices so each leads in a different direction.",
            )
        )

    choice_ids = [choice.id for choice in choices]
    if len(set(choice_ids)) != 3:
        issues.append(
            _issue(
                stage="choices",
                severity="blocking",
                target="choices",
                field="choices",
                message="The three choice IDs must be distinct.",
                suggestion="Regenerate the draft because its choice identifiers are invalid.",
            )
        )

    return issues


def duplicate_scene_issues(
    story_text: str,
    previous_nodes: list[dict[str, Any]],
) -> list[GenerationIssue]:
    """Block an exact normalized duplicate of an earlier branch scene."""
    candidate = _normalise(story_text)
    if not candidate:
        return []

    for node in previous_nodes:
        if candidate == _normalise(str(node.get("story_text", ""))):
            title = str(node.get("title", "Earlier scene"))
            return [
                _issue(
                    stage="duplicate",
                    severity="blocking",
                    target="story_text",
                    field="story_text",
                    value=title,
                    message=f"This draft is identical to the earlier scene '{title}'.",
                    suggestion=(
                        "Edit the scene so it advances the story, or regenerate it."
                    ),
                )
            ]

    return []


def validate_reviewed_draft(
    *,
    story_text: str,
    parent_memory: StoryMemory,
    applied_delta: StateDelta,
    previous_nodes: list[dict[str, Any]] | None = None,
    choices: list[StoryChoice] | None = None,
    include_duplicate: bool = True,
) -> list[GenerationIssue]:
    """Run every deterministic validation that must agree before persistence."""
    issues: list[GenerationIssue] = []
    issues.extend(story_length_issues(story_text))
    issues.extend(state_transition_issues(parent_memory, applied_delta))
    issues.extend(delta_grounding_issues(story_text, parent_memory, applied_delta))

    if choices is not None:
        issues.extend(choice_issues(choices))

    if include_duplicate:
        issues.extend(
            duplicate_scene_issues(
                story_text,
                previous_nodes or [],
            )
        )

    return issues


def blocking_issues(issues: list[GenerationIssue]) -> list[GenerationIssue]:
    return [issue for issue in issues if issue.severity == "blocking"]


def warning_issues(issues: list[GenerationIssue]) -> list[GenerationIssue]:
    return [issue for issue in issues if issue.severity == "warning"]


def format_issues_for_retry(issues: list[GenerationIssue]) -> str:
    """Build compact model retry feedback while preserving every detected issue."""
    lines: list[str] = []

    for issue in issues:
        location = issue.field or issue.target or issue.stage
        line = f"- {location}: {issue.message}"
        if issue.suggestion:
            line += f" Fix: {issue.suggestion}"
        lines.append(line)

    return "\n".join(lines)


# Compatibility helpers for any older imports/tests that expected one string.
def story_length_issue(story_text: str) -> str | None:
    issues = story_length_issues(story_text)
    return issues[0].message if issues else None


def state_transition_issue(memory: StoryMemory, delta: StateDelta) -> str | None:
    issues = state_transition_issues(memory, delta)
    return issues[0].message if issues else None


def delta_grounding_issue(
    story_text: str,
    memory: StoryMemory,
    delta: StateDelta,
) -> str | None:
    issues = delta_grounding_issues(story_text, memory, delta)
    return issues[0].message if issues else None
