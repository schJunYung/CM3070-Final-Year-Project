from backend.schemas import StateDelta, StoryMemory
from backend.validation_service import (
    clean_redundant_delta,
    delta_grounding_issues,
    state_transition_issues,
    validate_reviewed_draft,
)


def test_short_scene_reports_length_and_unsupported_goal_together() -> None:
    story = (
        "The zombie collapses as we aim and fire at it. We retrieve the key "
        "card and note its location. The medical bay's darkness fades slightly "
        "as we move forward. Sarah notices a faint glow near the door, "
        "suggesting something might be hidden there."
    )
    memory = StoryMemory(
        location="Medical bay",
        active_characters=["Sarah"],
    )
    delta = StateDelta(
        inventory_add=["key card"],
        goals_add=["locate and rescue remaining crew members"],
    )

    issues = validate_reviewed_draft(
        story_text=story,
        parent_memory=memory,
        applied_delta=delta,
        previous_nodes=[],
    )

    assert any(issue.stage == "length" for issue in issues)
    assert any(
        issue.field == "goals_add"
        and issue.value == "locate and rescue remaining crew members"
        and issue.severity == "blocking"
        for issue in issues
    )
    assert not any(
        issue.field == "inventory_add"
        and issue.value == "key card"
        for issue in issues
    )


def test_grounding_returns_field_and_value_for_every_supported_delta_field() -> None:
    memory = StoryMemory(
        location="Medical bay",
        active_characters=["Sarah", "Doctor Vale"],
        inventory=["old key"],
        relationships={"Sarah": "cautious ally"},
        goals=["restore power"],
        unresolved_clues=["red signal"],
    )
    delta = StateDelta(
        location_set="Command deck",
        active_characters_add=["Captain Imani"],
        active_characters_remove=["Doctor Vale"],
        inventory_add=["security badge"],
        inventory_remove=["old key"],
        relationships_set={"Sarah": "trusted ally"},
        goals_add=["rescue the crew"],
        goals_complete=["restore power"],
        unresolved_clues_add=["blue signal"],
        unresolved_clues_resolve=["red signal"],
        decisions_add=["seal the lower deck"],
    )

    issues = delta_grounding_issues(
        "A quiet ventilation fan turns above the empty corridor.",
        memory,
        delta,
    )

    fields = {issue.field for issue in issues}
    assert {
        "location_set",
        "active_characters_add",
        "active_characters_remove",
        "inventory_add",
        "inventory_remove",
        "relationships_set",
        "goals_add",
        "goals_complete",
        "unresolved_clues_add",
        "unresolved_clues_resolve",
        "decisions_add",
    } <= fields
    assert all(issue.value for issue in issues)
    assert all(issue.target == "applied_delta" for issue in issues)
    assert all(issue.severity == "blocking" for issue in issues)


def test_state_transition_reports_all_invalid_changes_not_only_first() -> None:
    memory = StoryMemory(
        location="Medical bay",
        inventory=["key card"],
        goals=["restore power"],
        threat="Medium",
    )
    delta = StateDelta(
        location_set="Medical bay",
        inventory_add=["key card"],
        inventory_remove=["plasma cutter"],
        goals_add=["restore power"],
        threat_set="Medium",
    )

    issues = state_transition_issues(memory, delta)
    fields = {issue.field for issue in issues}

    assert "location_set" in fields
    assert "inventory_add" in fields
    assert "inventory_remove" in fields
    assert "goals_add" in fields
    assert "threat_set" in fields
    assert len(issues) >= 5


def test_clean_redundant_delta_removes_harmless_writer_noops() -> None:
    memory = StoryMemory(
        location="Medical bay",
        inventory=["key card"],
        goals=["restore power"],
        relationships={"Sarah": "ally"},
        threat="Low",
    )
    delta = StateDelta(
        location_set="Medical bay",
        inventory_add=["key card", "medkit"],
        goals_add=["restore power", "reach engineering"],
        relationships_set={"Sarah": "ally", "Vale": "neutral"},
        threat_set="Low",
    )

    cleaned = clean_redundant_delta(memory, delta)

    assert cleaned.location_set is None
    assert cleaned.inventory_add == ["medkit"]
    assert cleaned.goals_add == ["reach engineering"]
    assert cleaned.relationships_set == {"Vale": "neutral"}
    assert cleaned.threat_set is None


def test_choice_validation_reports_duplicate_labels() -> None:
    from backend.schemas import StoryChoice
    from backend.validation_service import choice_issues

    choices = [
        StoryChoice(action_type="do", label="Open the door"),
        StoryChoice(action_type="ask", label="Open the door"),
        StoryChoice(action_type="continue", label="Wait and listen"),
    ]

    issues = choice_issues(choices)

    assert any(
        issue.stage == "choices"
        and issue.target == "choices"
        and issue.severity == "blocking"
        for issue in issues
    )
