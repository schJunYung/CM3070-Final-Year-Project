from __future__ import annotations

from pathlib import Path

import pytest

from backend import db, services
from backend.schemas import ProjectCreateRequest, SaveNodeRequest, StateDelta, StoryMemory


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()


def create_story(title: str = "Test story") -> dict:
    return db.create_project(
        ProjectCreateRequest(
            title=title,
            genre="Fantasy",
            tone="Mysterious",
            world_summary="A ruined tower stands beside a forest.",
            main_objective="Discover the tower's secret.",
            starting_scene_title="At the tower",
            starting_scene_text="The adventurer reaches a ruined tower where a guard waits beside the gate.",
            initial_memory=StoryMemory(
                location="Tower gate",
                inventory=["Cow skull"],
                relationships={"Guard": "neutral"},
                goals=["Investigate the tower"],
            ),
        )
    )


def test_multiple_projects_are_isolated():
    first = create_story("First story")
    second = create_story("Second story")
    projects = db.list_projects()
    assert {project["title"] for project in projects} == {"First story", "Second story"}
    assert first["project"]["id"] != second["project"]["id"]
    assert len(db.get_all_nodes(first["project"]["id"])) == 1
    assert len(db.get_all_nodes(second["project"]["id"])) == 1


def test_manual_node_can_be_saved_without_ai_or_choices():
    state = create_story()
    project_id = state["project"]["id"]
    branch = state["branches"][0]
    root = state["nodes"][0]

    result = services.save_story_node(
        SaveNodeRequest(
            project_id=project_id,
            source_branch_id=branch["id"],
            parent_node_id=root["id"],
            title="The user's scene",
            story_text="The Game Master writes this scene manually without calling a language model.",
            choices=[],
            authoring_mode="manual",
            interaction_type="write",
            applied_delta=StateDelta(inventory_add=["Silver key"]),
        )
    )

    assert result["fork_created"] is False
    assert result["node"]["authoring_mode"] == "manual"
    assert result["node"]["choices"] == []
    assert "Silver key" in result["node"]["memory_snapshot"]["inventory"]


def test_fork_uses_selected_node_snapshot_and_isolates_later_facts():
    state = create_story()
    project_id = state["project"]["id"]
    main_branch = state["branches"][0]
    root = state["nodes"][0]

    main_result = services.save_story_node(
        SaveNodeRequest(
            project_id=project_id,
            source_branch_id=main_branch["id"],
            parent_node_id=root["id"],
            title="Key found",
            story_text="The adventurer finds a silver key inside the gatehouse.",
            applied_delta=StateDelta(inventory_add=["Silver key"]),
            choices=[],
            authoring_mode="manual",
            interaction_type="write",
        )
    )

    fork_result = services.save_story_node(
        SaveNodeRequest(
            project_id=project_id,
            source_branch_id=main_branch["id"],
            parent_node_id=root["id"],
            branch_name="Talk to the guard",
            title="A peaceful conversation",
            story_text="The adventurer remains at the gate and earns the guard's trust.",
            applied_delta=StateDelta(relationships_set={"Guard": "friendly"}),
            choices=[],
            authoring_mode="manual",
            interaction_type="write",
        )
    )

    assert fork_result["fork_created"] is True
    assert "Silver key" in main_result["node"]["memory_snapshot"]["inventory"]
    assert "Silver key" not in fork_result["node"]["memory_snapshot"]["inventory"]
    assert fork_result["node"]["memory_snapshot"]["relationships"]["Guard"] == "friendly"
    assert main_result["node"]["memory_snapshot"]["relationships"]["Guard"] == "neutral"


def _three_choices():
    return [
        {
            "action_type": "do",
            "label": "Take the left path",
        },
        {
            "action_type": "ask",
            "label": "Question the guard",
        },
        {
            "action_type": "continue",
            "label": "Wait quietly",
        },
    ]


def _generation_log_for(state: dict, model: str = "qwen3:1.7b") -> int:
    return db.insert_generation_log(
        request_id="generation_test",
        pipeline_version="test-pipeline",
        project_id=state["project"]["id"],
        branch_id=state["branches"][0]["id"],
        parent_node_id=state["nodes"][0]["id"],
        selected_choice_id=None,
        requested_model=model,
        model=model,
        semantic_check_completed=True,
        continuity_check_completed=True,
        continuity_model="cross-encoder/nli-deberta-v3-small",
        continuity_checked_facts=3,
        continuity_max_contradiction=0.1,
        requested_temperature=0.5,
        effective_temperature=0.5,
        elapsed_seconds=12.0,
        prompt_tokens=50,
        output_tokens=200,
        tokens_per_second=10.0,
        attempts=1,
        interaction_type="continue",
        player_input="",
        gm_instruction="",
        request_payload={"model": model},
        response_payload={"message": {"content": "{}"}},
    )


def test_ai_save_requires_matching_unused_generation_log():
    state = create_story()
    root = state["nodes"][0]
    branch = state["branches"][0]
    log_id = _generation_log_for(state)

    request = SaveNodeRequest(
        project_id=state["project"]["id"],
        source_branch_id=branch["id"],
        parent_node_id=root["id"],
        title="Generated continuation",
        story_text=(
            "The guard finally lowers his spear and unlocks the tower gate. "
            "With a heavy scrape, the iron doors separate and reveal a dark "
            "corridor descending beneath the outer wall. The adventurer steps "
            "through while the guard remains close behind, warning that the "
            "passage has been sealed for many years. Old torch brackets line "
            "the walls, but only a faint blue glow from carved runes provides "
            "light. Dust covers most of the floor except for several recent "
            "footprints leading deeper inside. A cold draft carries the sound "
            "of metal striking stone somewhere ahead. The adventurer now has "
            "to decide whether to inspect the footprints, question the guard "
            "about the noise, or continue carefully into the corridor."
        ),
        choices=_three_choices(),
        authoring_mode="ai",
        interaction_type="continue",
        generated_by_model="qwen3:1.7b",
        generation_log_id=log_id,
    )

    services.save_story_node(request)

    with pytest.raises(ValueError, match="already been saved"):
        services.save_story_node(request)


def test_non_leaf_edit_requires_delta_and_recalculation_updates_descendants():
    state = create_story()
    project_id = state["project"]["id"]
    branch = state["branches"][0]
    root = state["nodes"][0]

    first = services.save_story_node(
        SaveNodeRequest(
            project_id=project_id,
            source_branch_id=branch["id"],
            parent_node_id=root["id"],
            title="Silver key",
            story_text="The adventurer takes a silver key from the gatehouse wall.",
            applied_delta=StateDelta(inventory_add=["Silver key"]),
            choices=[],
            authoring_mode="manual",
            interaction_type="write",
        )
    )["node"]

    second = services.save_story_node(
        SaveNodeRequest(
            project_id=project_id,
            source_branch_id=branch["id"],
            parent_node_id=first["id"],
            title="Guard alliance",
            story_text="The guard agrees to help and becomes a trusted ally.",
            applied_delta=StateDelta(relationships_set={"Guard": "ally"}),
            choices=[],
            authoring_mode="manual",
            interaction_type="write",
        )
    )["node"]

    with pytest.raises(ValueError, match="state-dependent memory"):
        db.update_node_content(
            first["id"],
            "Edited key",
            "The adventurer takes a gold key instead.",
            first["choices"],
        )

    db.update_node_content(
        first["id"],
        "Edited key",
        "The adventurer takes a gold key instead.",
        first["choices"],
        StateDelta(inventory_add=["Gold key"]).model_dump(),
    )

    updated_first = db.get_node(first["id"])
    updated_second = db.get_node(second["id"])
    assert updated_first is not None
    assert updated_second is not None
    assert "Gold key" in updated_first["memory_snapshot"]["inventory"]
    assert "Silver key" not in updated_first["memory_snapshot"]["inventory"]
    assert "Gold key" in updated_second["memory_snapshot"]["inventory"]
    assert updated_second["memory_snapshot"]["relationships"]["Guard"] == "ally"


def test_objective_three_measures_selected_applied_changes():
    state = create_story()
    project_id = state["project"]["id"]
    branch = state["branches"][0]
    root = state["nodes"][0]
    first_choice_id = root["choices"][0]["id"]

    result = services.save_story_node(
        SaveNodeRequest(
            project_id=project_id,
            source_branch_id=branch["id"],
            parent_node_id=root["id"],
            selected_choice_id=first_choice_id,
            title="Clue found",
            story_text="The adventurer follows the clue and finds a silver key.",
            applied_delta=StateDelta(inventory_add=["Silver key"]),
            choices=[],
            authoring_mode="manual",
            interaction_type="do",
            interaction_text="Follow the clue",
        )
    )

    evidence = services.objective_evidence(project_id)["objective_3"]
    assert result["node"]["source_choice_id"] == first_choice_id
    assert evidence["selected_choice_count"] == 1
    assert evidence["selected_choices_with_state_change"] == 1
    assert evidence["state_changing_choice_ratio"] == 1.0


def test_fork_creation_rolls_back_when_node_insert_fails(monkeypatch):
    state = create_story()
    project_id = state["project"]["id"]
    branch = state["branches"][0]
    root = state["nodes"][0]
    new_branch_id = "branch_atomic_test"

    def fail_insert(*args, **kwargs):
        raise RuntimeError("simulated node failure")

    monkeypatch.setattr(db, "_insert_node", fail_insert)

    with pytest.raises(RuntimeError, match="simulated node failure"):
        db.create_fork_with_node(
            branch_id=new_branch_id,
            project_id=project_id,
            branch_name="Atomic fork",
            parent_branch_id=branch["id"],
            fork_node_id=root["id"],
            node_id="node_atomic_test",
            source_choice_id=None,
            source_choice_label=None,
            title="Will fail",
            story_text="This insert is deliberately interrupted.",
            choices=[],
            state_delta={},
            memory_snapshot=root["memory_snapshot"],
            authoring_mode="manual",
            interaction_type="write",
            interaction_text="",
            generated_by_model=None,
        )

    assert db.get_branch(new_branch_id) is None


def test_database_initialisation_does_not_mark_unsaved_logs_as_saved():
    state = create_story()
    log_id = _generation_log_for(state)

    first = db.get_generation_log(log_id)
    assert first is not None
    assert first["validation_pass"] == 1
    assert first["saved_by_user"] == 0

    db.init_db()
    second = db.get_generation_log(log_id)
    assert second is not None
    assert second["saved_by_user"] == 0
