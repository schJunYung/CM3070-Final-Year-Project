from __future__ import annotations

import json

import pytest

from backend import ai_service, services
from backend.content_library import select_prompt_elements
from backend.schemas import GenerateStoryRequest, SaveNodeRequest, StoryChoice


@pytest.mark.asyncio
async def test_parseable_length_failure_is_returned_for_review(monkeypatch):
    parent = {
        "id": "node_parent",
        "project_id": "project_test",
        "branch_id": "branch_test",
        "title": "Starting scene",
        "story_text": (
            "This is a meaningful starting scene with enough distinct words "
            "to pass the low-information guard and provide useful context."
        ),
        "choices": [],
        "memory_snapshot": {
            "location": "Unknown",
            "active_characters": [],
            "inventory": [],
            "relationships": {},
            "goals": [],
            "unresolved_clues": [],
            "decisions": [],
            "threat": "Low",
        },
    }

    monkeypatch.setattr(
        ai_service.db,
        "get_project",
        lambda _project_id: {
            "id": "project_test",
            "title": "Test",
            "genre": "Science Fiction",
            "tone": "Adventurous",
            "world_summary": "A rescue mission.",
            "main_objective": "Rescue survivors",
            "story_element_mode": "off",
            "story_elements": [],
        },
    )
    monkeypatch.setattr(
        ai_service.db,
        "get_branch",
        lambda _branch_id: {
            "id": "branch_test",
            "project_id": "project_test",
            "name": "Main branch",
        },
    )
    monkeypatch.setattr(ai_service.db, "get_node", lambda _node_id: parent)
    monkeypatch.setattr(ai_service.db, "get_branch_path", lambda _branch_id: [parent])
    monkeypatch.setattr(ai_service.db, "get_node_path", lambda _node_id: [parent])
    monkeypatch.setattr(ai_service.db, "get_story_elements_by_ids", lambda _ids: [])
    monkeypatch.setattr(ai_service.db, "insert_generation_attempt", lambda **_kwargs: 1)

    captured_log: dict[str, object] = {}

    def fake_insert_generation_log(**kwargs):
        captured_log.update(kwargs)
        return 42

    monkeypatch.setattr(ai_service.db, "insert_generation_log", fake_insert_generation_log)

    async def fake_retrieve(*_args, **_kwargs):
        return []

    monkeypatch.setattr(ai_service, "retrieve_relevant_nodes", fake_retrieve)

    drafts = iter([46, 41])

    async def fake_call_ollama(_payload, timeout_seconds):
        word_count = next(drafts)
        content = json.dumps(
            {
                "title": "Review candidate",
                "story_text": " ".join(["word"] * word_count),
                "applied_delta": {},
                "choices": [
                    {"action_type": "do", "label": "A"},
                    {"action_type": "say", "label": "B"},
                    {"action_type": "ask", "label": "C"},
                ],
            }
        )
        return {
            "message": {"content": content},
            "eval_count": 100,
            "eval_duration": 1_000_000_000,
            "prompt_eval_count": 50,
        }

    monkeypatch.setattr(ai_service, "call_ollama", fake_call_ollama)

    request = GenerateStoryRequest(
        project_id="project_test",
        branch_id="branch_test",
        parent_node_id="node_parent",
        model=ai_service.FALLBACK_MODEL,
        interaction_type="continue",
    )

    response = await ai_service.generate_story(request)

    assert response.status == "review_required"
    assert response.validation_pass is False
    assert response.generation_log_id == 42
    assert response.validation_issues[0].stage == "length"
    assert len(response.draft.story_text.split()) == 46
    assert captured_log["validation_pass"] is False


def test_rejected_log_requires_explicit_review_override(monkeypatch):
    log = {
        "saved_by_user": 0,
        "validation_pass": 0,
        "project_id": "project_test",
        "branch_id": "branch_test",
        "parent_node_id": "node_parent",
        "selected_choice_id": None,
        "model": "qwen3:1.7b",
    }
    monkeypatch.setattr(services.db, "get_generation_log", lambda _log_id: log)

    choices = [
        StoryChoice(action_type="do", label="A"),
        StoryChoice(action_type="say", label="B"),
        StoryChoice(action_type="ask", label="C"),
    ]

    common = dict(
        project_id="project_test",
        source_branch_id="branch_test",
        parent_node_id="node_parent",
        title="Reviewed",
        story_text="Edited by the Game Master.",
        choices=choices,
        authoring_mode="ai",
        generated_by_model="qwen3:1.7b",
        generation_log_id=7,
    )

    with pytest.raises(ValueError, match="explicit human review"):
        services._validate_generation_log(
            SaveNodeRequest(**common, review_override=False)
        )

    validated = services._validate_generation_log(
        SaveNodeRequest(**common, review_override=True)
    )
    assert validated is log


def test_guided_library_ignores_generic_pack_context():
    project = {
        "story_element_mode": "guided",
        "story_elements": [
            {"element_id": "drone", "preference": "available"},
            {"element_id": "tripwire", "preference": "available"},
        ],
    }
    catalog = [
        {
            "id": "drone",
            "pack": "Science Fiction",
            "category": "creature",
            "name": "Security Drone",
            "tags": ["science", "fiction", "security", "combat"],
            "description": "A hovering drone.",
            "details": {},
            "is_builtin": True,
        },
        {
            "id": "tripwire",
            "pack": "Science Fiction",
            "category": "hazard",
            "name": "Laser Tripwire",
            "tags": ["science", "fiction", "laser", "corridor"],
            "description": "A laser beam.",
            "details": {},
            "is_builtin": True,
        },
    ]

    selected = select_prompt_elements(
        project,
        catalog,
        "Use the Scanner Device to scan the ship for survivors.",
    )

    assert selected == []
