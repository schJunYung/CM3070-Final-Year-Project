from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from backend import similarity
from backend.ai_service import _interaction_instruction
from backend.validation_service import story_length_issue
from backend.schemas import (
    GenerateStoryRequest,
    StateDelta,
)


def make_request(
    **updates,
) -> GenerateStoryRequest:
    values = {
        "project_id": "project_test",
        "branch_id": "branch_test",
        "parent_node_id": "node_test",
        "interaction_type": "continue",
        "player_input": "",
    }

    values.update(updates)

    return GenerateStoryRequest(
        **values
    )


def test_continue_preserves_optional_direction():
    request = make_request(
        player_input=(
            "Grandma asks the knight "
            "where he came from."
        )
    )

    instruction = _interaction_instruction(
        request,
        selected_choice=None,
    )

    assert "Grandma asks the knight" in instruction

    assert (
        "No specific action was supplied"
        not in instruction
    )


def test_selected_choice_is_mandatory():
    request = make_request()

    instruction = _interaction_instruction(
        request,
        selected_choice={
            "action_type": "ask",
            "label": (
                "Ask the knight about "
                "his homeland"
            ),
        },
    )

    assert "REQUIRED SELECTED CHOICE:" in instruction
    assert "ASK: Ask the knight about his homeland" in instruction
    assert "Resolve the selected choice directly" in instruction


def test_unknown_delta_field_is_rejected():
    with pytest.raises(
        ValidationError
    ):
        StateDelta.model_validate(
            {
                "threat_level_change": "+10%"
            }
        )

def test_state_delta_rejects_conflicting_changes():
    with pytest.raises(
        ValidationError,
        match="conflicting changes",
    ):
        StateDelta.model_validate(
            {
                "inventory_add": [
                    "Silver key"
                ],

                "inventory_remove": [
                    "silver KEY"
                ],
            }
        )

def test_semantic_paraphrase_is_rejected(
    monkeypatch,
):
    async def fake_embed_texts(values):
        assert len(values) == 2

        return [
            [1.0, 0.0],
            [0.99, 0.01],
        ]

    monkeypatch.setattr(
        similarity,
        "embed_texts",
        fake_embed_texts,
    )

    previous = [
        {
            "id": "parent",
            "title": "The Knight's Knock",
            "story_text": (
                "The knight forces the door open "
                "and asks Grandma for guidance. "
                "She freezes and retreats from "
                "him in fear."
            ),
        }
    ]

    candidate = (
        "The warrior pushes through the door "
        "and pleads for Grandma's advice. "
        "She freezes before backing away from "
        "him in fear."
    )

    reason = asyncio.run(
        similarity.semantic_repetition_reason(
            candidate,
            previous,
        )
    )

    assert reason is not None
    assert "semantically" in reason

def test_state_delta_exposes_only_supported_fields():
    expected_fields = {
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
        "threat_set",
    }

    assert (
        set(StateDelta.model_fields)
        == expected_fields
    )

def test_generated_choices_must_be_distinct():
    from backend.schemas import AIGeneratedDraft

    repeated = {
        "title":
            "A scene",

        "story_text":
            "A sufficiently long generated scene "
            "continues the story with a new event.",

        "applied_delta": {},

        "choices": [
            {
                "action_type": "do",
                "label": "Repeat",
            },
            {
                "action_type": "ask",
                "label": "Repeat",
            },
            {
                "action_type": "continue",
                "label": "Different",
            },
        ],
    }

    with pytest.raises(ValidationError, match="distinct labels"):
        AIGeneratedDraft.model_validate(repeated)

def test_story_length_guard_accepts_required_range():
    for word_count in (50, 120, 150):
        valid_story = " ".join(
            f"word{index}"
            for index in range(word_count)
        )

        assert story_length_issue(valid_story) is None


def test_story_length_guard_rejects_short_and_long_scenes():
    too_short = "word " * 49
    too_long = "word " * 151

    short_issue = story_length_issue(too_short)
    long_issue = story_length_issue(too_long)

    assert short_issue is not None
    assert long_issue is not None

    assert "at least 50" in short_issue
    assert "no more than 150" in long_issue



def test_generated_contract_requires_explicit_delta_and_choice_fields():
    from backend.schemas import (
        AIGeneratedDraft,
    )

    missing_applied_delta = {
        "title": "A scene",

        "story_text": (
            "A sufficiently long generated scene "
            "continues with a new event."
        ),

        "choices": [
            {
                "action_type": "do",
                "label": "Inspect",
            },
            {
                "action_type": "ask",
                "label": "Question",
            },
            {
                "action_type": "continue",
                "label": "Wait",
            },
        ],
    }

    with pytest.raises(
        ValidationError,
        match="applied_delta",
    ):
        AIGeneratedDraft.model_validate(
            missing_applied_delta
        )

    missing_action_type = {
        **missing_applied_delta,
        "applied_delta": {},
    }

    missing_action_type["choices"] = [
        dict(choice)
        for choice
        in missing_applied_delta["choices"]
    ]

    missing_action_type[
        "choices"
    ][0].pop(
        "action_type"
    )

    with pytest.raises(
        ValidationError,
        match="action_type",
    ):
        AIGeneratedDraft.model_validate(
            missing_action_type
        )

def _valid_generated_payload():
    import json

    return {
        "message": {
            "content": json.dumps(
                {
                    "title": "A new chamber",
                    "story_text": (
                        "The hidden wall slides aside after the selected action, revealing a chamber "
                        "that was not visible from the corridor. Dust spills across the floor as the "
                        "adventurer and guard step through the opening. At the centre, a bronze "
                        "mechanism turns beneath a cracked stone pedestal, while pale runes pulse "
                        "around its rim. A narrow passage continues beyond the mechanism, but each "
                        "rotation makes the floor tremble more violently. The guard warns that the "
                        "device may be controlling the sealed doors deeper inside the tower. Fresh "
                        "scrape marks on the pedestal suggest that somebody activated it recently, "
                        "although no footprints remain in the dust. The party must decide whether to "
                        "inspect the mechanism, question the guard, or wait and observe its next cycle."
                    ),
                    "applied_delta": {
                        "location_set": "Hidden chamber",
                        "unresolved_clues_add": ["Who activated the mechanism?"],
                    },
                    "choices": [
                        {
                            "action_type": "do",
                            "label":
                                "Inspect the mechanism",
                        },

                        {
                            "action_type": "ask",
                            "label":
                                "Ask the guard about the chamber",
                        },

                        {
                            "action_type": "continue",
                            "label":
                                "Wait for the mechanism to stop",
                        },
                    ],
                }
            )
        },
        "prompt_eval_count": 50,
        "eval_count": 120,
        "eval_duration": 12_000_000_000,
    }


def _create_generation_story():
    from backend import db
    from backend.schemas import ProjectCreateRequest, StoryMemory

    return db.create_project(
        ProjectCreateRequest(
            title="Generation test",
            genre="Fantasy",
            tone="Mysterious",
            world_summary="A guarded ruin contains hidden rooms.",
            main_objective="Explore the ruin.",
            starting_scene_title="At the sealed wall",
            starting_scene_text=(
                "The adventurer stands beside a sealed wall while a guard "
                "watches the ancient markings begin to glow."
            ),
            initial_memory=StoryMemory(
                location="Sealed wall",
                active_characters=["Adventurer", "Guard"],
            ),
        )
    )


def test_generation_retries_invalid_structure_then_logs_success(
    tmp_path,
    monkeypatch,
):
    from backend import ai_service, db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "generation.db")
    db.init_db()
    state = _create_generation_story()
    calls = 0

    async def fake_call_ollama(payload, timeout_seconds):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"message": {"content": '{"title": "Incomplete"}'}}
        return _valid_generated_payload()

    async def no_semantic_repeat(candidate_story, reference_nodes):
        return None

    monkeypatch.setattr(ai_service, "call_ollama", fake_call_ollama)
    monkeypatch.setattr(
        ai_service,
        "semantic_repetition_reason",
        no_semantic_repeat,
    )

    request = GenerateStoryRequest(
        project_id=state["project"]["id"],
        branch_id=state["branches"][0]["id"],
        parent_node_id=state["nodes"][0]["id"],
        model="qwen3.5:2b",
        interaction_type="continue",
    )

    result = asyncio.run(ai_service.generate_story(request))
    log = db.get_generation_log(result.generation_log_id)

    attempts = db.get_generation_attempts(log["request_id"])

    assert len(attempts) == 2

    assert attempts[0]["outcome"] == "rejected"
    assert attempts[0]["failure_stage"] == "schema"

    assert attempts[1]["outcome"] == "accepted"
    assert attempts[1]["story_word_count"] is not None

    assert result.metrics.attempts == 2
    assert result.metrics.model == "qwen3.5:2b"
    assert result.metrics.semantic_check_completed is True
    assert log is not None
    assert log["validation_pass"] == 1
    assert log["saved_by_user"] == 0
    assert log["semantic_check_completed"] == 1


def test_generation_timeout_moves_to_fallback(
    tmp_path,
    monkeypatch,
):
    from backend import ai_service, db
    from fastapi import HTTPException

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "fallback.db")
    db.init_db()
    state = _create_generation_story()
    called_models = []

    async def fake_call_ollama(payload, timeout_seconds):
        called_models.append(payload["model"])
        if payload["model"] == "qwen3.5:4b":
            raise HTTPException(status_code=504, detail="simulated timeout")
        return _valid_generated_payload()

    async def no_semantic_repeat(candidate_story, reference_nodes):
        return None

    monkeypatch.setattr(ai_service, "call_ollama", fake_call_ollama)
    monkeypatch.setattr(
        ai_service,
        "semantic_repetition_reason",
        no_semantic_repeat,
    )

    request = GenerateStoryRequest(
        project_id=state["project"]["id"],
        branch_id=state["branches"][0]["id"],
        parent_node_id=state["nodes"][0]["id"],
        model="qwen3.5:4b",
        interaction_type="continue",
    )

    result = asyncio.run(ai_service.generate_story(request))

    assert called_models == ["qwen3.5:4b", "qwen3:1.7b"]
    assert result.metrics.model == "qwen3:1.7b"
    assert result.metrics.attempts == 2


def test_generation_retries_when_nli_reviewer_detects_contradiction(
    tmp_path,
    monkeypatch,
):
    from backend import ai_service, db
    from backend.continuity_service import (
        ContinuityContradiction,
        ContinuityReview,
    )

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "nli-retry.db")
    db.init_db()
    state = _create_generation_story()

    async def fake_call_ollama(payload, timeout_seconds):
        return _valid_generated_payload()

    async def no_semantic_repeat(candidate_story, reference_nodes):
        return None

    review_calls = 0

    def fake_continuity_review(candidate_story, proposed_memory, applied_delta):
        nonlocal review_calls
        review_calls += 1

        if review_calls == 1:
            return ContinuityReview(
                model="cross-encoder/nli-deberta-v3-small",
                checked_facts=3,
                contradictions=(
                    ContinuityContradiction(
                        fact="The current location is Hidden chamber.",
                        contradiction_score=0.97,
                        entailment_score=0.01,
                        neutral_score=0.02,
                    ),
                ),
                max_contradiction_score=0.97,
            )

        return ContinuityReview(
            model="cross-encoder/nli-deberta-v3-small",
            checked_facts=3,
            contradictions=(),
            max_contradiction_score=0.12,
        )

    monkeypatch.setattr(ai_service, "call_ollama", fake_call_ollama)
    monkeypatch.setattr(
        ai_service,
        "semantic_repetition_reason",
        no_semantic_repeat,
    )
    monkeypatch.setattr(
        ai_service,
        "review_story_continuity",
        fake_continuity_review,
    )

    request = GenerateStoryRequest(
        project_id=state["project"]["id"],
        branch_id=state["branches"][0]["id"],
        parent_node_id=state["nodes"][0]["id"],
        model="qwen3:1.7b",
        interaction_type="continue",
    )

    result = asyncio.run(ai_service.generate_story(request))
    log = db.get_generation_log(result.generation_log_id)

    assert review_calls == 2
    assert result.metrics.attempts == 2
    assert result.metrics.continuity_check_completed is True
    assert result.metrics.continuity_model == "cross-encoder/nli-deberta-v3-small"
    assert result.metrics.continuity_checked_facts == 3
    assert result.metrics.continuity_max_contradiction == 0.12
    assert log is not None
    assert log["continuity_check_completed"] == 1
    assert log["continuity_checked_facts"] == 3
