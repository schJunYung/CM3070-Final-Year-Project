from __future__ import annotations

import numpy as np

from backend import continuity_service
from backend.schemas import StateDelta, StoryMemory


def test_memory_to_facts_uses_proposed_state_and_explicit_removals():
    memory = StoryMemory(
        location="Dungeon",
        active_characters=["Player", "Barnaby"],
        inventory=["Sword"],
        relationships={"Barnaby": "ally"},
        goals=["Find the altar"],
        unresolved_clues=["Who built the dungeon?"],
        threat="Medium",
    )

    delta = StateDelta(
        inventory_remove=["Silver key"],
        active_characters_remove=["Guard"],
        goals_complete=["Enter the dungeon"],
        unresolved_clues_resolve=["Where is the entrance?"],
    )

    facts = continuity_service.memory_to_facts(
        memory,
        delta,
    )

    assert "The current location is Dungeon." in facts
    assert "The current inventory contains Sword." in facts
    assert "The current inventory does not contain Silver key." in facts
    assert "Guard is no longer an active character in the scene." in facts
    assert "The goal 'Enter the dungeon' is complete." in facts


def test_review_story_continuity_detects_high_confidence_contradiction(
    monkeypatch,
):
    class FakeModel:
        def predict(self, pairs, **kwargs):
            rows = []
            for _, fact in pairs:
                if fact == "The current location is Dungeon.":
                    rows.append([0.96, 0.02, 0.02])
                else:
                    rows.append([0.03, 0.07, 0.90])
            return np.asarray(rows, dtype=float)

    monkeypatch.setattr(
        continuity_service,
        "get_nli_model",
        lambda: FakeModel(),
    )

    review = continuity_service.review_story_continuity(
        "The party remains in the village and has not entered the dungeon.",
        StoryMemory(location="Dungeon"),
        StateDelta(location_set="Dungeon"),
    )

    assert review.passed is False
    assert review.checked_facts >= 1
    assert len(review.contradictions) == 1
    assert review.contradictions[0].fact == "The current location is Dungeon."
    assert review.max_contradiction_score == 0.96


def test_review_story_continuity_allows_neutral_omitted_facts(
    monkeypatch,
):
    class FakeModel:
        def predict(self, pairs, **kwargs):
            return np.asarray(
                [[0.02, 0.08, 0.90] for _ in pairs],
                dtype=float,
            )

    monkeypatch.setattr(
        continuity_service,
        "get_nli_model",
        lambda: FakeModel(),
    )

    review = continuity_service.review_story_continuity(
        "The player studies an ancient altar without mentioning their equipment.",
        StoryMemory(
            location="Dungeon",
            inventory=["Sword", "Map"],
        ),
        StateDelta(),
    )

    assert review.passed is True
    assert review.contradictions == ()
    assert review.max_contradiction_score == 0.02
