from __future__ import annotations

import json
from pathlib import Path

import pytest


from backend import content_library, db
from backend.schemas import (
    ProjectCreateRequest,
    StoryElementSelection,
    StoryElementWriteRequest,
    StoryMemory,
)

@pytest.fixture(autouse=True)
def clear_factory_element_cache():
    """
    Prevent tests that monkeypatch CATALOG_PATH from leaking
    cached factory elements into later tests.
    """
    content_library.load_factory_elements.cache_clear()

    yield

    content_library.load_factory_elements.cache_clear()
    
@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()


def _write_catalog(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "fantasy_troll",
                    "pack": "Fantasy",
                    "category": "creature",
                    "name": "Cave Troll",
                    "tags": ["fantasy", "tower", "dungeon"],
                    "threat": "High",
                    "description": "A hulking troll.",
                    "details": {"attacks": ["Stone club"]},
                },
                {
                    "id": "scifi_drone",
                    "pack": "Science Fiction",
                    "category": "creature",
                    "name": "Security Drone",
                    "tags": ["science", "fiction", "station", "security"],
                    "threat": "Medium",
                    "description": "A hovering security drone.",
                    "details": {"attacks": ["Stun pulse"]},
                },
            ]
        ),
        encoding="utf-8",
    )


def test_scifi_context_prefers_scifi_element(
    tmp_path,
    monkeypatch,
):
    catalogue = tmp_path / "elements.json"
    _write_catalog(catalogue)

    monkeypatch.setattr(
        content_library,
        "CATALOG_PATH",
        catalogue,
    )

    content_library.load_factory_elements.cache_clear()

    catalog = [
        element.model_dump()
        for element
        in content_library.load_factory_elements()
    ]

    project = {
        "story_element_mode": "guided",
        "story_elements": [
            {
                "element_id": "fantasy_troll",
                "preference": "available",
            },
            {
                "element_id": "scifi_drone",
                "preference": "preferred",
            },
        ],
    }

    selected = content_library.select_prompt_elements(
        project,
        catalog,
        (
            "Science fiction research station. "
            "Continue into the security deck."
        ),
    )

    assert selected
    assert selected[0]["id"] == "scifi_drone"

def test_off_mode_sends_no_library_elements(
    tmp_path,
    monkeypatch,
):
    catalogue = tmp_path / "elements.json"
    _write_catalog(catalogue)

    monkeypatch.setattr(
        content_library,
        "CATALOG_PATH",
        catalogue,
    )

    content_library.load_factory_elements.cache_clear()

    catalog = [
        element.model_dump()
        for element
        in content_library.load_factory_elements()
    ]

    selected = content_library.select_prompt_elements(
        {
            "story_element_mode": "off",
            "story_elements": [
                {
                    "element_id": "fantasy_troll",
                    "preference": "preferred",
                }
            ],
        },
        catalog,
        "A fantasy tower.",
    )

    assert selected == []




def test_story_element_settings_are_persisted_with_project():
    custom_element = db.create_story_element(
        StoryElementWriteRequest(
            pack="Custom",
            category="creature",
            name="Void Stalker",
            tags=[
                "alien",
                "station",
            ],
            threat="High",
            description=(
                "A translucent predator that appears near "
                "unstable warp machinery."
            ),
            details={
                "attacks": [
                    "Phase strike",
                ],
            },
        )
    )

    state = db.create_project(
        ProjectCreateRequest(
            title="Configured story",
            genre="Science Fiction",
            tone="Tense",
            world_summary=(
                "A derelict orbital research station."
            ),
            main_objective=(
                "Reach the command deck."
            ),
            starting_scene_title="Airlock",
            starting_scene_text=(
                "The crew enters through a damaged airlock "
                "while emergency lights pulse along the "
                "abandoned research station."
            ),
            initial_memory=StoryMemory(
                location="Airlock"
            ),
            story_element_mode="strict",
            story_elements=[
                StoryElementSelection(
                    element_id=custom_element["id"],
                    preference="preferred",
                ),
            ],
        )
    )

    project = state["project"]

    assert project["story_element_mode"] == "strict"

    selections = {
        selection["element_id"]: selection["preference"]
        for selection in project["story_elements"]
    }

    assert selections == {
        custom_element["id"]: "preferred",
    }

    stored_custom = db.get_story_element(
        custom_element["id"]
    )

    assert stored_custom is not None
    assert stored_custom["name"] == "Void Stalker"
    assert stored_custom["category"] == "creature"
    assert stored_custom["threat"] == "High"
    assert stored_custom["tags"] == [
        "alien",
        "station",
    ]
    assert stored_custom["details"] == {
        "attacks": [
            "Phase strike",
        ],
    }