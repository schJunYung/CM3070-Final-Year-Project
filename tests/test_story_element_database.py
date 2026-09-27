from __future__ import annotations

from backend import db,content_library
from backend.schemas import (
    ProjectCreateRequest,
    StoryElementSelection,
    StoryElementWriteRequest,
    StoryMemory,
)



def initialise_temp_database(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        db,
        "DB_PATH",
        tmp_path / "sekai_test.db",
    )

    # Ensure this database is seeded from the current
    # factory JSON rather than a catalogue cached by
    # an earlier test.
    content_library.load_factory_elements.cache_clear()

    db.init_db()


def test_factory_elements_are_seeded_into_sqlite(
    tmp_path,
    monkeypatch,
):
    initialise_temp_database(
        tmp_path,
        monkeypatch,
    )

    elements = db.list_story_elements()

    assert len(elements) >= 10
    assert any(
        element["id"]
        == "fantasy_cave_troll"
        for element in elements
    )
    assert all(
        "details" in element
        for element in elements
    )


def test_builtin_edit_and_reset(
    tmp_path,
    monkeypatch,
):
    initialise_temp_database(
        tmp_path,
        monkeypatch,
    )

    original = db.get_story_element(
        "fantasy_cave_troll"
    )

    assert original is not None

    db.update_story_element(
        original["id"],
        StoryElementWriteRequest(
            pack=original["pack"],
            category=original["category"],
            name="Edited Cave Troll",
            tags=original["tags"],
            threat=original["threat"],
            description=(
                "Temporary edited description."
            ),
            details=original["details"],
        ),
    )

    assert (
        db.get_story_element(
            original["id"]
        )["name"]
        == "Edited Cave Troll"
    )

    db.reset_story_element_defaults(
        original["id"]
    )

    restored = db.get_story_element(
        original["id"]
    )

    assert restored is not None
    assert (
        restored["name"]
        == original["name"]
    )
    assert (
        restored["description"]
        == original["description"]
    )


def test_custom_element_crud(
    tmp_path,
    monkeypatch,
):
    initialise_temp_database(
        tmp_path,
        monkeypatch,
    )

    created = db.create_story_element(
        StoryElementWriteRequest(
            pack="Custom",
            category="creature",
            name="Void Stalker",
            tags=[
                "space",
                "alien",
                "horror",
            ],
            threat="High",
            description=(
                "A translucent predator that "
                "appears near damaged machinery."
            ),
            details={
                "behaviour":
                    "Stalks isolated targets.",
                "attacks": [
                    "Phase strike",
                ],
                "defences": [
                    "Brief intangibility",
                ],
            },
        )
    )

    assert created["is_builtin"] is False

    updated = db.update_story_element(
        created["id"],
        StoryElementWriteRequest(
            pack="Science Horror",
            category="creature",
            name="Void Stalker",
            tags=[
                "space",
                "alien",
            ],
            threat="High",
            description=(
                "A translucent dimensional "
                "predator."
            ),
            details={
                "attacks": [
                    "Phase strike",
                ],
            },
        ),
    )

    assert (
        updated["pack"]
        == "Science Horror"
    )

    assert db.delete_story_element(
        created["id"]
    )

    assert (
        db.get_story_element(
            created["id"]
        )
        is None
    )


def test_project_stores_element_ids_not_full_records(
    tmp_path,
    monkeypatch,
):
    initialise_temp_database(
        tmp_path,
        monkeypatch,
    )

    project = db.create_project(
        ProjectCreateRequest(
            title="Library project",
            genre="Fantasy",
            tone="Tense",
            world_summary=(
                "An abandoned tower."
            ),
            main_objective=(
                "Reach the upper floor."
            ),
            starting_scene_title=(
                "Tower entrance"
            ),
            starting_scene_text=(
                "The party reaches an abandoned "
                "stone tower as night begins to fall."
            ),
            initial_memory=StoryMemory(
                location="Tower entrance"
            ),
            story_element_mode="guided",
            story_elements=[
                StoryElementSelection(
                    element_id=(
                        "fantasy_cave_troll"
                    ),
                    preference="preferred",
                )
            ],
        )
    )

    assert (
        project["project"][
            "story_elements"
        ]
        == [
            {
                "element_id":
                    "fantasy_cave_troll",
                "preference":
                    "preferred",
            }
        ]
    )

    with db.connection() as con:
        rows = con.execute(
            """
            SELECT element_id, preference
            FROM project_story_elements
            WHERE project_id = ?
            """,
            (
                project[
                    "project"
                ]["id"],
            ),
        ).fetchall()

    assert len(rows) == 1
    assert (
        rows[0]["element_id"]
        == "fantasy_cave_troll"
    )


def test_database_catalog_is_ranked_for_generation(
    tmp_path,
    monkeypatch,
):
    from backend.content_library import (
        select_prompt_elements,
    )

    initialise_temp_database(
        tmp_path,
        monkeypatch,
    )

    catalog = db.get_story_elements_by_ids([
        "fantasy_cave_troll",
        "scifi_security_drone",
    ])

    project = {
        "story_element_mode": "guided",
        "story_elements": [
            {
                "element_id":
                    "fantasy_cave_troll",
                "preference":
                    "available",
            },
            {
                "element_id":
                    "scifi_security_drone",
                "preference":
                    "preferred",
            },
        ],
    }

    selected = select_prompt_elements(
        project,
        catalog,
        (
            "Science fiction research station. "
            "Continue through the security deck."
        ),
    )

    assert selected
    assert (
        selected[0]["id"]
        == "scifi_security_drone"
    )
