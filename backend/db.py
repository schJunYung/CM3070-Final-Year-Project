from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from .schemas import (
    CustomStoryElement,
    ProjectCreateRequest,
    ProjectUpdateRequest,
    StateDelta,
    StoryChoice,
    StoryElementDefinition,
    StoryElementSelection,
    StoryElementWriteRequest,
    StoryMemory,
)
from .content_library import load_factory_elements
from .state import apply_state_delta


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DB_PATH = (
    DATA_DIR / "sekai_workspace.db"
)

DB_PATH = Path(
    os.getenv(
        "SEKAI_DB_PATH",
        str(DEFAULT_DB_PATH),
    )
).expanduser()

def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()



def _ensure_column(
    con: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> bool:
    """Add a missing column and report whether a migration occurred."""
    columns = {
        row["name"]
        for row in con.execute(f"PRAGMA table_info({table})")
    }
    if column in columns:
        return False

    con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    return True


def init_db() -> None:
    with connection() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                genre TEXT NOT NULL,
                tone TEXT NOT NULL,
                world_summary TEXT NOT NULL,
                main_objective TEXT NOT NULL,
                story_element_mode TEXT NOT NULL DEFAULT 'guided',
                story_elements_json TEXT NOT NULL DEFAULT '[]',
                custom_story_elements_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS story_element_defaults (
                id TEXT PRIMARY KEY,
                pack TEXT NOT NULL,
                category TEXT NOT NULL,
                name TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                threat TEXT,
                description TEXT NOT NULL DEFAULT '',
                details_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS story_elements (
                id TEXT PRIMARY KEY,
                pack TEXT NOT NULL,
                category TEXT NOT NULL,
                name TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                threat TEXT,
                description TEXT NOT NULL DEFAULT '',
                details_json TEXT NOT NULL DEFAULT '{}',
                is_builtin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );

            CREATE TABLE IF NOT EXISTS project_story_elements (
                project_id TEXT NOT NULL
                    REFERENCES projects(id) ON DELETE CASCADE,
                element_id TEXT NOT NULL
                    REFERENCES story_elements(id) ON DELETE CASCADE,
                preference TEXT NOT NULL DEFAULT 'available',
                PRIMARY KEY (project_id, element_id)
            );

            CREATE TABLE IF NOT EXISTS branches (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                parent_branch_id TEXT REFERENCES branches(id) ON DELETE SET NULL,
                fork_node_id TEXT,
                head_node_id TEXT,
                created_at TEXT NOT NULL,
                deleted_at TEXT,
                deletion_batch_id TEXT,
                deleted_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS image_generation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL UNIQUE,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                source_kind TEXT NOT NULL,
                model_key TEXT NOT NULL,
                model_id TEXT NOT NULL,
                device TEXT NOT NULL,
                precision TEXT NOT NULL,
                prompt TEXT NOT NULL,
                negative_prompt TEXT NOT NULL DEFAULT '',
                seed INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                steps INTEGER NOT NULL,
                guidance_scale REAL NOT NULL,
                model_load_seconds REAL NOT NULL,
                inference_seconds REAL NOT NULL,
                total_seconds REAL NOT NULL,
                cold_start INTEGER NOT NULL DEFAULT 0,
                file_path TEXT NOT NULL,
                image_url TEXT NOT NULL,
                selected_by_user INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
                parent_node_id TEXT REFERENCES nodes(id) ON DELETE RESTRICT,
                source_choice_id TEXT,
                source_choice_label TEXT,
                title TEXT NOT NULL,
                story_text TEXT NOT NULL,
                choices_json TEXT NOT NULL,
                state_delta_json TEXT NOT NULL,
                memory_snapshot_json TEXT NOT NULL,
                authoring_mode TEXT NOT NULL DEFAULT 'manual',
                interaction_type TEXT NOT NULL DEFAULT 'write',
                interaction_text TEXT NOT NULL DEFAULT '',
                generated_by_model TEXT,
                generation_log_id INTEGER,
                scene_image_log_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT,
                deletion_batch_id TEXT,
                deleted_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS generation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                pipeline_version TEXT NOT NULL DEFAULT 'legacy',
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL,
                parent_node_id TEXT NOT NULL,
                selected_choice_id TEXT,
                requested_model TEXT NOT NULL,
                model TEXT NOT NULL,
                requested_temperature REAL NOT NULL,
                effective_temperature REAL NOT NULL,
                elapsed_seconds REAL NOT NULL,
                prompt_tokens INTEGER,
                output_tokens INTEGER,
                tokens_per_second REAL,
                attempts INTEGER NOT NULL,
                validation_pass INTEGER NOT NULL DEFAULT 1,
                saved_by_user INTEGER NOT NULL DEFAULT 0,
                semantic_check_completed INTEGER NOT NULL DEFAULT 0,
                continuity_check_completed INTEGER NOT NULL DEFAULT 0,
                continuity_model TEXT,
                continuity_checked_facts INTEGER NOT NULL DEFAULT 0,
                continuity_max_contradiction REAL,
                interaction_type TEXT NOT NULL,
                player_input TEXT NOT NULL,
                gm_instruction TEXT NOT NULL,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );


            CREATE TABLE IF NOT EXISTS transcription_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
                engine TEXT NOT NULL,
                model TEXT NOT NULL,
                device TEXT NOT NULL,
                compute_type TEXT NOT NULL,
                target_field TEXT NOT NULL,
                audio_content_type TEXT,
                audio_bytes INTEGER NOT NULL,
                audio_seconds REAL NOT NULL,
                model_load_seconds REAL NOT NULL,
                inference_seconds REAL NOT NULL,
                total_seconds REAL NOT NULL,
                real_time_factor REAL,
                cold_start INTEGER NOT NULL DEFAULT 0,
                language TEXT,
                language_probability REAL,
                transcript TEXT NOT NULL,
                reviewed_text TEXT,
                reviewed_at TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS generation_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                pipeline_version TEXT NOT NULL,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL,
                parent_node_id TEXT NOT NULL,
                selected_choice_id TEXT,
                attempt_number INTEGER NOT NULL,
                model TEXT NOT NULL,
                temperature REAL NOT NULL,
                elapsed_seconds REAL NOT NULL,
                outcome TEXT NOT NULL,
                failure_stage TEXT,
                failure_detail TEXT,
                story_word_count INTEGER,
                created_at TEXT NOT NULL
            );




            CREATE INDEX IF NOT EXISTS idx_branches_project ON branches(project_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_project ON nodes(project_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_node_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_branch ON nodes(branch_id);
            CREATE INDEX IF NOT EXISTS idx_logs_project ON generation_logs(project_id);
            CREATE INDEX IF NOT EXISTS idx_attempts_project ON generation_attempts(project_id);
            CREATE INDEX IF NOT EXISTS idx_attempts_request ON generation_attempts(request_id);

            CREATE INDEX IF NOT EXISTS idx_transcription_logs_project
                ON transcription_logs(project_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_image_generation_logs_project
                ON image_generation_logs(project_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_story_elements_pack
                ON story_elements(pack, category, name);
            CREATE INDEX IF NOT EXISTS idx_story_elements_deleted
                ON story_elements(deleted_at);
            CREATE INDEX IF NOT EXISTS idx_project_story_elements_project
                ON project_story_elements(project_id);
            """
        )

        # Existing databases may predate the configurable Story Element Library.
        # CREATE TABLE IF NOT EXISTS does not add columns to an existing table.
        _ensure_column(
            con,
            "projects",
            "story_element_mode",
            "TEXT NOT NULL DEFAULT 'guided'",
        )
        _ensure_column(
            con,
            "projects",
            "story_elements_json",
            "TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            con,
            "projects",
            "custom_story_elements_json",
            "TEXT NOT NULL DEFAULT '[]'",
        )

        # Seed the editable library and immutable reset snapshot, then migrate
        # project-embedded story elements from SekAI v5 into normalized tables.
        _seed_story_element_tables(con)
        _migrate_legacy_project_story_elements(con)

        # Existing databases were created before the trash feature.
        # CREATE TABLE IF NOT EXISTS does not add new columns, so migrate them.
        _ensure_column(con, "branches", "deleted_at", "TEXT")
        _ensure_column(con, "branches", "deletion_batch_id", "TEXT")
        _ensure_column(con, "branches", "deleted_reason", "TEXT")
        _ensure_column(con, "nodes", "deleted_at", "TEXT")
        _ensure_column(con, "nodes", "deletion_batch_id", "TEXT")
        _ensure_column(con, "nodes", "deleted_reason", "TEXT")
        _ensure_column(con, "nodes", "generation_log_id","INTEGER",)
        _ensure_column(
            con,
            "nodes",
            "scene_image_log_id",
            "INTEGER",
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_nodes_generation_log " "ON nodes(generation_log_id)")
        saved_by_user_added = _ensure_column(
            con,
            "generation_logs",
            "saved_by_user",
            "INTEGER NOT NULL DEFAULT 0",
        )

        if saved_by_user_added:
            # Earlier versions used validation_pass to mean "saved by user".
            # Preserve that information once, then restore validation_pass to
            # its intended meaning: the response passed backend validation.
            con.execute(
                """
                UPDATE generation_logs
                SET saved_by_user = validation_pass
                """
            )
            con.execute("UPDATE generation_logs SET validation_pass = 1")

        requested_model_added = _ensure_column(
            con,
            "generation_logs",
            "requested_model",
            "TEXT",
        )

        
        if requested_model_added:
            con.execute(
                "UPDATE generation_logs SET requested_model = model"
            )

        _ensure_column(
            con,
            "generation_logs",
            "semantic_check_completed",
            "INTEGER NOT NULL DEFAULT 0",
        )
        _ensure_column(
            con,
            "generation_logs",
            "continuity_check_completed",
            "INTEGER NOT NULL DEFAULT 0",
        )
        _ensure_column(
            con,
            "generation_logs",
            "continuity_model",
            "TEXT",
        )
        _ensure_column(
            con,
            "generation_logs",
            "continuity_checked_facts",
            "INTEGER NOT NULL DEFAULT 0",
        )
        _ensure_column(
            con,
            "generation_logs",
            "continuity_max_contradiction",
            "REAL",
        )

        _ensure_column(
            con,
            "generation_logs",
            "request_id",
            "TEXT",
        )

        _ensure_column(
            con,
            "generation_logs",
            "pipeline_version",
            "TEXT NOT NULL DEFAULT 'legacy'",
        )

        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_branches_deleted "
            "ON branches(project_id, deleted_at)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_deleted "
            "ON nodes(project_id, deleted_at)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_deletion_batch "
            "ON nodes(deletion_batch_id)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_branches_deletion_batch "
            "ON branches(deletion_batch_id)"
        )


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _story_element_row_payload(
    element: StoryElementDefinition | StoryElementWriteRequest,
) -> tuple[str, str, str, str, str | None, str, str]:
    return (
        element.pack,
        element.category,
        element.name,
        json_dump(element.tags),
        element.threat,
        element.description,
        json_dump(element.details),
    )


def _decode_story_element(
    row: sqlite3.Row | dict[str, Any],
) -> dict[str, Any]:
    element = dict(row)
    element["tags"] = json_load(
        element.pop("tags_json", None),
        [],
    )
    element["details"] = json_load(
        element.pop("details_json", None),
        {},
    )
    element["is_builtin"] = bool(
        element.get("is_builtin", 0)
    )
    return element


def _seed_story_element_tables(
    con: sqlite3.Connection,
) -> None:
    """
    Seed immutable defaults and the editable library.

    INSERT OR IGNORE is deliberate: normal startup must never overwrite a
    user's edits or revive a deleted built-in. Reset endpoints do that
    explicitly by copying values from story_element_defaults.
    """
    now = utc_now()

    for element in load_factory_elements():
        (
            pack,
            category,
            name,
            tags_json,
            threat,
            description,
            details_json,
        ) = _story_element_row_payload(element)

        con.execute(
            """
            INSERT OR IGNORE INTO story_element_defaults
                (
                    id, pack, category, name, tags_json,
                    threat, description, details_json
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                element.id,
                pack,
                category,
                name,
                tags_json,
                threat,
                description,
                details_json,
            ),
        )

        con.execute(
            """
            INSERT OR IGNORE INTO story_elements
                (
                    id, pack, category, name, tags_json,
                    threat, description, details_json,
                    is_builtin, created_at, updated_at, deleted_at
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, NULL)
            """,
            (
                element.id,
                pack,
                category,
                name,
                tags_json,
                threat,
                description,
                details_json,
                now,
                now,
            ),
        )


def _migrate_legacy_project_story_elements(
    con: sqlite3.Connection,
) -> None:
    """
    Move SekAI v5 project-embedded selections/custom records into normalized
    database tables. The legacy JSON columns are cleared after successful
    migration so removed selections are not re-imported on the next startup.
    """
    rows = con.execute(
        """
        SELECT
            id,
            story_elements_json,
            custom_story_elements_json
        FROM projects
        """
    ).fetchall()

    now = utc_now()

    for row in rows:
        project_id = row["id"]
        selections = json_load(
            row["story_elements_json"],
            [],
        )
        custom_elements = json_load(
            row["custom_story_elements_json"],
            [],
        )

        for raw in custom_elements:
            try:
                legacy = CustomStoryElement.model_validate(raw)
            except ValueError:
                continue

            definition = StoryElementDefinition.model_validate(
                legacy.model_dump(
                    exclude={"preference"}
                )
            )

            (
                pack,
                category,
                name,
                tags_json,
                threat,
                description,
                details_json,
            ) = _story_element_row_payload(definition)

            con.execute(
                """
                INSERT OR IGNORE INTO story_elements
                    (
                        id, pack, category, name, tags_json,
                        threat, description, details_json,
                        is_builtin, created_at, updated_at, deleted_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL)
                """,
                (
                    definition.id,
                    pack,
                    category,
                    name,
                    tags_json,
                    threat,
                    description,
                    details_json,
                    now,
                    now,
                ),
            )

            con.execute(
                """
                INSERT OR IGNORE INTO project_story_elements
                    (project_id, element_id, preference)
                VALUES (?, ?, ?)
                """,
                (
                    project_id,
                    definition.id,
                    legacy.preference,
                ),
            )

        for raw in selections:
            try:
                selection = StoryElementSelection.model_validate(raw)
            except ValueError:
                continue

            exists = con.execute(
                """
                SELECT 1
                FROM story_elements
                WHERE id = ?
                """,
                (selection.element_id,),
            ).fetchone()

            if not exists:
                continue

            con.execute(
                """
                INSERT OR IGNORE INTO project_story_elements
                    (project_id, element_id, preference)
                VALUES (?, ?, ?)
                """,
                (
                    project_id,
                    selection.element_id,
                    selection.preference,
                ),
            )

        if selections or custom_elements:
            con.execute(
                """
                UPDATE projects
                SET
                    story_elements_json = '[]',
                    custom_story_elements_json = '[]'
                WHERE id = ?
                """,
                (project_id,),
            )


def _replace_project_story_elements(
    con: sqlite3.Connection,
    project_id: str,
    selections: list[StoryElementSelection],
) -> None:
    con.execute(
        "DELETE FROM project_story_elements WHERE project_id = ?",
        (project_id,),
    )

    for selection in selections:
        exists = con.execute(
            "SELECT 1 FROM story_elements WHERE id = ?",
            (selection.element_id,),
        ).fetchone()

        if not exists:
            raise ValueError(
                "Selected story element does not exist: "
                f"{selection.element_id}"
            )

        con.execute(
            """
            INSERT INTO project_story_elements
                (project_id, element_id, preference)
            VALUES (?, ?, ?)
            """,
            (
                project_id,
                selection.element_id,
                selection.preference,
            ),
        )


def get_project_story_element_selections(
    project_id: str,
) -> list[dict[str, Any]]:
    with connection() as con:
        rows = con.execute(
            """
            SELECT element_id, preference
            FROM project_story_elements
            WHERE project_id = ?
            ORDER BY element_id
            """,
            (project_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def list_story_elements(
    *,
    include_deleted: bool = False,
) -> list[dict[str, Any]]:
    where_clause = (
        ""
        if include_deleted
        else "WHERE deleted_at IS NULL"
    )

    with connection() as con:
        rows = con.execute(
            f"""
            SELECT *
            FROM story_elements
            {where_clause}
            ORDER BY
                pack COLLATE NOCASE,
                category COLLATE NOCASE,
                name COLLATE NOCASE
            """
        ).fetchall()

    return [
        _decode_story_element(row)
        for row in rows
    ]


def get_story_element(
    element_id: str,
    *,
    include_deleted: bool = False,
) -> dict[str, Any] | None:
    with connection() as con:
        row = con.execute(
            """
            SELECT *
            FROM story_elements
            WHERE id = ?
            """
            + (
                ""
                if include_deleted
                else " AND deleted_at IS NULL"
            ),
            (element_id,),
        ).fetchone()

    return (
        _decode_story_element(row)
        if row
        else None
    )


def get_story_elements_by_ids(
    element_ids: list[str],
) -> list[dict[str, Any]]:
    unique_ids = list(dict.fromkeys(
        element_id
        for element_id in element_ids
        if element_id
    ))

    if not unique_ids:
        return []

    placeholders = ",".join(
        "?"
        for _ in unique_ids
    )

    with connection() as con:
        rows = con.execute(
            f"""
            SELECT *
            FROM story_elements
            WHERE
                deleted_at IS NULL
                AND id IN ({placeholders})
            """,
            tuple(unique_ids),
        ).fetchall()

    by_id = {
        row["id"]: _decode_story_element(row)
        for row in rows
    }

    return [
        by_id[element_id]
        for element_id in unique_ids
        if element_id in by_id
    ]


def create_story_element(
    request: StoryElementWriteRequest,
) -> dict[str, Any]:
    element_id = f"element_{uuid4().hex}"
    now = utc_now()

    (
        pack,
        category,
        name,
        tags_json,
        threat,
        description,
        details_json,
    ) = _story_element_row_payload(request)

    with connection() as con:
        con.execute(
            """
            INSERT INTO story_elements
                (
                    id, pack, category, name, tags_json,
                    threat, description, details_json,
                    is_builtin, created_at, updated_at, deleted_at
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL)
            """,
            (
                element_id,
                pack,
                category,
                name,
                tags_json,
                threat,
                description,
                details_json,
                now,
                now,
            ),
        )

    element = get_story_element(element_id)
    if element is None:
        raise RuntimeError(
            "Story element was not found after creation."
        )
    return element


def update_story_element(
    element_id: str,
    request: StoryElementWriteRequest,
) -> dict[str, Any]:
    existing = get_story_element(element_id)

    if not existing:
        raise ValueError("Story element not found.")

    (
        pack,
        category,
        name,
        tags_json,
        threat,
        description,
        details_json,
    ) = _story_element_row_payload(request)

    with connection() as con:
        con.execute(
            """
            UPDATE story_elements
            SET
                pack = ?,
                category = ?,
                name = ?,
                tags_json = ?,
                threat = ?,
                description = ?,
                details_json = ?,
                updated_at = ?
            WHERE id = ? AND deleted_at IS NULL
            """,
            (
                pack,
                category,
                name,
                tags_json,
                threat,
                description,
                details_json,
                utc_now(),
                element_id,
            ),
        )

    updated = get_story_element(element_id)
    if updated is None:
        raise RuntimeError(
            "Story element was not found after update."
        )
    return updated


def delete_story_element(
    element_id: str,
) -> bool:
    with connection() as con:
        cursor = con.execute(
            """
            UPDATE story_elements
            SET deleted_at = ?, updated_at = ?
            WHERE id = ? AND deleted_at IS NULL
            """,
            (
                utc_now(),
                utc_now(),
                element_id,
            ),
        )

    return cursor.rowcount > 0


def reset_story_element_defaults(
    element_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Restore factory values for one or all built-in elements.

    User-created elements are deliberately untouched.
    """
    with connection() as con:
        if element_id is None:
            defaults = con.execute(
                """
                SELECT *
                FROM story_element_defaults
                ORDER BY id
                """
            ).fetchall()
        else:
            defaults = con.execute(
                """
                SELECT *
                FROM story_element_defaults
                WHERE id = ?
                """,
                (element_id,),
            ).fetchall()

        if element_id is not None and not defaults:
            raise ValueError(
                "This story element has no factory default."
            )

        now = utc_now()

        for row in defaults:
            con.execute(
                """
                INSERT INTO story_elements
                    (
                        id, pack, category, name, tags_json,
                        threat, description, details_json,
                        is_builtin, created_at, updated_at, deleted_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, NULL)
                ON CONFLICT(id) DO UPDATE SET
                    pack = excluded.pack,
                    category = excluded.category,
                    name = excluded.name,
                    tags_json = excluded.tags_json,
                    threat = excluded.threat,
                    description = excluded.description,
                    details_json = excluded.details_json,
                    is_builtin = 1,
                    updated_at = excluded.updated_at,
                    deleted_at = NULL
                """,
                (
                    row["id"],
                    row["pack"],
                    row["category"],
                    row["name"],
                    row["tags_json"],
                    row["threat"],
                    row["description"],
                    row["details_json"],
                    now,
                    now,
                ),
            )

    if element_id is not None:
        element = get_story_element(element_id)
        return [element] if element else []

    return list_story_elements()




def insert_image_generation_log(
    *,
    image_id: str,
    project_id: str,
    source_kind: str,
    model_key: str,
    model_id: str,
    device: str,
    precision: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    model_load_seconds: float,
    inference_seconds: float,
    total_seconds: float,
    cold_start: bool,
    file_path: str,
    image_url: str,
) -> int:
    with connection() as con:
        cursor = con.execute(
            """
            INSERT INTO image_generation_logs
                (
                    image_id, project_id, source_kind, model_key, model_id,
                    device, precision, prompt, negative_prompt, seed,
                    width, height, steps, guidance_scale,
                    model_load_seconds, inference_seconds, total_seconds,
                    cold_start, file_path, image_url, created_at
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                project_id,
                source_kind,
                model_key,
                model_id,
                device,
                precision,
                prompt,
                negative_prompt,
                seed,
                width,
                height,
                steps,
                guidance_scale,
                model_load_seconds,
                inference_seconds,
                total_seconds,
                int(cold_start),
                file_path,
                image_url,
                utc_now(),
            ),
        )
        return int(cursor.lastrowid)


def get_image_generation_log(log_id: int) -> dict[str, Any] | None:
    with connection() as con:
        row = con.execute(
            "SELECT * FROM image_generation_logs WHERE id = ?",
            (log_id,),
        ).fetchone()
    return dict(row) if row else None


def get_image_generation_logs(project_id: str) -> list[dict[str, Any]]:
    with connection() as con:
        rows = con.execute(
            """
            SELECT *
            FROM image_generation_logs
            WHERE project_id = ?
            ORDER BY created_at, id
            """,
            (project_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def _mark_image_selected_in_connection(
    con: sqlite3.Connection,
    log_id: int | None,
) -> None:
    if log_id is None:
        return

    cursor = con.execute(
        """
        UPDATE image_generation_logs
        SET selected_by_user = 1
        WHERE id = ? AND selected_by_user = 0
        """,
        (log_id,),
    )
    if cursor.rowcount != 1:
        raise ValueError(
            "The scene image is missing or has already been attached to a node."
        )

def replace_node_scene_image(
    node_id: str,
    image_log_id: int,
) -> dict[str, Any]:
    """
    Replace the current illustration attached to a node.

    Previous image-generation logs are preserved as evaluation history.
    """
    node = get_node(node_id)

    if not node:
        raise ValueError("Story node not found.")

    image_log = get_image_generation_log(image_log_id)

    if not image_log:
        raise ValueError("Scene image generation does not exist.")

    if (image_log["project_id"] != node["project_id"]):
        raise ValueError(
            "Scene image belongs to a different story."
        )

    if bool(
        image_log.get("selected_by_user")
    ):
        raise ValueError("Scene image has already been attached.")

    now = utc_now()

    with connection() as con:
        cursor = con.execute(
            """
            UPDATE nodes
            SET
                scene_image_log_id = ?,
                updated_at = ?
            WHERE
                id = ?
                AND deleted_at IS NULL
            """,
            (
                image_log_id,
                now,
                node_id,
            ),
        )

        if cursor.rowcount != 1:
            raise ValueError("Story node could not be updated.")

        _mark_image_selected_in_connection(
            con,
            image_log_id,
        )

        con.execute(
            """
            UPDATE projects
            SET updated_at = ?
            WHERE id = ?
            """,
            (
                now,
                node["project_id"],
            ),
        )

    updated = get_node(node_id)

    if updated is None:
        raise RuntimeError("Story node was not found after image replacement.")

    return updated

def insert_transcription_log(
    *,
    project_id: str | None,
    engine: str,
    model: str,
    device: str,
    compute_type: str,
    target_field: str,
    audio_content_type: str | None,
    audio_bytes: int,
    audio_seconds: float,
    model_load_seconds: float,
    inference_seconds: float,
    total_seconds: float,
    real_time_factor: float | None,
    cold_start: bool,
    language: str | None,
    language_probability: float | None,
    transcript: str,
) -> int:
    """Persist one STT run so Whisper and faster-whisper can be compared."""
    with connection() as con:
        cursor = con.execute(
            """
            INSERT INTO transcription_logs
                (
                    project_id, engine, model, device, compute_type,
                    target_field, audio_content_type, audio_bytes,
                    audio_seconds, model_load_seconds, inference_seconds,
                    total_seconds, real_time_factor, cold_start, language,
                    language_probability, transcript, created_at
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                engine,
                model,
                device,
                compute_type,
                target_field,
                audio_content_type,
                audio_bytes,
                audio_seconds,
                model_load_seconds,
                inference_seconds,
                total_seconds,
                real_time_factor,
                int(cold_start),
                language,
                language_probability,
                transcript,
                utc_now(),
            ),
        )
        return int(cursor.lastrowid)


def review_transcription_log(
    log_id: int,
    reviewed_text: str,
) -> bool:
    """Store the text the GM actually submits after reviewing the transcript."""
    with connection() as con:
        cursor = con.execute(
            """
            UPDATE transcription_logs
            SET reviewed_text = ?, reviewed_at = ?
            WHERE id = ?
            """,
            (reviewed_text, utc_now(), log_id),
        )
    return cursor.rowcount == 1


def get_transcription_logs(
    project_id: str,
) -> list[dict[str, Any]]:
    """Return STT measurements for later effectiveness/efficiency evaluation."""
    with connection() as con:
        rows = con.execute(
            """
            SELECT *
            FROM transcription_logs
            WHERE project_id = ?
            ORDER BY created_at, id
            """,
            (project_id,),
        ).fetchall()
    return [dict(row) for row in rows]

def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def decode_project(
    row: sqlite3.Row | dict[str, Any],
) -> dict[str, Any]:
    """Decode one project and attach normalized Story Element selections."""
    project = dict(row)

    project["story_element_mode"] = (
        project.get("story_element_mode", "guided")
        or "guided"
    )

    # Legacy JSON columns remain only so older databases can be migrated.
    project.pop("story_elements_json", None)
    project.pop("custom_story_elements_json", None)

    project["story_elements"] = (
        get_project_story_element_selections(
            project["id"]
        )
    )

    return project


def decode_node(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    node = dict(row)
    node["choices"] = json_load(node.pop("choices_json"), [])
    node["state_delta"] = json_load(node.pop("state_delta_json"), {})
    node["memory_snapshot"] = json_load(node.pop("memory_snapshot_json"), {})
    image_log_id = node.get("scene_image_log_id")
    node["scene_image"] = (
        get_image_generation_log(int(image_log_id))
        if image_log_id is not None
        else None
    )
    return node


def starter_choices() -> list[
    dict[str, Any]
]:
    choices = [
        StoryChoice(
            action_type="do",
            label=(
                "Investigate the most "
                "immediate clue"
            ),
        ),

        StoryChoice(
            action_type="ask",
            label=(
                "Ask a nearby character "
                "what they know"
            ),
        ),

        StoryChoice(
            action_type="continue",
            label=(
                "Wait and observe what "
                "happens next"
            ),
        ),
    ]

    return [
        choice.model_dump()
        for choice in choices
    ]


def create_project(setup: ProjectCreateRequest) -> dict[str, Any]:
    now = utc_now()
    project_id = f"project_{uuid4().hex}"
    branch_id = f"branch_{uuid4().hex}"
    node_id = f"node_{uuid4().hex}"

    with connection() as con:
        con.execute(
            """
            INSERT INTO projects
                (
                    id, title, genre, tone,
                    world_summary, main_objective,
                    story_element_mode,
                    story_elements_json,
                    custom_story_elements_json,
                    created_at, updated_at
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, '[]', '[]', ?, ?)
            """,
            (
                project_id,
                setup.title,
                setup.genre,
                setup.tone,
                setup.world_summary,
                setup.main_objective,
                setup.story_element_mode,
                now,
                now,
            ),
        )

        _replace_project_story_elements(
            con,
            project_id,
            setup.story_elements,
        )
        con.execute(
            """
            INSERT INTO branches
                (id, project_id, name, parent_branch_id, fork_node_id, head_node_id, created_at)
            VALUES (?, ?, ?, NULL, NULL, ?, ?)
            """,
            (branch_id, project_id, "Main branch", node_id, now),
        )
        con.execute(
            """
            INSERT INTO nodes
                (id, project_id, branch_id, parent_node_id, source_choice_id,
                 source_choice_label, title, story_text, choices_json,
                 state_delta_json, memory_snapshot_json, authoring_mode,
                 interaction_type, interaction_text, generated_by_model,
                 created_at, updated_at)
            VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?, 'manual',
                    'write', '', NULL, ?, ?)
            """,
            (
                node_id,
                project_id,
                branch_id,
                setup.starting_scene_title,
                setup.starting_scene_text,
                json_dump(starter_choices()),
                json_dump(StateDelta().model_dump()),
                json_dump(setup.initial_memory.model_dump()),
                now,
                now,
            ),
        )

    return get_state(project_id)


def _active_children(
    con: sqlite3.Connection,
    node_id: str,
) -> list[sqlite3.Row]:
    return con.execute(
        """
        SELECT * FROM nodes
        WHERE parent_node_id = ? AND deleted_at IS NULL
        ORDER BY created_at, id
        """,
        (node_id,),
    ).fetchall()


def _all_children(
    con: sqlite3.Connection,
    node_id: str,
) -> list[sqlite3.Row]:
    return con.execute(
        """
        SELECT * FROM nodes
        WHERE parent_node_id = ?
        ORDER BY created_at, id
        """,
        (node_id,),
    ).fetchall()


def _recalculate_descendant_memories(
    con: sqlite3.Connection,
    parent_node_id: str,
    parent_memory: StoryMemory,
) -> None:
    """Rebuild active descendant snapshots from their stored state deltas."""
    queue: list[tuple[str, StoryMemory]] = [
        (parent_node_id, parent_memory)
    ]
    visited: set[str] = set()

    while queue:
        current_parent_id, current_memory = queue.pop(0)
        if current_parent_id in visited:
            raise RuntimeError("Cycle detected while recalculating node memories.")
        visited.add(current_parent_id)

        # Include deleted descendants so restored branches do not revive
        # stale memory snapshots after an earlier edit.
        for child in _all_children(con, current_parent_id):
            delta = StateDelta.model_validate(
                json_load(child["state_delta_json"], {})
            )
            child_memory = apply_state_delta(current_memory, delta)
            con.execute(
                """
                UPDATE nodes
                SET memory_snapshot_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json_dump(child_memory.model_dump()),
                    utc_now(),
                    child["id"],
                ),
            )
            queue.append((child["id"], child_memory))


def update_project(
    project_id: str,
    setup: ProjectUpdateRequest,
) -> dict[str, Any]:
    project = get_project(project_id)
    if not project:
        raise ValueError("Story project not found.")

    root = get_root_node(project_id)
    if not root:
        raise ValueError("The story does not have a root node.")

    now = utc_now()
    new_root_memory = setup.initial_memory

    with connection() as con:
        con.execute(
            """
            UPDATE projects
            SET
                title = ?,
                genre = ?,
                tone = ?,
                world_summary = ?,
                main_objective = ?,
                story_element_mode = ?,
                story_elements_json = '[]',
                custom_story_elements_json = '[]',
                updated_at = ?
            WHERE id = ?
            """,
            (
                setup.title,
                setup.genre,
                setup.tone,
                setup.world_summary,
                setup.main_objective,
                setup.story_element_mode,
                now,
                project_id,
            ),
        )

        _replace_project_story_elements(
            con,
            project_id,
            setup.story_elements,
        )
        con.execute(
            """
            UPDATE nodes
            SET title = ?, story_text = ?, memory_snapshot_json = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                setup.starting_scene_title,
                setup.starting_scene_text,
                json_dump(new_root_memory.model_dump()),
                now,
                root["id"],
            ),
        )

        # Rebuild every active descendant from the updated initial memory.
        # This prevents root-memory edits from leaving stale branch snapshots.
        _recalculate_descendant_memories(
            con,
            root["id"],
            new_root_memory,
        )

    return get_state(project_id)


def delete_project(project_id: str) -> bool:
    with connection() as con:
        cursor = con.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    return cursor.rowcount > 0


def touch_project(project_id: str) -> None:
    with connection() as con:
        con.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (utc_now(), project_id),
        )


def list_projects() -> list[dict[str, Any]]:
    with connection() as con:
        rows = con.execute(
            """
            SELECT
                p.*,
                (SELECT COUNT(*) FROM branches b WHERE b.project_id = p.id AND b.deleted_at IS NULL) AS branch_count,
                (SELECT COUNT(*) FROM nodes n WHERE n.project_id = p.id AND n.deleted_at IS NULL) AS node_count,
                COALESCE(
                    (SELECT n.title FROM nodes n
                     WHERE n.project_id = p.id AND n.deleted_at IS NULL
                     ORDER BY n.updated_at DESC LIMIT 1),
                    ''
                ) AS latest_node_title,
                COALESCE(
                    (SELECT substr(replace(n.story_text, char(10), ' '), 1, 180)
                     FROM nodes n WHERE n.project_id = p.id AND n.deleted_at IS NULL
                     ORDER BY n.updated_at DESC LIMIT 1),
                    ''
                ) AS preview
            FROM projects p
            ORDER BY p.updated_at DESC, p.title COLLATE NOCASE
            """
        ).fetchall()
    return [decode_project(row) for row in rows]


def get_project(project_id: str) -> dict[str, Any] | None:
    with connection() as con:
        row = con.execute(
            "SELECT * FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
    return decode_project(row) if row else None


def get_root_node(project_id: str) -> dict[str, Any] | None:
    with connection() as con:
        row = con.execute(
            "SELECT * FROM nodes WHERE project_id = ? AND parent_node_id IS NULL AND deleted_at IS NULL LIMIT 1",
            (project_id,),
        ).fetchone()
    return decode_node(row) if row else None


def get_node(node_id: str) -> dict[str, Any] | None:
    with connection() as con:
        row = con.execute("SELECT * FROM nodes WHERE id = ? AND deleted_at IS NULL", (node_id,)).fetchone()
    return decode_node(row) if row else None


def get_branch(branch_id: str) -> dict[str, Any] | None:
    with connection() as con:
        row = con.execute("SELECT * FROM branches WHERE id = ? AND deleted_at IS NULL", (branch_id,)).fetchone()
    return row_to_dict(row)


def get_all_branches(project_id: str) -> list[dict[str, Any]]:
    with connection() as con:
        rows = con.execute(
            "SELECT * FROM branches WHERE project_id = ? AND deleted_at IS NULL ORDER BY created_at, name",
            (project_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_nodes(project_id: str) -> list[dict[str, Any]]:
    with connection() as con:
        rows = con.execute(
            "SELECT * FROM nodes WHERE project_id = ? AND deleted_at IS NULL ORDER BY created_at",
            (project_id,),
        ).fetchall()
    return [decode_node(row) for row in rows]


def get_node_path(node_id: str) -> list[dict[str, Any]]:
    path: list[dict[str, Any]] = []
    current = get_node(node_id)
    visited: set[str] = set()
    while current:
        if current["id"] in visited:
            raise RuntimeError("Cycle detected in story-node graph.")
        visited.add(current["id"])
        path.append(current)
        parent_id = current.get("parent_node_id")
        current = get_node(parent_id) if parent_id else None
    return list(reversed(path))


def get_branch_path(branch_id: str) -> list[dict[str, Any]]:
    branch = get_branch(branch_id)
    if not branch or not branch.get("head_node_id"):
        return []
    return get_node_path(branch["head_node_id"])


def get_recent_nodes(node_id: str, limit: int = 5) -> list[dict[str, Any]]:
    return get_node_path(node_id)[-limit:]


def get_state(project_id: str) -> dict[str, Any]:
    project = get_project(project_id)
    if not project:
        raise ValueError("Story project not found.")
    branches = get_all_branches(project_id)
    nodes = get_all_nodes(project_id)
    return {
        "project": project,
        "branches": branches,
        "nodes": nodes,
        "branch_paths": {
            branch["id"]: [node["id"] for node in get_branch_path(branch["id"])]
            for branch in branches
        },
    }


def _mark_generation_saved_in_connection(
    con: sqlite3.Connection,
    log_id: int | None,
) -> None:
    if log_id is None:
        return

    cursor = con.execute(
        """
        UPDATE generation_logs
        SET saved_by_user = 1
        WHERE id = ? AND saved_by_user = 0
        """,
        (log_id,),
    )
    if cursor.rowcount != 1:
        raise ValueError(
            "The generation log is missing or has already been saved."
        )


def _insert_node(
    con: sqlite3.Connection,
    *,
    node_id: str,
    project_id: str,
    branch_id: str,
    parent_node_id: str,
    source_choice_id: str | None,
    source_choice_label: str | None,
    title: str,
    story_text: str,
    choices: list[dict[str, Any]],
    state_delta: dict[str, Any],
    memory_snapshot: dict[str, Any],
    authoring_mode: str,
    interaction_type: str,
    interaction_text: str,
    generated_by_model: str | None,
    generation_log_id: int | None,
    scene_image_log_id: int | None,
    now: str,
) -> None:
    con.execute(
        """
        INSERT INTO nodes
            (id, project_id, branch_id, parent_node_id, source_choice_id,
             source_choice_label, title, story_text, choices_json,
             state_delta_json, memory_snapshot_json, authoring_mode,
             interaction_type, interaction_text, generated_by_model,
             generation_log_id, scene_image_log_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            node_id,
            project_id,
            branch_id,
            parent_node_id,
            source_choice_id,
            source_choice_label,
            title,
            story_text,
            json_dump(choices),
            json_dump(state_delta),
            json_dump(memory_snapshot),
            authoring_mode,
            interaction_type,
            interaction_text,
            generated_by_model,
            generation_log_id,
            scene_image_log_id,
            now,
            now,
        ),
    )


def create_node(
    *,
    node_id: str,
    project_id: str,
    branch_id: str,
    parent_node_id: str,
    source_choice_id: str | None,
    source_choice_label: str | None,
    title: str,
    story_text: str,
    choices: list[dict[str, Any]],
    state_delta: dict[str, Any],
    memory_snapshot: dict[str, Any],
    authoring_mode: str,
    interaction_type: str,
    interaction_text: str,
    generated_by_model: str | None,
    generation_log_id: int | None = None,
    scene_image_log_id: int | None = None,
) -> None:
    now = utc_now()
    with connection() as con:
        _insert_node(
            con,
            node_id=node_id,
            project_id=project_id,
            branch_id=branch_id,
            parent_node_id=parent_node_id,
            source_choice_id=source_choice_id,
            source_choice_label=source_choice_label,
            title=title,
            story_text=story_text,
            choices=choices,
            state_delta=state_delta,
            memory_snapshot=memory_snapshot,
            authoring_mode=authoring_mode,
            interaction_type=interaction_type,
            interaction_text=interaction_text,
            generated_by_model=generated_by_model,
            generation_log_id=generation_log_id,
            scene_image_log_id=scene_image_log_id,
            now=now,
        )
        con.execute(
            "UPDATE branches SET head_node_id = ? WHERE id = ?",
            (node_id, branch_id),
        )
        con.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (now, project_id),
        )
        _mark_generation_saved_in_connection(con, generation_log_id)
        _mark_image_selected_in_connection(con, scene_image_log_id)


def create_fork_with_node(
    *,
    branch_id: str,
    project_id: str,
    branch_name: str,
    parent_branch_id: str,
    fork_node_id: str,
    node_id: str,
    source_choice_id: str | None,
    source_choice_label: str | None,
    title: str,
    story_text: str,
    choices: list[dict[str, Any]],
    state_delta: dict[str, Any],
    memory_snapshot: dict[str, Any],
    authoring_mode: str,
    interaction_type: str,
    interaction_text: str,
    generated_by_model: str | None,
    generation_log_id: int | None = None,
    scene_image_log_id: int | None = None,
) -> None:
    """Create a fork and its first unique node in one transaction."""
    now = utc_now()
    with connection() as con:
        con.execute(
            """
            INSERT INTO branches
                (id, project_id, name, parent_branch_id, fork_node_id,
                 head_node_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                branch_id,
                project_id,
                branch_name,
                parent_branch_id,
                fork_node_id,
                fork_node_id,
                now,
            ),
        )
        _insert_node(
            con,
            node_id=node_id,
            project_id=project_id,
            branch_id=branch_id,
            parent_node_id=fork_node_id,
            source_choice_id=source_choice_id,
            source_choice_label=source_choice_label,
            title=title,
            story_text=story_text,
            choices=choices,
            state_delta=state_delta,
            memory_snapshot=memory_snapshot,
            authoring_mode=authoring_mode,
            interaction_type=interaction_type,
            interaction_text=interaction_text,
            generated_by_model=generated_by_model,
            generation_log_id=generation_log_id,
            scene_image_log_id=scene_image_log_id,
            now=now,
        )
        con.execute(
            "UPDATE branches SET head_node_id = ? WHERE id = ?",
            (node_id, branch_id),
        )
        con.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (now, project_id),
        )
        _mark_generation_saved_in_connection(con, generation_log_id)
        _mark_image_selected_in_connection(con, scene_image_log_id)


def update_node_content(
    node_id: str,
    title: str,
    story_text: str,
    choices: list[dict[str, Any]],
    applied_delta: dict[str, Any] | None = None,
) -> None:
    node = get_node(node_id)
    if not node:
        raise ValueError("Story node not found.")

    new_delta = (
        StateDelta.model_validate(applied_delta)
        if applied_delta is not None
        else StateDelta.model_validate(node["state_delta"])
    )

    with connection() as con:
        has_children = bool(_active_children(con, node_id))
        story_changed = story_text != node["story_text"]

        existing_delta = StateDelta.model_validate(node["state_delta"])

        if node["parent_node_id"] is None and applied_delta is not None:
            if new_delta.has_meaningful_change():
                raise ValueError(
                    "The root node cannot have an applied delta. Update its "
                    "initial memory through Story setup instead."
                )

        if (
            story_changed
            and applied_delta is None
            and (has_children or existing_delta.has_meaningful_change())
        ):
            raise ValueError(
                "This scene has state-dependent memory. Supply its "
                "applied_delta so SekAI can recalculate its snapshot and "
                "all active descendants safely."
            )

        if node["parent_node_id"]:
            parent_row = con.execute(
                "SELECT * FROM nodes WHERE id = ? AND deleted_at IS NULL",
                (node["parent_node_id"],),
            ).fetchone()
            if not parent_row:
                raise ValueError("The parent node is unavailable.")
            parent_memory = StoryMemory.model_validate(
                json_load(parent_row["memory_snapshot_json"], {})
            )
            node_memory = apply_state_delta(parent_memory, new_delta)
        else:
            # Root-node memory is controlled by update_project().
            node_memory = StoryMemory.model_validate(node["memory_snapshot"])

        now = utc_now()
        con.execute(
            """
            UPDATE nodes
            SET title = ?, 
                story_text = ?, 
                choices_json = ?,
                state_delta_json = ?, 
                memory_snapshot_json = ?,
                scene_image_log_id = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (
                title,
                story_text,
                json_dump(choices),
                json_dump(new_delta.model_dump()),
                json_dump(node_memory.model_dump()),
                now,
                node_id,
            ),
        )
        con.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (now, node["project_id"]),
        )

        _recalculate_descendant_memories(con, node_id, node_memory)


def _deletion_batch_id() -> str:
    return f"trash_{uuid4().hex}"


def _active_branch_descendants(con: sqlite3.Connection, branch_id: str) -> list[str]:
    """Return the selected active branch and every active descendant branch."""
    result: list[str] = []
    queue = [branch_id]
    while queue:
        current = queue.pop(0)
        if current in result:
            continue
        result.append(current)
        rows = con.execute(
            """
            SELECT id FROM branches
            WHERE parent_branch_id = ? AND deleted_at IS NULL
            """,
            (current,),
        ).fetchall()
        queue.extend(row["id"] for row in rows)
    return result


def delete_leaf_node(node_id: str) -> dict[str, Any]:
    """Soft-delete a leaf node so it can be restored from the trash."""
    node = get_node(node_id)
    if not node:
        raise ValueError("Story node not found.")
    if node.get("parent_node_id") is None:
        raise ValueError("The root scene cannot be deleted.")

    deleted_at = utc_now()
    batch_id = _deletion_batch_id()

    with connection() as con:
        child = con.execute(
            """
            SELECT id FROM nodes
            WHERE parent_node_id = ? AND deleted_at IS NULL
            LIMIT 1
            """,
            (node_id,),
        ).fetchone()
        if child:
            raise ValueError("Only leaf nodes can be deleted. Delete later nodes first.")

        branch = con.execute(
            "SELECT * FROM branches WHERE id = ? AND deleted_at IS NULL",
            (node["branch_id"],),
        ).fetchone()
        if not branch:
            raise ValueError("The node's branch no longer exists.")

        con.execute(
            """
            UPDATE nodes
            SET deleted_at = ?, deletion_batch_id = ?, deleted_reason = ?
            WHERE id = ?
            """,
            (deleted_at, batch_id, "User deleted node", node_id),
        )

        parent_id = node["parent_node_id"]
        removed_empty_fork = bool(
            branch["parent_branch_id"]
            and parent_id == branch["fork_node_id"]
        )

        if removed_empty_fork:
            # This was the only unique node in a fork. Keep the branch row in
            # the database, but hide it together with the deleted node.
            con.execute(
                """
                UPDATE branches
                SET deleted_at = ?, deletion_batch_id = ?, deleted_reason = ?
                WHERE id = ?
                """,
                (
                    deleted_at,
                    batch_id,
                    "Branch became empty after node deletion",
                    branch["id"],
                ),
            )
        else:
            con.execute(
                "UPDATE branches SET head_node_id = ? WHERE id = ?",
                (parent_id, branch["id"]),
            )

        con.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (deleted_at, node["project_id"]),
        )

    return {
        "node": node,
        "batch_id": batch_id,
        "branch_was_trashed": removed_empty_fork,
    }


def delete_branch(branch_id: str) -> dict[str, Any]:
    """Soft-delete a forked branch, its unique nodes, and descendant forks."""
    branch = get_branch(branch_id)
    if not branch:
        raise ValueError("Story branch not found.")
    if not branch.get("parent_branch_id"):
        raise ValueError("The main branch cannot be deleted.")

    deleted_at = utc_now()
    batch_id = _deletion_batch_id()

    with connection() as con:
        branch_ids = _active_branch_descendants(con, branch_id)
        placeholders = ",".join("?" for _ in branch_ids)

        con.execute(
            f"""
            UPDATE nodes
            SET deleted_at = ?, deletion_batch_id = ?, deleted_reason = ?
            WHERE branch_id IN ({placeholders}) AND deleted_at IS NULL
            """,
            (
                deleted_at,
                batch_id,
                "User deleted branch",
                *branch_ids,
            ),
        )
        con.execute(
            f"""
            UPDATE branches
            SET deleted_at = ?, deletion_batch_id = ?, deleted_reason = ?
            WHERE id IN ({placeholders}) AND deleted_at IS NULL
            """,
            (
                deleted_at,
                batch_id,
                "User deleted branch",
                *branch_ids,
            ),
        )
        con.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (deleted_at, branch["project_id"]),
        )

    return {
        "branch": branch,
        "batch_id": batch_id,
        "deleted_branch_ids": branch_ids,
        "parent_branch_id": branch["parent_branch_id"],
    }


def list_deleted_items(project_id: str) -> list[dict[str, Any]]:
    """Return trash entries grouped by one delete operation."""
    with connection() as con:
        branch_rows = con.execute(
            """
            SELECT id, name, deleted_at, deletion_batch_id, deleted_reason
            FROM branches
            WHERE project_id = ? AND deleted_at IS NOT NULL
            ORDER BY deleted_at DESC
            """,
            (project_id,),
        ).fetchall()
        node_rows = con.execute(
            """
            SELECT id, title, branch_id, parent_node_id, deleted_at,
                   deletion_batch_id, deleted_reason
            FROM nodes
            WHERE project_id = ? AND deleted_at IS NOT NULL
            ORDER BY deleted_at DESC
            """,
            (project_id,),
        ).fetchall()

    groups: dict[str, dict[str, Any]] = {}

    def ensure_group(batch_id: str, deleted_at: str) -> dict[str, Any]:
        group = groups.setdefault(
            batch_id,
            {
                "batch_id": batch_id,
                "deleted_at": deleted_at,
                "branches": [],
                "nodes": [],
            },
        )
        if deleted_at > group["deleted_at"]:
            group["deleted_at"] = deleted_at
        return group

    for row in branch_rows:
        batch_id = row["deletion_batch_id"] or f"legacy_branch_{row['id']}"
        ensure_group(batch_id, row["deleted_at"])["branches"].append(dict(row))

    for row in node_rows:
        batch_id = row["deletion_batch_id"] or f"legacy_node_{row['id']}"
        ensure_group(batch_id, row["deleted_at"])["nodes"].append(dict(row))

    result = []
    for group in groups.values():
        branch_names = [item["name"] for item in group["branches"]]
        node_titles = [item["title"] for item in group["nodes"]]
        if branch_names:
            title = branch_names[0]
            kind = "branch"
        else:
            title = node_titles[0] if node_titles else "Deleted item"
            kind = "node"
        result.append(
            {
                **group,
                "kind": kind,
                "title": title,
                "branch_count": len(group["branches"]),
                "node_count": len(group["nodes"]),
            }
        )

    return sorted(result, key=lambda item: item["deleted_at"], reverse=True)


def restore_deletion_batch(batch_id: str) -> dict[str, Any]:
    """Restore one node deletion or one deleted branch subtree."""
    with connection() as con:
        branch_rows = con.execute(
            """
            SELECT * FROM branches
            WHERE deletion_batch_id = ? AND deleted_at IS NOT NULL
            """,
            (batch_id,),
        ).fetchall()
        node_rows = con.execute(
            """
            SELECT * FROM nodes
            WHERE deletion_batch_id = ? AND deleted_at IS NOT NULL
            """,
            (batch_id,),
        ).fetchall()

    if not branch_rows and not node_rows:
        raise ValueError("Deleted item was not found in the trash.")

    restoring_branch_ids = {row["id"] for row in branch_rows}
    restoring_node_ids = {row["id"] for row in node_rows}
    project_id = (branch_rows[0] if branch_rows else node_rows[0])["project_id"]

    # Validate dependencies before changing any row.
    for branch in branch_rows:
        parent_branch_id = branch["parent_branch_id"]
        if parent_branch_id and parent_branch_id not in restoring_branch_ids:
            if not get_branch(parent_branch_id):
                raise ValueError(
                    "Restore the parent branch before restoring this branch."
                )
        fork_node_id = branch["fork_node_id"]
        if fork_node_id and fork_node_id not in restoring_node_ids:
            if not get_node(fork_node_id):
                raise ValueError(
                    "The branch's fork node is not active, so the branch cannot be restored yet."
                )

    for node in node_rows:
        parent_node_id = node["parent_node_id"]
        if parent_node_id and parent_node_id not in restoring_node_ids:
            if not get_node(parent_node_id):
                raise ValueError(
                    "Restore the parent node before restoring this node."
                )

    # A single deleted node can only be returned to the same linear branch
    # when no new continuation has been written after its deletion.
    if not branch_rows and len(node_rows) == 1:
        node = node_rows[0]
        branch = get_branch(node["branch_id"])
        if not branch:
            raise ValueError("The original branch is not active.")
        if branch["head_node_id"] != node["parent_node_id"]:
            raise ValueError(
                "This branch has advanced since the node was deleted. "
                "Restore is blocked to avoid creating two heads on one branch."
            )

    with connection() as con:
        con.execute(
            """
            UPDATE branches
            SET deleted_at = NULL, deletion_batch_id = NULL, deleted_reason = NULL
            WHERE deletion_batch_id = ?
            """,
            (batch_id,),
        )
        con.execute(
            """
            UPDATE nodes
            SET deleted_at = NULL, deletion_batch_id = NULL, deleted_reason = NULL
            WHERE deletion_batch_id = ?
            """,
            (batch_id,),
        )

        if not branch_rows and len(node_rows) == 1:
            node = node_rows[0]
            con.execute(
                "UPDATE branches SET head_node_id = ? WHERE id = ?",
                (node["id"], node["branch_id"]),
            )

        con.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (utc_now(), project_id),
        )

    return get_state(project_id)

def insert_generation_attempt(
    *,
    request_id: str,
    pipeline_version: str,
    project_id: str,
    branch_id: str,
    parent_node_id: str,
    selected_choice_id: str | None,
    attempt_number: int,
    model: str,
    temperature: float,
    elapsed_seconds: float,
    outcome: str,
    failure_stage: str | None = None,
    failure_detail: str | None = None,
    story_word_count: int | None = None,
) -> int:

    with connection() as con:
        cursor = con.execute(
            """
            INSERT INTO generation_attempts
                (
                    request_id,
                    pipeline_version,
                    project_id,
                    branch_id,
                    parent_node_id,
                    selected_choice_id,
                    attempt_number,
                    model,
                    temperature,
                    elapsed_seconds,
                    outcome,
                    failure_stage,
                    failure_detail,
                    story_word_count,
                    created_at
                )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                request_id,
                pipeline_version,
                project_id,
                branch_id,
                parent_node_id,
                selected_choice_id,
                attempt_number,
                model,
                temperature,
                elapsed_seconds,
                outcome,
                failure_stage,
                (
                    failure_detail[:2_000]
                    if failure_detail
                    else None
                ),
                story_word_count,
                utc_now(),
            ),
        )

    return int(
        cursor.lastrowid
    )


def insert_generation_log(
    *,
    request_id: str,
    pipeline_version: str,
    project_id: str,
    branch_id: str,
    parent_node_id: str,
    selected_choice_id: str | None,
    requested_model: str,
    model: str,
    semantic_check_completed: bool,
    continuity_check_completed: bool,
    continuity_model: str | None,
    continuity_checked_facts: int,
    continuity_max_contradiction: float | None,
    requested_temperature: float,
    effective_temperature: float,
    elapsed_seconds: float,
    prompt_tokens: int | None,
    output_tokens: int | None,
    tokens_per_second: float | None,
    attempts: int,
    interaction_type: str,
    player_input: str,
    gm_instruction: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    validation_pass: bool = True,
) -> int:
    """
    Store the final draft returned to the user.

    validation_pass=False identifies a parseable draft that SekAI exposed for
    human review after automated validation rejected it. If the user later
    saves that reviewed draft, saved_by_user becomes 1 while validation_pass
    remains 0, preserving a clear audit trail of the human override.
    """
    with connection() as con:
        cursor = con.execute(
            """
            INSERT INTO generation_logs
                (
                    request_id,
                    pipeline_version,
                    project_id,
                    branch_id,
                    parent_node_id,
                    selected_choice_id,
                    requested_model,
                    model,
                    requested_temperature,
                    effective_temperature,
                    elapsed_seconds,
                    prompt_tokens,
                    output_tokens,
                    tokens_per_second,
                    attempts,
                    validation_pass,
                    saved_by_user,
                    semantic_check_completed,
                    continuity_check_completed,
                    continuity_model,
                    continuity_checked_facts,
                    continuity_max_contradiction,
                    interaction_type,
                    player_input,
                    gm_instruction,
                    request_json,
                    response_json,
                    created_at
                )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                request_id,
                pipeline_version,
                project_id,
                branch_id,
                parent_node_id,
                selected_choice_id,
                requested_model,
                model,
                requested_temperature,
                effective_temperature,
                elapsed_seconds,
                prompt_tokens,
                output_tokens,
                tokens_per_second,
                attempts,
                int(validation_pass),
                int(semantic_check_completed),
                int(continuity_check_completed),
                continuity_model,
                continuity_checked_facts,
                continuity_max_contradiction,
                interaction_type,
                player_input,
                gm_instruction,
                json_dump(request_payload),
                json_dump(response_payload),
                utc_now(),
            ),
        )
    return int(cursor.lastrowid)

def get_generation_log(log_id: int) -> dict[str, Any] | None:
    with connection() as con:
        row = con.execute(
            "SELECT * FROM generation_logs WHERE id = ?",
            (log_id,),
        ).fetchone()
    return row_to_dict(row)


def mark_generation_saved(log_id: int) -> None:
    with connection() as con:
        _mark_generation_saved_in_connection(con, log_id)


def get_generation_logs(project_id: str) -> list[dict[str, Any]]:
    with connection() as con:
        rows = con.execute(
            "SELECT * FROM generation_logs WHERE project_id = ? ORDER BY id",
            (project_id,),
        ).fetchall()
    return [dict(row) for row in rows]

def get_generation_attempts(
    request_id: str,
) -> list[dict[str, Any]]:
    with connection() as con:
        rows = con.execute(
            """
            SELECT *
            FROM generation_attempts
            WHERE request_id = ?
            ORDER BY attempt_number, id
            """,
            (request_id,),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]