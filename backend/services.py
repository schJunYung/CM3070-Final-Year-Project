from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any
from uuid import uuid4

from . import db
from .schemas import (
    DraftValidationRequest,
    DraftValidationResponse,
    SaveNodeRequest,
    StoryMemory,
)
from .state import apply_state_delta
from .validation_service import (
    blocking_issues,
    story_word_count,
    validate_reviewed_draft,
    warning_issues,
)


def find_choice(
    node: dict[str, Any],
    choice_id: str | None,
) -> dict[str, Any] | None:
    if not choice_id:
        return None
    return next(
        (
            choice
            for choice in node.get("choices", [])
            if choice.get("id") == choice_id
        ),
        None,
    )


def _validate_generation_log(
    request: SaveNodeRequest,
) -> dict[str, Any] | None:
    if request.authoring_mode != "ai":
        return None

    if request.generation_log_id is None:
        raise ValueError("An AI-authored scene requires its generation log ID.")

    log = db.get_generation_log(request.generation_log_id)
    if not log:
        raise ValueError("The generation log does not exist.")
    if log["saved_by_user"]:
        raise ValueError("This generated draft has already been saved.")

    if (
        not bool(log.get("validation_pass"))
        and not request.review_override
    ):
        raise ValueError(
            "This draft requires explicit human review before it can be saved."
        )

    expected_values = {
        "project_id": request.project_id,
        "branch_id": request.source_branch_id,
        "parent_node_id": request.parent_node_id,
        "selected_choice_id": request.selected_choice_id,
    }
    for field, expected in expected_values.items():
        if log[field] != expected:
            raise ValueError(
                "The generation log does not match the current "
                f"{field.replace('_', ' ')}."
            )

    if not request.generated_by_model:
        raise ValueError("The accepted model name is required for an AI scene.")
    if log["model"] != request.generated_by_model:
        raise ValueError(
            "The accepted model does not match the generation log."
        )

    return log


def _validate_scene_image(
    request: SaveNodeRequest,
) -> dict[str, Any] | None:
    if request.scene_image_log_id is None:
        return None

    log = db.get_image_generation_log(
        request.scene_image_log_id
    )
    if not log:
        raise ValueError("The selected scene image does not exist.")
    if log["project_id"] != request.project_id:
        raise ValueError(
            "The selected scene image belongs to a different story."
        )
    if bool(log.get("selected_by_user")):
        raise ValueError(
            "This scene image has already been attached to a node."
        )
    if log.get("source_kind") != request.authoring_mode:
        raise ValueError(
            "The scene image was generated for a different authoring mode."
        )
    return log


def validate_ai_draft(
    request: DraftValidationRequest,
) -> DraftValidationResponse:
    """Validate the current human-edited AI draft without persisting it."""
    project = db.get_project(request.project_id)
    branch = db.get_branch(request.branch_id)
    parent_node = db.get_node(request.parent_node_id)

    if not project:
        raise ValueError("Story project not found.")
    if not branch or branch["project_id"] != request.project_id:
        raise ValueError("The selected branch does not belong to this story.")
    if not parent_node or parent_node["project_id"] != request.project_id:
        raise ValueError("The selected parent node does not belong to this story.")

    branch_path_ids = {
        node["id"]
        for node in db.get_branch_path(branch["id"])
    }
    if parent_node["id"] not in branch_path_ids:
        raise ValueError("The selected node is not on the selected branch path.")

    parent_memory = StoryMemory.model_validate(
        parent_node["memory_snapshot"]
    )
    issues = validate_reviewed_draft(
        story_text=request.story_text,
        parent_memory=parent_memory,
        applied_delta=request.applied_delta,
        previous_nodes=db.get_node_path(parent_node["id"]),
        choices=request.choices,
    )

    blocking = blocking_issues(issues)
    warnings = warning_issues(issues)

    return DraftValidationResponse(
        can_save=not blocking,
        has_warnings=bool(warnings),
        word_count=story_word_count(request.story_text),
        blocking_count=len(blocking),
        warning_count=len(warnings),
        issues=issues,
    )


def save_story_node(request: SaveNodeRequest) -> dict[str, Any]:
    project = db.get_project(request.project_id)
    source_branch = db.get_branch(request.source_branch_id)
    parent_node = db.get_node(request.parent_node_id)

    if not project:
        raise ValueError("Story project not found.")
    if not source_branch or source_branch["project_id"] != request.project_id:
        raise ValueError("The selected branch does not belong to this story.")
    if not parent_node or parent_node["project_id"] != request.project_id:
        raise ValueError("The selected parent node does not belong to this story.")

    branch_path_ids = {
        node["id"]
        for node in db.get_branch_path(source_branch["id"])
    }
    if parent_node["id"] not in branch_path_ids:
        raise ValueError("The selected node is not on the selected branch path.")

    selected_choice = find_choice(parent_node, request.selected_choice_id)
    if request.selected_choice_id and not selected_choice:
        raise ValueError("The selected choice does not belong to the selected node.")

    _validate_generation_log(request)
    _validate_scene_image(request)

    parent_memory = StoryMemory.model_validate(
        parent_node["memory_snapshot"]
    )

    if request.authoring_mode == "ai":
        validation_issues = validate_reviewed_draft(
            story_text=request.story_text,
            parent_memory=parent_memory,
            applied_delta=request.applied_delta,
            previous_nodes=db.get_node_path(parent_node["id"]),
        )
        blocking = blocking_issues(validation_issues)
        warnings = warning_issues(validation_issues)

        if blocking:
            details = "; ".join(issue.message for issue in blocking)
            raise ValueError(
                "The reviewed AI scene still contains blocking validation "
                f"issues: {details}"
            )

        if warnings and not request.review_override:
            details = "; ".join(issue.message for issue in warnings)
            raise ValueError(
                "This reviewed AI scene still has non-blocking warnings and "
                "requires explicit human review before saving: "
                f"{details}"
            )

    next_memory = apply_state_delta(
        parent_memory,
        request.applied_delta,
    )

    continuing_head = (
        source_branch["head_node_id"] == parent_node["id"]
        and not request.force_fork
    )

    node_id = f"node_{uuid4().hex}"
    choice_label = selected_choice["label"] if selected_choice else None
    choices = [choice.model_dump() for choice in request.choices]
    state_delta = request.applied_delta.model_dump()
    memory_snapshot = next_memory.model_dump()

    if continuing_head:
        target_branch_id = source_branch["id"]
        fork_created = False
        db.create_node(
            node_id=node_id,
            project_id=request.project_id,
            branch_id=target_branch_id,
            parent_node_id=parent_node["id"],
            source_choice_id=request.selected_choice_id,
            source_choice_label=choice_label,
            title=request.title,
            story_text=request.story_text,
            choices=choices,
            state_delta=state_delta,
            memory_snapshot=memory_snapshot,
            authoring_mode=request.authoring_mode,
            interaction_type=request.interaction_type,
            interaction_text=request.interaction_text,
            generated_by_model=request.generated_by_model,
            generation_log_id=request.generation_log_id,
            scene_image_log_id=request.scene_image_log_id,
        )
    else:
        fork_created = True
        target_branch_id = f"branch_{uuid4().hex}"
        default_label = choice_label or request.interaction_text or parent_node["title"]
        branch_name = (request.branch_name or "").strip()
        if not branch_name:
            branch_name = f"Fork: {default_label.strip()[:70]}"

        db.create_fork_with_node(
            branch_id=target_branch_id,
            project_id=request.project_id,
            branch_name=branch_name,
            parent_branch_id=source_branch["id"],
            fork_node_id=parent_node["id"],
            node_id=node_id,
            source_choice_id=request.selected_choice_id,
            source_choice_label=choice_label,
            title=request.title,
            story_text=request.story_text,
            choices=choices,
            state_delta=state_delta,
            memory_snapshot=memory_snapshot,
            authoring_mode=request.authoring_mode,
            interaction_type=request.interaction_type,
            interaction_text=request.interaction_text,
            generated_by_model=request.generated_by_model,
            generation_log_id=request.generation_log_id,
            scene_image_log_id=request.scene_image_log_id,
        )

    return {
        "node": db.get_node(node_id),
        "branch": db.get_branch(target_branch_id),
        "fork_created": fork_created,
        "state": db.get_state(request.project_id),
    }


def common_ancestor(
    left_path: list[dict[str, Any]],
    right_path: list[dict[str, Any]],
) -> dict[str, Any] | None:
    common = None
    for left_node, right_node in zip(left_path, right_path):
        if left_node["id"] != right_node["id"]:
            break
        common = left_node
    return common


def memory_diff(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    fields = list(StoryMemory.model_fields)
    return {
        field: {
            "left": left.get(field),
            "right": right.get(field),
            "different": left.get(field) != right.get(field),
        }
        for field in fields
    }


def compare_branches(
    left_branch_id: str,
    right_branch_id: str,
) -> dict[str, Any]:
    left_branch = db.get_branch(left_branch_id)
    right_branch = db.get_branch(right_branch_id)
    if not left_branch or not right_branch:
        raise ValueError("Both branches must exist.")
    if left_branch["project_id"] != right_branch["project_id"]:
        raise ValueError("Branches from different stories cannot be compared.")

    left_path = db.get_branch_path(left_branch_id)
    right_path = db.get_branch_path(right_branch_id)
    if not left_path or not right_path:
        raise ValueError("Both branches must contain an active story path.")

    ancestor = common_ancestor(left_path, right_path)
    left_index = next(
        (
            index
            for index, node in enumerate(left_path)
            if ancestor and node["id"] == ancestor["id"]
        ),
        -1,
    )
    right_index = next(
        (
            index
            for index, node in enumerate(right_path)
            if ancestor and node["id"] == ancestor["id"]
        ),
        -1,
    )

    return {
        "left_branch": left_branch,
        "right_branch": right_branch,
        "common_ancestor": ancestor,
        "left_unique_nodes": left_path[left_index + 1 :],
        "right_unique_nodes": right_path[right_index + 1 :],
        "memory_diff": memory_diff(
            left_path[-1]["memory_snapshot"],
            right_path[-1]["memory_snapshot"],
        ),
    }


def objective_evidence(project_id: str) -> dict[str, Any]:
    state = db.get_state(project_id)
    nodes = state["nodes"]
    branches = state["branches"]
    all_logs = db.get_generation_logs(project_id)
    logs = [
        log
        for log in all_logs
        if bool(log.get("validation_pass"))
    ]

    ai_nodes = [node for node in nodes if node.get("authoring_mode") == "ai"]
    manual_nodes = [
        node for node in nodes if node.get("authoring_mode") == "manual"
    ]
    ai_choice_counts = [len(node.get("choices", [])) for node in ai_nodes]

    # Objective 3 concerns choices that users actually selected, not every
    # unselected option merely displayed by the model.

    selected_choice_nodes = [
        node
        for node in nodes
        if node.get("source_choice_id")
    ]

    nodes_by_id = {
        node["id"]: node
        for node in nodes
    }

    def realised_state_change(
        node: dict[str, Any],
    ) -> bool:
        parent = nodes_by_id.get(
            node.get("parent_node_id")
        )

        if not parent:
            return False

        parent_memory = parent.get(
            "memory_snapshot",
            {},
        )

        child_memory = node.get(
            "memory_snapshot",
            {},
        )

        return any(
            parent_memory.get(field)
            != child_memory.get(field)
            for field in StoryMemory.model_fields
        )

    selected_with_change = sum(
        realised_state_change(node)
        for node in selected_choice_nodes
    )

    selected_change_ratio = (
        selected_with_change
        / len(selected_choice_nodes)
        if selected_choice_nodes
        else 0.0
    )

    shared_fork = any(
        branch.get("fork_node_id")
        and branch.get("parent_branch_id")
        for branch in branches
    )

    generation_latencies = [
        float(log["elapsed_seconds"])
        for log in logs
    ]

    average_latency = (
        mean(generation_latencies)
        if generation_latencies
        else None
    )

    median_latency = (
        median(generation_latencies)
        if generation_latencies
        else None
    )

    complete_snapshots = (
        all(
            all(
                field in node.get("memory_snapshot", {})
                for field in StoryMemory.model_fields
            )
            for node in nodes
        )
        if nodes
        else False
    )

    return {
        "objective_1": {
            "description": "Create and edit alternative story branches.",
            "scene_count": len(nodes),
            "manual_scene_count": len(manual_nodes),
            "ai_scene_count": len(ai_nodes),
            "branch_count": len(branches),
            "all_ai_nodes_have_three_choices": (
                bool(ai_nodes)
                and all(count == 3 for count in ai_choice_counts)
            ),
            "two_or_more_branches": len(branches) >= 2,
            "branches_share_a_starting_point": shared_fork,
            "average_generation_seconds": (
                round(average_latency, 3)
                if average_latency is not None
                else None
            ),
            "generation_sample_count": len(generation_latencies),
            "review_required_generation_count": sum(
                not bool(log.get("validation_pass"))
                for log in all_logs
            ),
            "median_generation_seconds": (
                round(median_latency, 3)
                if median_latency is not None
                else None
            ),
            "target_generation_seconds": 15,
            "latency_target_met": (
                average_latency is not None
                and average_latency < 15
            ),
        },
        "objective_2": {
            "description": "Maintain branch-specific narrative continuity.",
            "node_memory_snapshot_count": sum(
                bool(node.get("memory_snapshot"))
                for node in nodes
            ),
            "all_nodes_have_complete_memory_schema": complete_snapshots,
            "fact_preservation_target": 0.85,
            "critical_contradiction_target": 0,
            "continuity_errors_per_branch_target": 2,
            "note": (
                "Schema completeness is implementation evidence only; the "
                "continuity targets require a separate labelled evaluation."
            ),
        },
        "objective_3": {
            "description": (
                "Make selected-choice consequences visible and meaningful."
            ),
            "selected_choice_count": len(selected_choice_nodes),
            "selected_choices_with_state_change": selected_with_change,
            "state_changing_choice_ratio": round(selected_change_ratio, 3),
            "target_ratio": 0.80,
            "target_met": selected_change_ratio >= 0.80,
        },
    }
