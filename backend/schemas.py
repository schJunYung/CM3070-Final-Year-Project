from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


InteractionType = Literal["write", "continue", "do", "say", "ask"]
AuthoringMode = Literal["manual", "ai"]
ChoiceType = Literal["do", "say", "ask", "continue"]
ThreatLevel = Literal["Low", "Medium", "High"]
StoryElementMode = Literal["off", "guided", "strict"]
StoryElementPreference = Literal["available", "preferred"]
StoryElementCategory = Literal[
    "creature",
    "hazard",
    "item",
    "location",
    "npc",
    "encounter",
]


def _normalise_threat(value: object) -> object:
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    key = value.strip().casefold().replace("_", "-")
    mapping = {
        "low": "Low",
        "slight": "Low",
        "minor": "Low",
        "medium": "Medium",
        "moderate": "Medium",
        "high": "High",
        "medium-high": "High",
        "severe": "High",
    }
    return mapping.get(key, value)


class StrictAIModel(BaseModel):
    """Reject AI-generated fields that SekAI does not understand."""

    model_config = ConfigDict(extra="forbid")


class StoryMemory(BaseModel):
    location: str = "Unknown"
    active_characters: list[str] = Field(default_factory=list)
    inventory: list[str] = Field(default_factory=list)
    relationships: dict[str, str] = Field(default_factory=dict)
    goals: list[str] = Field(default_factory=list)
    unresolved_clues: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    threat: ThreatLevel = "Low"

    @field_validator("threat", mode="before")
    @classmethod
    def normalise_threat(cls, value: object) -> object:
        return _normalise_threat(value)


class StateDelta(StrictAIModel):
    location_set: str | None = None
    active_characters_add: list[str] = Field(default_factory=list)
    active_characters_remove: list[str] = Field(default_factory=list)
    inventory_add: list[str] = Field(default_factory=list)
    inventory_remove: list[str] = Field(default_factory=list)
    relationships_set: dict[str, str] = Field(default_factory=dict)
    goals_add: list[str] = Field(default_factory=list)
    goals_complete: list[str] = Field(default_factory=list)
    unresolved_clues_add: list[str] = Field(default_factory=list)
    unresolved_clues_resolve: list[str] = Field(default_factory=list)
    decisions_add: list[str] = Field(default_factory=list)
    threat_set: ThreatLevel | None = None

    @field_validator("threat_set", mode="before")
    @classmethod
    def normalise_threat_set(cls, value: object) -> object:
        return _normalise_threat(value)

    @model_validator(mode="after")
    def reject_conflicting_changes(
        self,
    ) -> "StateDelta":

        def overlap(
            added: list[str],
            removed: list[str],
        ) -> set[str]:
            added_keys = {
                value.strip().casefold()
                for value in added
                if value.strip()
            }

            removed_keys = {
                value.strip().casefold()
                for value in removed
                if value.strip()
            }

            return (
                added_keys
                & removed_keys
            )

        conflicts = {
            "active_characters":
                overlap(
                    self.active_characters_add,
                    self.active_characters_remove,
                ),

            "inventory":
                overlap(
                    self.inventory_add,
                    self.inventory_remove,
                ),

            "goals":
                overlap(
                    self.goals_add,
                    self.goals_complete,
                ),

            "unresolved_clues":
                overlap(
                    self.unresolved_clues_add,
                    self.unresolved_clues_resolve,
                ),
        }

        invalid = {
            field: sorted(values)
            for field, values
            in conflicts.items()
            if values
        }

        if invalid:
            raise ValueError(
                "StateDelta contains "
                "conflicting changes: "
                f"{invalid}"
            )

        return self

    def has_meaningful_change(self) -> bool:
        return any(
            value not in (None, "", [], {})
            for value in self.model_dump().values()
        )


class StoryChoice(StrictAIModel):
    """
    A prepared action the user may select for the
    next story node.

    A choice contains intent only. SekAI does not
    predict its consequence in advance. The realised
    consequence is recorded as the applied_delta of
    the next generated node.
    """

    id: str = Field(
        default_factory=lambda:
            f"choice_{uuid4().hex}"
    )

    action_type: ChoiceType = "do"

    label: str = Field(
        min_length=1,
        max_length=240,
    )

class AIGeneratedDraft(StrictAIModel):
    title: str = Field(min_length=1, max_length=180)
    story_text: str = Field(min_length=30, max_length=12_000)
    applied_delta: StateDelta = Field(default_factory=StateDelta)
    choices: list[StoryChoice] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_ai_contract(self) -> "AIGeneratedDraft":
        required_top_level = {
            "title",
            "story_text",
            "applied_delta",
            "choices",
        }
        missing_top_level = required_top_level - self.model_fields_set
        if missing_top_level:
            raise ValueError(
                "Generated response is missing required fields: "
                + ", ".join(sorted(missing_top_level))
            )

        required_choice_fields = {
            "action_type",
            "label",
        }
        for index, choice in enumerate(self.choices, start=1):
            missing_choice_fields = (
                required_choice_fields - choice.model_fields_set
            )
            if missing_choice_fields:
                raise ValueError(
                    f"Generated choice {index} is missing required fields: "
                    + ", ".join(sorted(missing_choice_fields))
                )

        labels = [choice.label.strip().casefold() for choice in self.choices]
        if len(set(labels)) != len(labels):
            raise ValueError("Generated choices must have distinct labels.")

        choice_ids = [choice.id for choice in self.choices]
        if len(set(choice_ids)) != len(choice_ids):
            raise ValueError("Generated choices must have distinct IDs.")

        return self


class StoryElementFields(BaseModel):
    """Editable fields shared by built-in and user-created story elements."""

    model_config = ConfigDict(extra="forbid")

    pack: str = Field(default="Custom", min_length=1, max_length=80)
    category: StoryElementCategory
    name: str = Field(min_length=1, max_length=160)
    tags: list[str] = Field(default_factory=list, max_length=20)
    threat: ThreatLevel | None = None
    description: str = Field(default="", max_length=1_500)
    details: dict[str, str | list[str]] = Field(default_factory=dict)

    @field_validator("threat", mode="before")
    @classmethod
    def normalise_element_threat(cls, value: object) -> object:
        if value in ("", None):
            return None
        return _normalise_threat(value)


class StoryElementDefinition(StoryElementFields):
    """Validated story-element definition loaded from the factory catalogue."""

    id: str = Field(min_length=1, max_length=120)


class StoryElementWriteRequest(StoryElementFields):
    """Request body used when creating or editing a library element."""


class StoryElementSelection(BaseModel):
    """Project preference for one database-backed story element."""

    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(min_length=1, max_length=120)
    preference: StoryElementPreference = "available"


class CustomStoryElement(StoryElementDefinition):
    """
    Legacy compatibility model.

    Older SekAI databases stored complete custom elements inside a project.
    The database migration imports those records into the global library.
    """

    preference: StoryElementPreference = "available"


class ProjectSetupBase(BaseModel):
    """Shared schema for creating and updating a story project."""

    title: str = Field(
        min_length=1,
        max_length=180,
    )

    genre: str = Field(
        default="Fantasy",
        max_length=120,
    )

    tone: str = Field(
        default="Adventurous",
        max_length=120,
    )

    world_summary: str = Field(
        default="",
        max_length=6_000,
    )

    main_objective: str = Field(
        default="",
        max_length=2_000,
    )

    starting_scene_title: str = Field(
        default="Starting scene",
        max_length=180,
    )

    starting_scene_text: str = Field(
        min_length=20,
        max_length=12_000,
    )

    initial_memory: StoryMemory

    story_element_mode: StoryElementMode = "guided"
    story_elements: list[StoryElementSelection] = Field(
        default_factory=list,
        max_length=64,
    )

    @model_validator(mode="after")
    def validate_story_element_configuration(self) -> "ProjectSetupBase":
        selected_ids = [
            selection.element_id
            for selection in self.story_elements
        ]

        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(
                "story_elements cannot contain duplicate element IDs."
            )

        return self


class ProjectCreateRequest(ProjectSetupBase):
    """Request body used when creating a new story project."""


class ProjectUpdateRequest(ProjectSetupBase):
    """Request body used when updating an existing story project."""


class GenerateStoryRequest(BaseModel):
    project_id: str
    branch_id: str
    parent_node_id: str
    selected_choice_id: str | None = None
    model: str = Field(default="qwen2.5:1.5b", min_length=1, max_length=180)
    temperature: float = Field(default=0.5, ge=0.0, le=1.5)
    interaction_type: InteractionType = "continue"
    player_input: str = Field(default="", max_length=2_000)
    gm_instruction: str = Field(default="", max_length=3_000)

    @model_validator(mode="after")
    def validate_player_input(self) -> "GenerateStoryRequest":
        if (
            self.selected_choice_id is None
            and self.interaction_type
            in {"do", "say", "ask"}
            and not self.player_input.strip()
        ):
            raise ValueError(
                "player_input is required when interaction_type is "
                f"'{self.interaction_type}'."
            )
        return self


ImageSourceKind = Literal[
    "manual",
    "ai",
]

ImageModelKey = Literal[
    "sd15-int8",
    "sd-turbo",
]


class SceneImageGenerationRequest(BaseModel):
    project_id: str
    branch_id: str
    parent_node_id: str
    source_kind: ImageSourceKind
    model_key: ImageModelKey = "sd15-int8"
    title: str = Field(
        min_length=1,
        max_length=180,
    )
    story_text: str = Field(
        min_length=1,
        max_length=12_000,
    )
    applied_delta: StateDelta = Field(default_factory=StateDelta)
    prompt_override: str = Field(
        default="",
        max_length=2_000,
    )
    seed: int = Field(
        default=42,
        ge=0,
        le=2_147_483_647,
    )

class SceneImageGenerationResponse(BaseModel):
    image_generation_log_id: int
    image_id: str
    source_kind: ImageSourceKind
    model_key: ImageModelKey
    model_id: str
    device: str
    precision: str
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    steps: int
    guidance_scale: float
    model_load_seconds: float
    inference_seconds: float
    total_seconds: float
    cold_start: bool
    file_path: str
    image_url: str


class NodeSceneImageRequest(BaseModel):
    """Generate or replace the image attached to an existing node."""

    model_key: ImageModelKey = "sd15-int8"

    prompt_override: str = Field(
        default="",
        max_length=2_000,
    )

    seed: int = Field(
        default=42,
        ge=0,
        le=2_147_483_647,
    )


class SaveNodeRequest(BaseModel):
    project_id: str
    source_branch_id: str
    parent_node_id: str
    selected_choice_id: str | None = None
    branch_name: str | None = Field(default=None, max_length=180)
    force_fork: bool = False
    title: str = Field(min_length=1, max_length=180)
    story_text: str = Field(min_length=1, max_length=12_000)
    applied_delta: StateDelta = Field(default_factory=StateDelta)
    choices: list[StoryChoice] = Field(default_factory=list, max_length=6)
    authoring_mode: AuthoringMode = "manual"
    interaction_type: InteractionType = "write"
    interaction_text: str = Field(default="", max_length=2_000)
    generated_by_model: str | None = Field(default=None, max_length=180)
    generation_log_id: int | None = None
    review_override: bool = False
    scene_image_log_id: int | None = None

    @model_validator(mode="after")
    def validate_ai_save_contract(
        self,
    ) -> "SaveNodeRequest":
        if self.authoring_mode == "ai":
            if len(self.choices) != 3:
                raise ValueError(
                    "A reviewed AI scene must contain "
                    "exactly three choices."
                )

            labels = [
                choice.label.strip().casefold()
                for choice in self.choices
            ]

            if len(set(labels)) != 3:
                raise ValueError(
                    "A reviewed AI scene must contain "
                    "three distinct choice labels."
                )

            choice_ids = [
                choice.id
                for choice in self.choices
            ]

            if len(set(choice_ids)) != 3:
                raise ValueError(
                    "A reviewed AI scene must contain "
                    "three distinct choice IDs."
                )

        if (
            self.authoring_mode != "ai"
            and self.generation_log_id is not None
        ):
            raise ValueError(
                "generation_log_id is only valid "
                "for AI-authored scenes."
            )

        return self


class UpdateNodeRequest(BaseModel):
    title: str = Field(min_length=1, max_length=180)
    story_text: str = Field(min_length=1, max_length=12_000)
    choices: list[StoryChoice] = Field(default_factory=list, max_length=6)
    applied_delta: StateDelta | None = None


ValidationTarget = Literal[
    "title",
    "story_text",
    "choices",
    "applied_delta",
]


class GenerationIssue(BaseModel):
    """Structured validation feedback that the frontend can map to an editor."""

    stage: Literal[
        "length",
        "title",
        "state",
        "state_transition",
        "state_grounding",
        "duplicate",
        "choices",
        "repetition",
        "semantic_repetition",
        "continuity",
    ]
    severity: Literal["warning", "blocking"] = "warning"
    target: ValidationTarget | None = None
    field: str | None = Field(default=None, max_length=120)
    value: str | None = Field(default=None, max_length=1_000)
    message: str = Field(min_length=1, max_length=2_000)
    suggestion: str | None = Field(default=None, max_length=2_000)
    can_override: bool = False


class DraftValidationRequest(BaseModel):
    """Current human-edited AI draft submitted for deterministic preflight checks."""

    project_id: str
    branch_id: str
    parent_node_id: str
    title: str = Field(min_length=1, max_length=180)
    story_text: str = Field(min_length=1, max_length=12_000)
    applied_delta: StateDelta = Field(default_factory=StateDelta)
    choices: list[StoryChoice] = Field(default_factory=list, max_length=6)


class DraftValidationResponse(BaseModel):
    can_save: bool
    has_warnings: bool
    word_count: int
    blocking_count: int
    warning_count: int
    issues: list[GenerationIssue] = Field(default_factory=list)


class GenerationMetrics(BaseModel):
    requested_model: str
    model: str
    fallback_used: bool
    semantic_check_completed: bool
    continuity_check_completed: bool
    continuity_model: str | None = None
    continuity_checked_facts: int = 0
    continuity_max_contradiction: float | None = None
    requested_temperature: float
    effective_temperature: float
    elapsed_seconds: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    tokens_per_second: float | None = None
    attempts: int = 1


class GenerateStoryResponse(BaseModel):
    status: Literal["accepted", "review_required"] = "accepted"
    validation_pass: bool = True
    validation_issues: list[GenerationIssue] = Field(default_factory=list)
    draft: AIGeneratedDraft
    metrics: GenerationMetrics
    generation_log_id: int
    authoritative_context: dict[str, Any]


class TranscriptionReviewRequest(BaseModel):
    """Human-reviewed text produced from one recorded voice instruction."""

    reviewed_text: str = Field(default="", max_length=3_000)


class TranscriptionResponse(BaseModel):
    """Comparable local STT result returned by either experimental backend."""

    transcription_log_id: int
    engine: str
    model: str
    device: str
    compute_type: str
    target_field: Literal["manual_scene", "player_input", "gm_instruction"]
    text: str
    language: str | None = None
    language_probability: float | None = None
    audio_seconds: float
    audio_bytes: int
    model_load_seconds: float
    inference_seconds: float
    total_seconds: float
    real_time_factor: float | None = None
    cold_start: bool = False

