from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schemas import StateDelta, StoryMemory


MODEL_ID = os.getenv(
    "SEKAI_NLI_MODEL_ID",
    "cross-encoder/nli-deberta-v3-small",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "nli-deberta-v3-small"
MODEL_PATH = Path(
    os.getenv(
        "SEKAI_NLI_MODEL_PATH",
        str(DEFAULT_MODEL_DIR),
    )
)

CONTRADICTION_THRESHOLD = float(
    os.getenv(
        "SEKAI_NLI_CONTRADICTION_THRESHOLD",
        "0.90",
    )
)

MAX_FACTS = int(
    os.getenv(
        "SEKAI_NLI_MAX_FACTS",
        "32",
    )
)

# The model card documents this output order.
LABELS = ("contradiction", "entailment", "neutral")


class ContinuityModelUnavailableError(RuntimeError):
    """Raised when the local NLI model cannot be loaded."""


@dataclass(frozen=True)
class ContinuityContradiction:
    fact: str
    contradiction_score: float
    entailment_score: float
    neutral_score: float


@dataclass(frozen=True)
class ContinuityReview:
    model: str
    checked_facts: int
    contradictions: tuple[ContinuityContradiction, ...]
    max_contradiction_score: float | None

    @property
    def passed(self) -> bool:
        return not self.contradictions


def _clean(value: object) -> str:
    return str(value or "").strip()


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        clean = value.strip()
        key = clean.casefold()

        if not clean or key in seen:
            continue

        seen.add(key)
        result.append(clean)

    return result


def memory_to_facts(
    memory: StoryMemory,
    applied_delta: StateDelta | None = None,
) -> list[str]:
    """Convert proposed child-state memory into NLI hypotheses."""
    facts: list[str] = []

    location = _clean(memory.location)
    if location and location.casefold() != "unknown":
        facts.append(f"The current location is {location}.")

    for character in memory.active_characters:
        character_clean = _clean(character)
        if character_clean:
            facts.append(
                f"{character_clean} is currently present and active in the scene."
            )

    for item in memory.inventory:
        item_clean = _clean(item)
        if item_clean:
            facts.append(f"The current inventory contains {item_clean}.")

    for character, status in memory.relationships.items():
        character_clean = _clean(character)
        status_clean = _clean(status)
        if character_clean and status_clean:
            facts.append(
                f"The relationship with {character_clean} is {status_clean}."
            )

    for goal in memory.goals:
        goal_clean = _clean(goal)
        if goal_clean:
            facts.append(f"The active story goal is: {goal_clean}.")

    for clue in memory.unresolved_clues:
        clue_clean = _clean(clue)
        if clue_clean:
            facts.append(f"An unresolved story clue is: {clue_clean}.")

    for decision in memory.decisions:
        decision_clean = _clean(decision)
        if decision_clean:
            facts.append(f"A prior recorded decision is: {decision_clean}.")

    threat = _clean(memory.threat)
    if threat:
        facts.append(f"The current threat level is {threat}.")

    priority_facts: list[str] = []

    if applied_delta is not None:
        for character in applied_delta.active_characters_remove:
            character_clean = _clean(character)
            if character_clean:
                priority_facts.append(
                    f"{character_clean} is no longer an active character "
                    "in the scene."
                )

        for item in applied_delta.inventory_remove:
            item_clean = _clean(item)
            if item_clean:
                priority_facts.append(
                    f"The current inventory does not contain {item_clean}."
                )

        for goal in applied_delta.goals_complete:
            goal_clean = _clean(goal)
            if goal_clean:
                priority_facts.append(f"The goal '{goal_clean}' is complete.")

        for clue in applied_delta.unresolved_clues_resolve:
            clue_clean = _clean(clue)
            if clue_clean:
                priority_facts.append(f"The clue '{clue_clean}' is resolved.")

    return _deduplicate([*priority_facts, *facts])[:MAX_FACTS]


@lru_cache(maxsize=1)
def get_nli_model() -> Any:
    """Load the local DeBERTa CrossEncoder once and reuse it."""
    has_model_files = (
        MODEL_PATH.is_dir()
        and any(
            item.is_file() and not item.name.startswith(".")
            for item in MODEL_PATH.iterdir()
        )
    )

    if not has_model_files:
        raise ContinuityModelUnavailableError(
            "The local continuity model is not installed. Run "
            "'.\\.venv\\Scripts\\python.exe "
            ".\\scripts\\setup_models.py setup --only nli' "
            "from the SekAI project folder first."
        )

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise ContinuityModelUnavailableError(
            "sentence-transformers is not installed. "
            "Install SekAI dependencies with "
            "'.\\.venv\\Scripts\\python.exe -m pip "
            "install -r requirements.txt'."
        ) from error

    try:
        return CrossEncoder(
            str(MODEL_PATH),
            device="cpu",
            local_files_only=True,
        )
    except Exception as error:
        raise ContinuityModelUnavailableError(
            f"Could not load the local NLI model from {MODEL_PATH}: {error}"
        ) from error


def review_story_continuity(
    candidate_story: str,
    proposed_memory: StoryMemory,
    applied_delta: StateDelta,
) -> ContinuityReview:
    """Check candidate prose against authoritative proposed child-state facts."""
    story = candidate_story.strip()
    if not story:
        raise ValueError("Candidate story is empty.")

    facts = memory_to_facts(proposed_memory, applied_delta)

    if not facts:
        return ContinuityReview(
            model=MODEL_ID,
            checked_facts=0,
            contradictions=(),
            max_contradiction_score=None,
        )

    model = get_nli_model()
    pairs = [(story, fact) for fact in facts]

    scores = model.predict(
        pairs,
        batch_size=min(8, len(pairs)),
        show_progress_bar=False,
        apply_softmax=True,
        convert_to_numpy=True,
        device="cpu",
    )

    if len(scores) != len(facts):
        raise ValueError(
            "The NLI model returned an unexpected number of predictions."
        )

    contradictions: list[ContinuityContradiction] = []
    max_score = 0.0

    contradiction_index = LABELS.index("contradiction")
    entailment_index = LABELS.index("entailment")
    neutral_index = LABELS.index("neutral")

    for fact, row in zip(facts, scores):
        if len(row) != len(LABELS):
            raise ValueError(
                "The NLI model did not return three class scores."
            )

        contradiction_score = float(row[contradiction_index])
        entailment_score = float(row[entailment_index])
        neutral_score = float(row[neutral_index])
        max_score = max(max_score, contradiction_score)

        if contradiction_score >= CONTRADICTION_THRESHOLD:
            contradictions.append(
                ContinuityContradiction(
                    fact=fact,
                    contradiction_score=round(contradiction_score, 4),
                    entailment_score=round(entailment_score, 4),
                    neutral_score=round(neutral_score, 4),
                )
            )

    return ContinuityReview(
        model=MODEL_ID,
        checked_facts=len(facts),
        contradictions=tuple(contradictions),
        max_contradiction_score=round(max_score, 4),
    )
