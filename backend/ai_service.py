from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import time

from uuid import uuid4

from difflib import SequenceMatcher
from typing import Any

import httpx
from fastapi import HTTPException
from pydantic import ValidationError

from . import db
from .continuity_service import (
    ContinuityModelUnavailableError,
    review_story_continuity,
)
from .content_library import select_prompt_elements
from .schemas import (
    AIGeneratedDraft,
    GenerateStoryRequest,
    GenerateStoryResponse,
    GenerationIssue,
    GenerationMetrics,
    StateDelta,
    StoryMemory,
)
from .similarity import (
    retrieve_relevant_nodes,
    semantic_repetition_reason,
)
from .state import apply_state_delta
from .validation_service import (
    STORY_MAX_WORDS,
    STORY_MIN_WORDS,
    blocking_issues,
    clean_redundant_delta,
    format_issues_for_retry,
    story_word_count,
    validate_reviewed_draft,
)

from .config import (
    EVALUATION_MODE,
    PIPELINE_VERSION,
)


FALLBACK_MODEL = os.getenv(
    "OLLAMA_FALLBACK_MODEL",
    "qwen3:1.7b",
)

REQUIRE_NLI = os.getenv(
    "SEKAI_REQUIRE_NLI",
    "0",
).strip().casefold() in {"1", "true", "yes", "on"}



OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
logger = logging.getLogger("sekai.ai")


def _collapse_whitespace(value: str) -> str:
    """Replace repeated spaces, tabs, and line breaks with one space."""
    return re.sub(r"\s+", " ", value).strip()


def _context_excerpt(value: str, limit: int = 1_200) -> str:
    """Return the most recent part of a scene for continuation context."""
    clean = _collapse_whitespace(value)
    if len(clean) <= limit:
        return clean
    return "…" + clean[-limit:]


def _first_sentence(value: str) -> str:
    """Extract an opening sentence for novelty comparison."""
    clean = _collapse_whitespace(value)
    if not clean:
        return ""
    return re.split(r"(?<=[.!?])\s+", clean, maxsplit=1)[0][:240]


def _is_low_information_scene(value: str) -> bool:
    """Detect placeholder text such as repeated digits or repeated words."""
    clean = _collapse_whitespace(value)
    if not clean:
        return True

    characters = re.sub(r"[^a-z0-9]", "", clean.casefold())
    if len(characters) >= 80 and len(set(characters)) <= 4:
        return True

    words = re.findall(r"[a-z0-9']+", clean.casefold())
    if len(words) >= 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.20:
            return True

    return False


def _normalise_for_similarity(value: str) -> str:
    """Normalise punctuation and spacing before comparing two scenes."""
    return " ".join(re.findall(r"[a-z0-9']+", value.casefold()))


def _word_ngrams(value: str, size: int = 4) -> set[str]:
    """Convert text into overlapping word phrases."""
    words = _normalise_for_similarity(value).split()
    if len(words) < size:
        return set()
    return {
        " ".join(words[index:index + size])
        for index in range(len(words) - size + 1)
    }


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def repetition_reason(
    candidate_story: str,
    reference_nodes: list[dict[str, Any]],
) -> str | None:
    """Explain why a generated scene is too similar, or return None."""
    candidate = _normalise_for_similarity(candidate_story)
    candidate_ngrams = _word_ngrams(candidate_story)
    candidate_opening = _normalise_for_similarity(_first_sentence(candidate_story))

    for node in reference_nodes:
        previous_story = node.get("story_text", "")
        previous = _normalise_for_similarity(previous_story)
        if not candidate or not previous:
            continue

        if candidate == previous:
            return (
                "The generated scene is an exact duplicate of "
                f"the earlier node '{node['title']}'."
            )

        sequence_similarity = SequenceMatcher(
            None, candidate, previous, autojunk=False
        ).ratio()
        phrase_similarity = _jaccard_similarity(
            candidate_ngrams, _word_ngrams(previous_story)
        )
        previous_opening = _normalise_for_similarity(
            _first_sentence(previous_story)
        )
        opening_similarity = (
            SequenceMatcher(
                None, candidate_opening, previous_opening, autojunk=False
            ).ratio()
            if candidate_opening and previous_opening
            else 0.0
        )

        if sequence_similarity >= 0.82:
            return (
                f"The generated scene is {sequence_similarity:.0%} textually "
                f"similar to '{node['title']}'."
            )
        if phrase_similarity >= 0.72:
            return (
                f"The generated scene reuses {phrase_similarity:.0%} of the "
                f"earlier four-word phrases from '{node['title']}'."
            )
        if opening_similarity >= 0.90:
            return f"The generated scene repeats the opening of '{node['title']}'."

    return None



SYSTEM_PROMPT = """
You are SekAI, a local assistant helping a tabletop RPG Game Master create ONE next scene.

Follow these priorities in order:

1. RESOLVE THE USER'S ACTION
The mandatory interaction is the event this scene must perform.
Do not merely mention, postpone or repeat it.
When a REQUIRED USER REFINEMENT is supplied, every concrete requested detail in it is mandatory.
Do not reduce a multi-part direction to only one of its requested details.
DO: perform the attempted action and show its immediate result.
SAY: include the supplied speech and show a response.
ASK: ask the supplied question and provide an answer, refusal or reaction.
CONTINUE: perform the supplied direction or introduce the next event.

2. OBEY THE GAME MASTER
The Game Master direction applies to the scene being generated NOW.
If it says "next scene" or "next node", that means THIS generated scene.

3. PRESERVE CONTINUITY
current_memory_read_only is the authoritative state BEFORE this scene.
Do not claim that a possession, character, relationship, goal or prior event
already existed unless the current branch context supports that claim.
New characters, items, goals, clues or locations may be introduced when this
scene explicitly introduces, discovers, acquires or establishes them, and the
corresponding change must be recorded in applied_delta.
Do not import facts from another branch.

4. USE STORY ELEMENTS AS OPTIONAL SUPPORT
story_element_policy contains reusable GM-selected story elements.
Explicit player instructions and explicit Game Master instructions ALWAYS
outrank the Story Element Library.
Never replace a user-requested creature, item, hazard, location, NPC, event
or outcome with a library element.
In guided mode, use a supplied element only when it fits the requested scene
and fills a detail the user did not specify. It is valid to ignore the library.
In strict mode, when an unspecified configured element is needed, prefer one
of the supplied elements instead of inventing a different one.
An explicit user or GM instruction may still introduce something outside the library.
Library fields such as possible_rewards, drops or obtain are POSSIBILITIES only.
They are not current possessions and must never be copied into inventory_add
unless story_text explicitly shows the player obtaining that item in this scene.

5. ADVANCE THE STORY
Begin after the selected scene ends.
Do not replay, summarise or paraphrase an earlier scene.
The first paragraph must perform a new action, response, discovery or consequence.
Do not reuse earlier titles, openings or memorable dialogue.

6. KEEP STATE CHANGES GROUNDED
- current_memory_read_only describes the state before this scene.
- applied_delta describes only differences caused by this scene.
- Do not restate the complete current state in applied_delta.
- Every non-empty applied_delta field must be explicitly supported by story_text.
- Never add an item unless it is obtained, received, found, created or otherwise acquired.
- Never remove an item unless it is lost, consumed, destroyed, given away or otherwise removed.
- Never change location unless movement or arrival actually occurs.
- Never change a relationship without an event or interaction that justifies the change.
- Never add a goal unless a new objective is established.
- Never complete a goal unless the objective is achieved or abandoned.
- Never change threat level unless the level of danger meaningfully changes.
- Never re-add state already present in current_memory_read_only.
- Never remove state that is not present in current_memory_read_only.
- Do not manufacture state changes merely to populate applied_delta.
- A scan result, observation, clue or report is not automatically an inventory item.
- Prefer an empty applied_delta when uncertain; a valid scene does not need a state change.
- If no tracked state changes, return an empty applied_delta.

The writer automatically tracks only:
- location changes
- inventory gains or losses
- relationship changes
- goal additions or completions
- threat-level changes

Other memory fields may appear in current_memory_read_only,
but they are read-only context for the writer. Do not attempt
to modify them in applied_delta.

7. CREATE THREE DISTINCT NEXT CHOICES
Each choice is an unresolved future action.
Do not describe something that already happened in this scene.
DO must describe an action.
ASK must be a question or request for information.
SAY must describe something the player may say.
CONTINUE must describe allowing events to progress.
The three choices must lead in meaningfully different directions.

story_text MUST contain 50 to 150 words.
Return only the required structured response.
""".strip()


JSON_OUTPUT_CONTRACT = """
Return exactly one JSON object with these four top-level fields:

{
  "title": string,
  "story_text": string,
  "applied_delta": object,
  "choices": array
}
story_text MUST contain 50 to 150 words.
Use those words to resolve the requested interaction and advance the scene.
Do not satisfy the length requirement by summarising previous scenes or repeating information.

applied_delta contains ONLY changes that actually happen in this scene.

Allowed applied_delta fields and exact types:

location_set:
  string or null

inventory_add:
  array of strings

inventory_remove:
  array of strings

relationships_set:
  object mapping character-name strings to relationship-status strings

goals_add:
  array of strings

goals_complete:
  array of strings

threat_set:
  "Low", "Medium", "High", or null

Important StateDelta rules:
- Never put objects inside a string array.
- inventory_add and inventory_remove contain item names only.
- Never include quantity objects.
- location_set is one string, never an array or object.
- Do not use full-memory fields such as location, inventory, relationships,
  goals, unresolved_clues, decisions, or threat inside applied_delta.
- Never invent alternative field names such as relationship_set,
  goal_complete, choices_add, memory_snapshot, or journal_picked_up.
- Omit fields that do not change.

If this scene causes no tracked state changes, return:

"applied_delta": {}

Only include a field in applied_delta when the value actually changes
from current_memory_read_only during THIS generated scene.

Never copy values from an example into applied_delta.
Never copy existing inventory, characters, locations, relationships,
goals or clues into applied_delta simply because they appear in the
current memory.

choices must contain exactly three objects.

Each choice contains exactly:
{
  "action_type": "do" | "say" | "ask" | "continue",
  "label": string
}

Choices describe possible NEXT actions only. They do not contain predicted
consequences or future state changes.

Do not include expected_consequence, projected_delta, Markdown fences,
comments, explanatory text, or additional top-level fields.
""".strip()

def _build_ollama_response_schema() -> dict[str, Any]:
    """Build the strict schema Ollama must follow when generating a draft."""
    schema = AIGeneratedDraft.model_json_schema()

    # These fields are required by AIGeneratedDraft.validate_ai_contract().
    schema["required"] = [
        "title",
        "story_text",
        "applied_delta",
        "choices",
    ]
    # The writer only manages the core state
    # variables required for narrative consequences.
    # More subjective memory such as clues and
    # decisions remains visible to the GM but is not
    # automatically changed by the writer.
    delta_schema = (
        schema
        .get("$defs", {})
        .get("StateDelta")
    )

    if not delta_schema:
        raise RuntimeError(
            "StateDelta schema is missing "
            "from AIGeneratedDraft."
        )

    ai_managed_delta_fields = {
        "location_set",
        "inventory_add",
        "inventory_remove",
        "relationships_set",
        "goals_add",
        "goals_complete",
        "threat_set",
    }

    properties = (
        delta_schema.get(
            "properties",
            {},
        )
    )

    for field_name in list(properties):
        if (
            field_name
            not in ai_managed_delta_fields
        ):
            properties.pop(
                field_name,
                None,
            )

    # StoryChoice.id is generated by SekAI, not by the LLM.
    choice_schema = schema.get("$defs", {}).get("StoryChoice")
    if not choice_schema:
        raise RuntimeError("StoryChoice schema is missing from AIGeneratedDraft.")

    choice_schema["properties"].pop("id", None)
    choice_schema["required"] = ["action_type", "label"]

    return schema


OLLAMA_RESPONSE_SCHEMA = _build_ollama_response_schema()

def _choice_for_node(node: dict[str, Any], choice_id: str | None) -> dict[str, Any] | None:
    if not choice_id:
        return None
    return next(
        (choice for choice in node.get("choices", []) if choice.get("id") == choice_id),
        None,
    )

def _choice_prompt_view(
    choice: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Return only fields relevant to future generation.

    This also keeps legacy choices containing old
    expected_consequence/projected_delta data from
    leaking into new prompts.
    """

    if not choice:
        return None

    return {
        "id": choice.get("id"),
        "action_type": choice.get(
            "action_type",
            "continue",
        ),
        "label": str(
            choice.get(
                "label",
                "",
            )
        ).strip(),
    }

def _interaction_instruction(
    request: GenerateStoryRequest,
    selected_choice: dict[str, Any] | None,
) -> str:
    """
    Build the mandatory next-scene instruction.

    Prepared choices describe intent only. Their actual
    consequences are generated in the next node.
    """

    player_input = (
        request.player_input.strip()
    )

    instructions: list[str] = []

    # -------------------------------------------------
    # Prepared choice selected from the parent node.
    # -------------------------------------------------
    if selected_choice:
        action_type = str(
            selected_choice.get(
                "action_type",
                "continue",
            )
        ).upper()

        label = str(
            selected_choice.get(
                "label",
                "",
            )
        ).strip()

        instructions.append(
            "REQUIRED SELECTED CHOICE:\n"
            f"{action_type}: {label}"
        )

        if player_input:
            instructions.append(
                "REQUIRED USER REFINEMENT:\n"
                f"{player_input}"
            )

            instructions.append(
                "Resolve the selected choice AND satisfy every concrete detail "
                "in the user refinement. The refinement specifies how the selected "
                "choice unfolds; it is not optional. Do not silently omit requested "
                "objects, events, characters, descriptions or outcomes."
            )
        else:
            instructions.append(
                "Resolve the selected choice directly and show its immediate result."
            )

        return "\n".join(
            instructions
        )

    # -------------------------------------------------
    # No prepared choice: free-form interaction.
    # -------------------------------------------------
    if request.interaction_type == "do":
        instructions.append(
            "The player attempts this action. "
            "Generate its immediate outcome:\n"
            f"{player_input}"
        )

    elif request.interaction_type == "say":
        instructions.append(
            "The player says:\n"
            f"{player_input}\n"
            "Show the relevant response."
        )

    elif request.interaction_type == "ask":
        instructions.append(
            "The player asks:\n"
            f"{player_input}\n"
            "Provide an answer, refusal, interruption, "
            "or relevant reaction."
        )

    elif request.interaction_type == "continue":
        if player_input:
            instructions.append(
                "Continue the story using this "
                "direction:\n"
                f"{player_input}"
            )

        else:
            instructions.append(
                "Introduce one genuinely new event "
                "that follows the selected scene."
            )

    return "\n".join(
        instructions
    )

def build_context(request: GenerateStoryRequest) -> dict[str, Any]:
    project = db.get_project(request.project_id)
    branch = db.get_branch(request.branch_id)
    parent = db.get_node(request.parent_node_id)

    if not project:
        raise HTTPException(status_code=404, detail="Story project not found.")
    if not branch or branch["project_id"] != request.project_id:
        raise HTTPException(status_code=400, detail="Selected branch is invalid.")
    if not parent or parent["project_id"] != request.project_id:
        raise HTTPException(status_code=400, detail="Selected parent node is invalid.")

    path_ids = {node["id"] for node in db.get_branch_path(branch["id"])}
    if parent["id"] not in path_ids:
        raise HTTPException(
            status_code=400,
            detail="The selected node is not on the selected branch path.",
        )

    selected_choice = _choice_for_node(parent, request.selected_choice_id)
    if request.selected_choice_id and not selected_choice:
        raise HTTPException(
            status_code=400,
            detail="The selected choice does not belong to the selected node.",
        )
    
    # Convert the stored choice into the smaller
    # representation that is safe to send to the AI.
    selected_choice_context = (_choice_prompt_view(selected_choice))

    resolved_interaction_type = (
        selected_choice_context.get("action_type")
        if selected_choice_context
        else request.interaction_type)
    
    full_path = db.get_node_path(parent["id"])

    if _is_low_information_scene(parent["story_text"]):
        raise HTTPException(
            status_code=422,
            detail=(
                "The selected node contains mostly repeated or low-information "
                "text. Edit it into a meaningful scene before AI generation."
            ),
        )

    meaningful_previous_nodes = [
        node
        for node in full_path[:-1]
        if not _is_low_information_scene(node["story_text"])
    ]
    recent_nodes = meaningful_previous_nodes[-2:]
    forbidden_titles = [node["title"] for node in full_path[-4:]]
    forbidden_openings = [
        _first_sentence(node["story_text"])
        for node in full_path[-4:]
        if _first_sentence(node["story_text"])
    ]

    return {
        "project": {
            "title": project["title"],
            "genre": project["genre"],
            "tone": project["tone"],
            "world_summary": project["world_summary"],
            "main_objective": project["main_objective"],
            "story_element_mode": project.get(
                "story_element_mode",
                "guided",
            ),
            "story_elements": project.get(
                "story_elements",
                [],
            ),
        },
        "branch": {
            "id": branch["id"],
            "name": branch["name"],
        },
        "selected_node": {
            "id": parent["id"],
            "title": parent["title"],
            "ending_excerpt": _context_excerpt(parent["story_text"], limit=1_600),
            "memory_snapshot": parent["memory_snapshot"],
        },
        "recent_story_path": [
            {
                "id": node["id"],
                "title": node["title"],
                "ending_excerpt": _context_excerpt(node["story_text"], limit=600),
                "source_choice_label": node.get("source_choice_label"),
            }
            for node in recent_nodes
        ],
        "selected_choice": selected_choice_context,
        "interaction": {
            "type": resolved_interaction_type,
            "player_input": request.player_input,
            "instruction": _interaction_instruction(
                request,
                selected_choice_context,
            ),
        },
        "game_master_instruction": request.gm_instruction.strip()
        or "Continue coherently while preserving the branch memory.",
        "novelty_constraints": {
            "do_not_reuse_titles": forbidden_titles,
            "do_not_reuse_openings": forbidden_openings,
        },
    }



def build_prompt(context: dict[str, Any]) -> str:
    selected_node = context["selected_node"]

    recent_events = [
        {
            "title": node["title"],
            "action_that_led_here": node.get("source_choice_label"),
        }
        for node in context["recent_story_path"]
    ]

    relevant_history = [
        {
            "title": node["title"],
            "reference_excerpt": node["excerpt"],
        }
        for node in context.get("retrieved_branch_context", [])
    ]

    prompt_context = {
        "story": {
            "genre": context["project"]["genre"],
            "tone": context["project"]["tone"],
            "objective": context["project"]["main_objective"],
            "world": context["project"]["world_summary"],
        },
        "story_element_policy": context.get(
            "story_element_policy",
            {"mode": "off", "elements": []},
        ),
        "current_scene": {
            "title": selected_node["title"],
            "story_text": selected_node["ending_excerpt"],
            "current_memory_read_only": selected_node["memory_snapshot"],
        },
        "recent_events": recent_events,
        "relevant_older_history": relevant_history,
        "do_not_reuse": context["novelty_constraints"],
    }

    return (
        "MANDATORY INTERACTION — THIS MUST HAPPEN IN THIS SCENE:\n"
        f"{context['interaction']['instruction']}\n\n"
        "GAME MASTER REQUIREMENT — APPLY THIS TO THIS SCENE:\n"
        f"{context['game_master_instruction']}\n\n"
        "STORY ELEMENT LIBRARY — LOWER PRIORITY THAN USER/GM INSTRUCTIONS:\n"
        "Treat supplied elements as reusable supporting knowledge. Never replace "
        "explicit user or GM details with library content.\n\n"
        "AUTHORITATIVE CONTEXT:\n"
        f"{json.dumps(prompt_context, ensure_ascii=False, indent=2)}\n\n"
        "Before answering, ensure that story_text performs the mandatory interaction "
        "and that every applied_delta change is visibly supported by story_text.\n\n"
        "STRUCTURED OUTPUT RULES:\n"
        f"{JSON_OUTPUT_CONTRACT}"
    )

async def call_ollama(
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        timeout = httpx.Timeout(
            timeout_seconds,
            connect=5.0,
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    except httpx.ConnectError as error:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama.",
        ) from error

    except httpx.TimeoutException as error:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Model generation exceeded "
                f"{timeout_seconds:.0f} seconds."
            ),
        ) from error

    except httpx.HTTPStatusError as error:
        response_text = error.response.text[:2_000]

        if (
            error.response.status_code == 400
            and "failed to parse grammar"
            in response_text.casefold()
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Ollama could not initialise JSON-constrained "
                    "generation. Restart or update Ollama, or temporarily "
                    "remove the format setting while diagnosing the issue. "
                    f"Ollama response: {response_text}"
                ),
            ) from error

        raise HTTPException(
            status_code=502,
            detail=(
                "Ollama rejected the request: "
                f"{response_text}"
            ),
        ) from error

async def generate_story(request: GenerateStoryRequest) -> GenerateStoryResponse:
    """Generate a story with the selected model, then fall back if needed."""

    generation_request_id = (f"generation_{uuid4().hex}")

    started = time.perf_counter()
    
    context = build_context(request)
    parent_memory = StoryMemory.model_validate(
        context["selected_node"]["memory_snapshot"]
    )
    full_path = db.get_node_path(request.parent_node_id)

    recent_context_ids = {
        node["id"]
        for node in context["recent_story_path"]
    }

    retrieval_candidates = [
        node
        for node in full_path[:-1]
        if node["id"] not in recent_context_ids
        and not _is_low_information_scene(node["story_text"])
    ]

    effective_temperature = min(request.temperature, 0.8)

    model_candidates = [
        {
            "name": request.model,
            "timeout": 150.0,
            "num_predict": 600,
        }
    ]

    selected_choice = context.get("selected_choice") or {}

    retrieval_query = "\n".join(
        value
        for value in [
            str(selected_choice.get("label", "")).strip(),
            request.player_input.strip(),
            request.gm_instruction.strip(),
            context["project"]["main_objective"].strip(),
        ]
        if value
    )
    try:
        retrieved_nodes = (
            await retrieve_relevant_nodes(
                retrieval_query,
                retrieval_candidates,
                top_k=2,
            )
        )

    except (
        httpx.HTTPError,
        ValueError,
    ) as error:

        if EVALUATION_MODE:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Semantic retrieval is "
                    "required in evaluation mode "
                    "but EmbeddingGemma failed: "
                    f"{error}"
                ),
            ) from error

        logger.warning(
            "Semantic branch retrieval "
            "could not run: %s",
            error,
        )

        retrieved_nodes = []

    context["retrieved_branch_context"] = [
        {
            "title": node["title"],
            "excerpt": _context_excerpt(node["story_text"], limit=300),
            "similarity": round(node["retrieval_score"],3,),
        }

        for node
        in retrieved_nodes
    ]

    element_query = "\n".join(
        value
        for value in [
            str(selected_choice.get("label", "")).strip(),
            request.player_input.strip(),
            request.gm_instruction.strip(),
            context["selected_node"]["ending_excerpt"].strip(),
            str(
                context["selected_node"]["memory_snapshot"].get(
                    "location",
                    "",
                )
            ).strip(),
        ]
        if value
    )

    recent_story_text = "\n".join(
        node["story_text"]
        for node in full_path[-3:]
    )

    selected_element_ids = [
        str(
            selection.get(
                "element_id",
                "",
            )
        )
        for selection
        in context["project"].get(
            "story_elements",
            [],
        )
        if selection.get("element_id")
    ]

    active_library_elements = (
        db.get_story_elements_by_ids(
            selected_element_ids
        )
    )

    context["story_element_policy"] = {
        "mode": context["project"].get(
            "story_element_mode",
            "guided",
        ),
        "elements": select_prompt_elements(
            context["project"],
            active_library_elements,
            element_query,
            avoid_text=recent_story_text,
        ),
    }

    if request.model != FALLBACK_MODEL:
        model_candidates.append(
            {
                "name": FALLBACK_MODEL,
                "timeout": 180.0,
                "num_predict": 600,
            }
        )



    parsed: AIGeneratedDraft | None = None
    successful_raw: dict[str, Any] | None = None
    successful_payload: dict[str, Any] | None = None
    used_model: str | None = None
    used_temperature: float | None = None
    used_semantic_check_completed: bool | None = None
    used_continuity_check_completed: bool | None = None
    used_continuity_model: str | None = None
    used_continuity_checked_facts = 0
    used_continuity_max_contradiction: float | None = None
    last_error: Exception | None = None
    total_attempts = 0

    review_candidate: AIGeneratedDraft | None = None
    review_raw: dict[str, Any] | None = None
    review_payload: dict[str, Any] | None = None
    review_model: str | None = None
    review_temperature: float | None = None
    review_issue_stage: str | None = None
    review_issue_detail: str | None = None
    review_score: float | None = None
    review_semantic_check_completed = False
    review_continuity_check_completed = False
    review_continuity_model: str | None = None
    review_continuity_checked_facts = 0
    review_continuity_max_contradiction: float | None = None

    def record_attempt(
        *,
        model: str,
        temperature: float,
        attempt_started: float,
        outcome: str,
        failure_stage: str | None = None,
        failure_detail: str | None = None,
        story_text: str | None = None,
    ) -> None:
        """Store one model-generation attempt for later evaluation."""

        word_count = (
            story_word_count(story_text)
            if story_text
            else None
        )

        db.insert_generation_attempt(
            request_id=generation_request_id,
            pipeline_version=PIPELINE_VERSION,
            project_id=request.project_id,
            branch_id=request.branch_id,
            parent_node_id=request.parent_node_id,
            selected_choice_id=(
                request.selected_choice_id
            ),
            attempt_number=total_attempts,
            model=model,
            temperature=temperature,
            elapsed_seconds=round(
                time.perf_counter()
                - attempt_started,
                3,
            ),
            outcome=outcome,
            failure_stage=failure_stage,
            failure_detail=failure_detail,
            story_word_count=word_count,
        )

    def remember_review_candidate(
        *,
        draft: AIGeneratedDraft,
        raw: dict[str, Any],
        request_payload: dict[str, Any],
        model: str,
        temperature: float,
        stage: str,
        detail: str,
        semantic_check_completed: bool = False,
        continuity_check_completed: bool = False,
        continuity_model: str | None = None,
        continuity_checked_facts: int = 0,
        continuity_max_contradiction: float | None = None,
    ) -> None:
        """Keep the most useful parseable rejected draft for human review."""
        nonlocal review_candidate
        nonlocal review_raw
        nonlocal review_payload
        nonlocal review_model
        nonlocal review_temperature
        nonlocal review_issue_stage
        nonlocal review_issue_detail
        nonlocal review_score
        nonlocal review_semantic_check_completed
        nonlocal review_continuity_check_completed
        nonlocal review_continuity_model
        nonlocal review_continuity_checked_facts
        nonlocal review_continuity_max_contradiction

        word_count = story_word_count(
            draft.story_text
        )

        if stage == "length":
            distance = (
                STORY_MIN_WORDS - word_count
                if word_count < STORY_MIN_WORDS
                else word_count - STORY_MAX_WORDS
            )
            score = 10 + max(distance, 0)
        else:
            score = {
                "repetition": 20,
                "semantic_repetition": 22,
                "continuity": 25,
                "state": 30,
            }.get(stage, 40)

        if review_score is not None and score >= review_score:
            return

        review_candidate = draft
        review_raw = raw
        review_payload = request_payload
        review_model = model
        review_temperature = temperature
        review_issue_stage = stage
        review_issue_detail = detail
        review_score = score
        review_semantic_check_completed = semantic_check_completed
        review_continuity_check_completed = continuity_check_completed
        review_continuity_model = continuity_model
        review_continuity_checked_facts = continuity_checked_facts
        review_continuity_max_contradiction = continuity_max_contradiction

    reference_nodes = [
        node
        for node in full_path
        if not _is_low_information_scene(
            node["story_text"]
        )
    ]

    for candidate in model_candidates:
        retry_kind: str | None = None
        retry_feedback = ""

        for generation_attempt in range(1, 3):
            if generation_attempt == 1:
                attempt_temperature = effective_temperature

            elif retry_kind == "repetition":
                # A repetitive scene needs more variation.
                attempt_temperature = max(
                    effective_temperature,
                    0.65,
                )
            elif retry_kind == "continuity":
                # Continuity repair should stay controlled so the model
                # follows the authoritative facts supplied in retry feedback.
                attempt_temperature = min(
                    effective_temperature,
                    0.4,
                )
            elif retry_kind == "length":
                attempt_temperature = min(
                    effective_temperature,
                    0.4,
                )

            elif retry_kind == "state":
                attempt_temperature = min(
                    effective_temperature,
                    0.3,
                )
            else:
                # JSON repair should be controlled, but not completely
                # deterministic because the model regenerates the whole
                # scene rather than merely reformatting existing text.
                attempt_temperature = min(
                    effective_temperature,
                    0.3,
                )

            messages: list[dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(context)},
            ]
            if retry_feedback:
                messages.append({"role": "user", "content": retry_feedback})

            payload: dict[str, Any] = {
                "model": candidate["name"],
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "format": OLLAMA_RESPONSE_SCHEMA,
                "messages": messages,
                "options": {
                    "temperature": attempt_temperature,
                    "num_ctx": 4096,
                    "num_predict": candidate["num_predict"],
                    "repeat_last_n": 256,
                    "repeat_penalty": 1.15,
                    "top_k": 40,
                    "top_p": 0.90,
                    "seed": secrets.randbelow(2_147_483_647),
                },
            }

            attempt_started = time.perf_counter()
            total_attempts += 1
            try:
                raw = await call_ollama(
                    payload,
                    timeout_seconds=candidate["timeout"],
                )
            except HTTPException as error:
                last_error = error

                logger.warning(
                    "Model %s failed on API attempt %s: %s",
                    candidate["name"],
                    generation_attempt,
                    error.detail,
                )

                record_attempt(
                    model=candidate["name"],
                    temperature=attempt_temperature,
                    attempt_started=attempt_started,
                    outcome="rejected",
                    failure_stage="api",
                    failure_detail=str(
                        error.detail
                    ),
                )

                if error.status_code in {
                    422,
                    503,
                }:
                    # The request/schema or a required
                    # service failed. Changing writer
                    # models cannot necessarily repair it.
                    raise

                break

            content = str(raw.get("message", {}).get("content", "")).strip()

            try:
                candidate_draft = AIGeneratedDraft.model_validate_json(content)
            except (
                ValidationError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as error:
                last_error = error
                retry_kind = "json"
                retry_feedback = (
                    "REGENERATION REQUIRED: the previous output failed "
                    "SekAI's response contract. "
                    f"Validation error: {str(error)[:1_200]}\n\n"
                    "Correct the structure using this contract:\n"
                    f"{JSON_OUTPUT_CONTRACT}\n\n"
                    "Generate a new scene rather than copying the rejected response."
                )
                logger.warning(
                    "Model %s returned invalid JSON on attempt %s: %s. Raw: %s",
                    candidate["name"],
                    generation_attempt,
                    error,
                    content[:2_000],
                )
                record_attempt(
                    model=candidate["name"],
                    temperature=attempt_temperature,
                    attempt_started=attempt_started,
                    outcome="rejected",
                    failure_stage="schema",
                    failure_detail=str(error)[:2_000],
                )
                continue

            # Stage 1: run every deterministic content/state check together.
            # This prevents a short scene from hiding a second blocking state
            # problem that would otherwise only appear when the user presses Save.
            cleaned_delta = clean_redundant_delta(
                parent_memory,
                candidate_draft.applied_delta,
            )
            candidate_draft = candidate_draft.model_copy(
                update={"applied_delta": cleaned_delta}
            )

            deterministic_issues = validate_reviewed_draft(
                story_text=candidate_draft.story_text,
                parent_memory=parent_memory,
                applied_delta=candidate_draft.applied_delta,
                previous_nodes=[],
                include_duplicate=False,
            )

            if deterministic_issues:
                blocking = blocking_issues(deterministic_issues)
                failure_stage = "state" if blocking else "length"
                issue_summary = format_issues_for_retry(deterministic_issues)

                last_error = ValueError(issue_summary)
                retry_kind = failure_stage
                retry_feedback = (
                    "REGENERATION REQUIRED: the previous draft failed one or "
                    "more deterministic validation checks. Correct every issue "
                    "below in the next complete draft:\n"
                    f"{issue_summary}\n"
                    "Keep all applied_delta changes visibly grounded in story_text, "
                    "do not re-add existing state, do not remove absent state, and "
                    "keep story_text between 50 and 150 words."
                )

                logger.warning(
                    "Model %s failed deterministic validation on attempt %s: %s",
                    candidate["name"],
                    generation_attempt,
                    issue_summary,
                )
                remember_review_candidate(
                    draft=candidate_draft,
                    raw=raw,
                    request_payload=payload,
                    model=candidate["name"],
                    temperature=attempt_temperature,
                    stage=failure_stage,
                    detail=issue_summary,
                )
                record_attempt(
                    model=candidate["name"],
                    temperature=attempt_temperature,
                    attempt_started=attempt_started,
                    outcome="rejected",
                    failure_stage=failure_stage,
                    failure_detail=issue_summary,
                    story_text=candidate_draft.story_text,
                )
                continue

            # Stage 2: inexpensive lexical repetition detection.
            duplicate_reason = repetition_reason(
                candidate_draft.story_text,
                reference_nodes,
            )

            if duplicate_reason:
                last_error = ValueError(
                    duplicate_reason
                )

                retry_kind = "repetition"

                retry_feedback = (
                    "REGENERATION REQUIRED: the previous draft "
                    "repeated an earlier scene. "
                    f"{duplicate_reason} "
                    "Write a genuinely new scene beginning after "
                    "the selected node. Directly perform or answer "
                    "the supplied interaction. Use a different "
                    "title, opening, action, and outcome."
                )

                logger.warning(
                    "Model %s produced a lexically repetitive "
                    "scene on attempt %s: %s",
                    candidate["name"],
                    generation_attempt,
                    duplicate_reason,
                )

                remember_review_candidate(
                    draft=candidate_draft,
                    raw=raw,
                    request_payload=payload,
                    model=candidate["name"],
                    temperature=attempt_temperature,
                    stage="repetition",
                    detail=duplicate_reason,
                )
                record_attempt(
                    model=candidate["name"],
                    temperature=attempt_temperature,
                    attempt_started=attempt_started,
                    outcome="rejected",
                    failure_stage="repetition",
                    failure_detail=duplicate_reason,
                    story_text=candidate_draft.story_text,
                )

                continue

            # Stage 3: semantic repetition detection.
            #
            # This detects paraphrased repetition where the words
            # differ but substantially the same event is described.
            semantic_reason: str | None = None
            semantic_check_completed = False

            try:
                semantic_reason = (
                    await semantic_repetition_reason(
                        candidate_draft.story_text,
                        reference_nodes,
                    )
                )
                semantic_check_completed = True

            except (
                httpx.HTTPError,
                ValueError,
            ) as error:

                if EVALUATION_MODE:
                    record_attempt(
                        model=candidate["name"],
                        temperature=attempt_temperature,
                        attempt_started=attempt_started,
                        outcome="rejected",
                        failure_stage="semantic_service",
                        failure_detail=str(error),
                        story_text=candidate_draft.story_text,
                    )

                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "Semantic repetition checking "
                            "is required in evaluation mode "
                            "but EmbeddingGemma failed: "
                            f"{error}"
                        ),
                    ) from error

                logger.warning(
                    "Semantic repetition check "
                    "could not run for model %s "
                    "on attempt %s: %s",
                    candidate["name"],
                    generation_attempt,
                    error,
                )

            if semantic_reason:
                last_error = ValueError(
                    semantic_reason
                )

                retry_kind = "repetition"

                retry_feedback = (
                    "REGENERATION REQUIRED: the draft "
                    "describes substantially the same event "
                    "as an earlier scene. "
                    f"{semantic_reason} "
                    "Begin after the final event of the parent "
                    "scene and introduce a different action, "
                    "answer, discovery, location change, "
                    "relationship change, or consequence."
                )

                logger.warning(
                    "Model %s produced a semantically repetitive "
                    "scene on attempt %s: %s",
                    candidate["name"],
                    generation_attempt,
                    semantic_reason,
                )
                remember_review_candidate(
                    draft=candidate_draft,
                    raw=raw,
                    request_payload=payload,
                    model=candidate["name"],
                    temperature=attempt_temperature,
                    stage="semantic_repetition",
                    detail=semantic_reason,
                    semantic_check_completed=True,
                )
                record_attempt(
                    model=candidate["name"],
                    temperature=attempt_temperature,
                    attempt_started=attempt_started,
                    outcome="rejected",
                    failure_stage="semantic_repetition",
                    failure_detail=semantic_reason,
                    story_text=candidate_draft.story_text,
                )
                continue

            # Stage 5: specialised NLI continuity review.
            
            # Check the story against the state that WOULD exist after
            # applying the candidate delta. This avoids falsely rejecting
            # legitimate changes such as moving from the village to a dungeon.
            continuity_check_completed = False
            continuity_model: str | None = None
            continuity_checked_facts = 0
            continuity_max_contradiction: float | None = None

            proposed_memory = apply_state_delta(
                parent_memory,
                candidate_draft.applied_delta,
            )

            try:
                continuity_review = await asyncio.to_thread(
                    review_story_continuity,
                    candidate_draft.story_text,
                    proposed_memory,
                    candidate_draft.applied_delta,
                )
                continuity_check_completed = True
                continuity_model = continuity_review.model
                continuity_checked_facts = continuity_review.checked_facts
                continuity_max_contradiction = (
                    continuity_review.max_contradiction_score
                )
                

            except ContinuityModelUnavailableError as error:
                if REQUIRE_NLI or EVALUATION_MODE:
                    # Record the writer attempt before strict
                    # evaluation aborts because DeBERTa is unavailable.
                    record_attempt(
                        model=candidate["name"],
                        temperature=attempt_temperature,
                        attempt_started=attempt_started,
                        outcome="rejected",
                        failure_stage="continuity_service",
                        failure_detail=str(error),
                        story_text=candidate_draft.story_text,
                    )

                    raise HTTPException(
                        status_code=503,
                        detail=str(error),
                    ) from error

                logger.warning(
                    "NLI continuity review is unavailable: %s",
                    error,
                )

                continuity_review = None

            except (ValueError, RuntimeError) as error:
                if REQUIRE_NLI or EVALUATION_MODE:
                    # Record the writer attempt before strict
                    # evaluation aborts because the NLI stage failed.
                    record_attempt(
                        model=candidate["name"],
                        temperature=attempt_temperature,
                        attempt_started=attempt_started,
                        outcome="rejected",
                        failure_stage="continuity_service",
                        failure_detail=str(error),
                        story_text=candidate_draft.story_text,
                    )

                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "The local NLI continuity "
                            "review failed: "
                            f"{error}"
                        ),
                    ) from error

                logger.warning(
                    "NLI continuity review failed open "
                    "for model %s: %s",
                    candidate["name"],
                    error,
                )

                continuity_review = None

            if (
                continuity_review is not None
                and not continuity_review.passed
            ):
                contradiction_text = "\n".join(
                    (
                        f"- {item.fact} "
                        f"(contradiction={item.contradiction_score:.2f})"
                    )
                    for item in continuity_review.contradictions
                )

                last_error = ValueError(
                    "The continuity reviewer detected contradictions."
                )
                retry_kind = "continuity"
                retry_feedback = (
                    "REGENERATION REQUIRED: a specialised Natural Language "
                    "Inference model detected likely contradictions between "
                    "the draft and the state that should exist after its "
                    "applied_delta. Preserve these authoritative facts unless "
                    "the requested interaction explicitly changes them:\n"
                    f"{contradiction_text}"
                )

                logger.warning(
                    "Model %s failed NLI continuity review on attempt %s: %s",
                    candidate["name"],
                    generation_attempt,
                    contradiction_text,
                )
                remember_review_candidate(
                    draft=candidate_draft,
                    raw=raw,
                    request_payload=payload,
                    model=candidate["name"],
                    temperature=attempt_temperature,
                    stage="continuity",
                    detail=contradiction_text,
                    semantic_check_completed=semantic_check_completed,
                    continuity_check_completed=True,
                    continuity_model=continuity_model,
                    continuity_checked_facts=continuity_checked_facts,
                    continuity_max_contradiction=continuity_max_contradiction,
                )
                record_attempt(
                    model=candidate["name"],
                    temperature=attempt_temperature,
                    attempt_started=attempt_started,
                    outcome="rejected",
                    failure_stage="continuity",
                    failure_detail=contradiction_text,
                    story_text=candidate_draft.story_text,
                )

                continue

            # Stage 5: accept the draft only after all checks pass.
            record_attempt(
                model=candidate["name"],
                temperature=attempt_temperature,
                attempt_started=attempt_started,
                outcome="accepted",
                story_text=candidate_draft.story_text,
            )

            parsed = candidate_draft
            successful_raw = raw
            successful_payload = payload
            used_model = candidate["name"]
            used_temperature = attempt_temperature
            used_semantic_check_completed = semantic_check_completed
            used_continuity_check_completed = continuity_check_completed
            used_continuity_model = continuity_model
            used_continuity_checked_facts = continuity_checked_facts
            used_continuity_max_contradiction = (
                continuity_max_contradiction
            )

            logger.info(
                "Accepted story draft from model %s on model attempt %s "
                "after %s total API attempts.",
                candidate["name"],
                generation_attempt,
                total_attempts,
            )

            break

        if parsed is not None:
            break

    if (
        parsed is None
        or successful_raw is None
        or successful_payload is None
        or used_model is None
        or used_temperature is None
        or used_semantic_check_completed is None
        or used_continuity_check_completed is None
    ):
        # If Ollama produced at least one structurally valid draft, expose the
        # best rejected candidate for human review instead of discarding it.
        # Transport failures or totally invalid JSON still return an HTTP error
        # because there is no safe structured draft for the editor to render.
        if (
            review_candidate is not None
            and review_raw is not None
            and review_payload is not None
            and review_model is not None
            and review_temperature is not None
            and review_issue_stage is not None
            and review_issue_detail is not None
        ):
            elapsed = time.perf_counter() - started
            review_output_tokens = review_raw.get("eval_count")
            review_eval_duration_ns = review_raw.get("eval_duration")
            review_tokens_per_second = None

            if review_output_tokens and review_eval_duration_ns:
                review_duration_seconds = (
                    review_eval_duration_ns / 1_000_000_000
                )
                if review_duration_seconds > 0:
                    review_tokens_per_second = round(
                        review_output_tokens / review_duration_seconds,
                        2,
                    )

            review_log_id = db.insert_generation_log(
                request_id=generation_request_id,
                pipeline_version=PIPELINE_VERSION,
                project_id=request.project_id,
                branch_id=request.branch_id,
                parent_node_id=request.parent_node_id,
                selected_choice_id=request.selected_choice_id,
                requested_model=request.model,
                model=review_model,
                semantic_check_completed=review_semantic_check_completed,
                continuity_check_completed=review_continuity_check_completed,
                continuity_model=review_continuity_model,
                continuity_checked_facts=review_continuity_checked_facts,
                continuity_max_contradiction=(
                    review_continuity_max_contradiction
                ),
                requested_temperature=request.temperature,
                effective_temperature=review_temperature,
                elapsed_seconds=round(elapsed, 3),
                prompt_tokens=review_raw.get("prompt_eval_count"),
                output_tokens=review_output_tokens,
                tokens_per_second=review_tokens_per_second,
                attempts=total_attempts,
                interaction_type=request.interaction_type,
                player_input=request.player_input,
                gm_instruction=request.gm_instruction,
                request_payload=review_payload,
                response_payload=review_raw,
                validation_pass=False,
            )

            review_validation_issues = validate_reviewed_draft(
                story_text=review_candidate.story_text,
                parent_memory=parent_memory,
                applied_delta=review_candidate.applied_delta,
                previous_nodes=full_path,
            )

            # Lexical/semantic repetition and NLI are separate specialist stages
            # and are not reproduced by the deterministic preflight validator.
            if review_issue_stage in {
                "repetition",
                "semantic_repetition",
                "continuity",
            }:
                review_validation_issues.append(
                    GenerationIssue(
                        stage=review_issue_stage,
                        severity="warning",
                        target="story_text",
                        message=review_issue_detail,
                        suggestion=(
                            "Review the scene against the earlier branch context. "
                            "Edit it if the automated specialist warning is valid, "
                            "or explicitly accept it after human review."
                        ),
                        can_override=True,
                    )
                )

            if not review_validation_issues:
                review_validation_issues = [
                    GenerationIssue(
                        stage=review_issue_stage,
                        severity=(
                            "blocking"
                            if review_issue_stage == "state"
                            else "warning"
                        ),
                        target=(
                            "applied_delta"
                            if review_issue_stage == "state"
                            else "story_text"
                        ),
                        message=review_issue_detail,
                        can_override=review_issue_stage != "state",
                    )
                ]

            return GenerateStoryResponse(
                status="review_required",
                validation_pass=False,
                validation_issues=review_validation_issues,
                draft=review_candidate,
                metrics=GenerationMetrics(
                    requested_model=request.model,
                    model=review_model,
                    fallback_used=review_model != request.model,
                    semantic_check_completed=(
                        review_semantic_check_completed
                    ),
                    continuity_check_completed=(
                        review_continuity_check_completed
                    ),
                    continuity_model=review_continuity_model,
                    continuity_checked_facts=(
                        review_continuity_checked_facts
                    ),
                    continuity_max_contradiction=(
                        review_continuity_max_contradiction
                    ),
                    requested_temperature=request.temperature,
                    effective_temperature=review_temperature,
                    elapsed_seconds=round(elapsed, 3),
                    prompt_tokens=review_raw.get("prompt_eval_count"),
                    output_tokens=review_output_tokens,
                    tokens_per_second=review_tokens_per_second,
                    attempts=total_attempts,
                ),
                generation_log_id=review_log_id,
                authoritative_context=context,
            )

        if isinstance(last_error, HTTPException):
            final_detail = str(last_error.detail)
            final_status = last_error.status_code
        else:
            final_detail = str(last_error)
            final_status = 502

        raise HTTPException(
            status_code=final_status,
            detail=(
                "Story generation failed and no structured draft could be "
                f"recovered for review. Last error: {final_detail}"
            ),
        )

    elapsed = time.perf_counter() - started
    output_tokens = successful_raw.get("eval_count")
    eval_duration_ns = successful_raw.get("eval_duration")
    tokens_per_second = None

    if output_tokens and eval_duration_ns:
        duration_seconds = eval_duration_ns / 1_000_000_000
        if duration_seconds > 0:
            tokens_per_second = round(
                output_tokens / duration_seconds,
                2,
            )

    log_id = db.insert_generation_log(
        request_id=generation_request_id,
        pipeline_version=PIPELINE_VERSION,
        project_id=request.project_id,
        branch_id=request.branch_id,
        parent_node_id=request.parent_node_id,
        selected_choice_id=request.selected_choice_id,
        requested_model=request.model,
        model=used_model,
        semantic_check_completed=used_semantic_check_completed,
        continuity_check_completed=used_continuity_check_completed,
        continuity_model=used_continuity_model,
        continuity_checked_facts=used_continuity_checked_facts,
        continuity_max_contradiction=used_continuity_max_contradiction,
        requested_temperature=request.temperature,
        effective_temperature=used_temperature,
        elapsed_seconds=round(elapsed, 3),
        prompt_tokens=successful_raw.get("prompt_eval_count"),
        output_tokens=output_tokens,
        tokens_per_second=tokens_per_second,
        attempts=total_attempts,
        interaction_type=request.interaction_type,
        player_input=request.player_input,
        gm_instruction=request.gm_instruction,
        request_payload=successful_payload,
        response_payload=successful_raw,
    )

    return GenerateStoryResponse(
        status="accepted",
        validation_pass=True,
        validation_issues=[],
        draft=parsed,
        metrics=GenerationMetrics(
            requested_model=request.model,
            model=used_model,
            fallback_used=used_model != request.model,
            semantic_check_completed=used_semantic_check_completed,
            continuity_check_completed=used_continuity_check_completed,
            continuity_model=used_continuity_model,
            continuity_checked_facts=used_continuity_checked_facts,
            continuity_max_contradiction=used_continuity_max_contradiction,
            requested_temperature=request.temperature,
            effective_temperature=used_temperature,
            elapsed_seconds=round(elapsed, 3),
            prompt_tokens=successful_raw.get("prompt_eval_count"),
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
            attempts=total_attempts,
        ),
        generation_log_id=log_id,
        authoritative_context=context,
    )

