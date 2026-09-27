from __future__ import annotations

import math
import os
import re
from difflib import SequenceMatcher
from typing import Any

import httpx


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)
EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "embeddinggemma:latest",
)
MAX_EMBED_CHARACTERS = 4_000


def normalise_text(value: str) -> str:
    return " ".join(
        re.findall(
            r"[a-z0-9']+",
            value.casefold(),
        )
    )


def lexical_similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None,
        normalise_text(left),
        normalise_text(right),
        autojunk=False,
    ).ratio()


def cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors have different dimensions.")

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    )
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))

    if left_length == 0 or right_length == 0:
        return 0.0

    return dot_product / (left_length * right_length)


def _embedding_excerpt(value: str) -> str:
    clean = " ".join(value.split())
    if len(clean) <= MAX_EMBED_CHARACTERS:
        return clean
    return clean[-MAX_EMBED_CHARACTERS:]


async def embed_texts(values: list[str]) -> list[list[float]]:
    if not values or any(not value.strip() for value in values):
        raise ValueError("Embedding input must contain non-empty text.")

    payload = {
        "model": EMBEDDING_MODEL,
        "input": [_embedding_excerpt(value) for value in values],
        "truncate": True,
        "keep_alive": "10m",
    }
    timeout = httpx.Timeout(60.0, connect=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json=payload,
        )
        response.raise_for_status()
        body = response.json()

    embeddings = body.get("embeddings", [])
    if len(embeddings) != len(values):
        raise ValueError(
            "Ollama returned an unexpected number of embeddings."
        )

    dimensions = {len(vector) for vector in embeddings}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("Ollama returned invalid embedding dimensions.")

    return embeddings


async def retrieve_relevant_nodes(
    query_text: str,
    nodes: list[dict[str, Any]],
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Retrieve earlier branch nodes that are most
    semantically related to the selected choice and
    current generation direction.
    """

    candidates = [
        node
        for node in nodes[-12:]
        if node.get(
            "story_text",
            ""
        ).strip()
    ]

    if (
        not query_text.strip()
        or not candidates
    ):
        return []

    candidate_texts = [
        (
            f"{node.get('title', '')}\n"
            f"{node.get('story_text', '')}"
        )
        for node in candidates
    ]

    vectors = await embed_texts(
        [
            query_text,
            *candidate_texts,
        ]
    )

    query_vector = vectors[0]

    scored_nodes = []

    for node, vector in zip(
        candidates,
        vectors[1:],
    ):
        score = cosine_similarity(
            query_vector,
            vector,
        )

        scored_nodes.append(
            {
                **node,
                "retrieval_score":
                    float(score),
            }
        )

    scored_nodes.sort(
        key=lambda node:
            node["retrieval_score"],
        reverse=True,
    )

    return scored_nodes[:top_k]

async def semantic_repetition_reason(
    candidate_story: str,
    reference_nodes: list[dict[str, Any]],
) -> str | None:
    """Detect a likely paraphrase of one of the three latest branch scenes."""
    recent_nodes = reference_nodes[-3:]
    if not recent_nodes:
        return None

    texts = [
        candidate_story,
        *[node["story_text"] for node in recent_nodes],
    ]
    vectors = await embed_texts(texts)
    candidate_vector = vectors[0]

    for index, node in enumerate(recent_nodes, start=1):
        semantic_score = cosine_similarity(
            candidate_vector,
            vectors[index],
        )
        lexical_score = lexical_similarity(
            candidate_story,
            node["story_text"],
        )
        is_immediate_parent = node["id"] == recent_nodes[-1]["id"]

        # Application thresholds, to be tuned against labelled repeated and
        # valid continuation pairs during final evaluation.
        semantic_limit = 0.88 if is_immediate_parent else 0.92

        if semantic_score >= semantic_limit and lexical_score >= 0.55:
            return (
                "The generated scene appears to repeat "
                f"'{node['title']}' semantically "
                f"(semantic={semantic_score:.2f}, "
                f"lexical={lexical_score:.2f})."
            )

    return None
