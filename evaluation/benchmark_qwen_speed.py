from __future__ import annotations

import csv
import json
import random
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


OLLAMA_URL = "http://localhost:11434"

# IMPORTANT:
# Replace these with the EXACT names printed by:
#
#   ollama list
#
MODELS = [
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_0",
    "qwen2.5:1.5b-instruct-q3_K_M",
    "qwen2.5:1.5b-instruct-q2_K",
]

WORD_TARGETS = [50, 70, 100, 150]

# Start with 3. Increase to 5 later if time permits.
REPETITIONS = 3

TEMPERATURE = 0.0
SEED = 42
NUM_PREDICT = 400
NUM_CTX = 4096

RESULT_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULT_DIR / "qwen_quantization_speed.csv"
OUTPUT_PATH = RESULT_DIR / "qwen_quantization_outputs.jsonl"


SYSTEM_PROMPT = """
You are a tabletop role-playing game narrative writer.

Continue the supplied scene from exactly where it ends.

Rules:
- Write one coherent narrative paragraph.
- Follow the player's stated action.
- Advance the story rather than summarising previous events.
- Preserve all facts supplied in the current story state.
- Do not use headings, bullet points, JSON or commentary.
- Do not repeat the starting scene.
- Return only the narrative continuation.
""".strip()


STORY_CONTEXT = """
GENRE:
Fantasy

TONE:
Mysterious adventure

MAIN OBJECTIVE:
Reach the top of an abandoned tower and discover why its beacon
has begun glowing again.

CURRENT STORY STATE:
Location: Entrance hall of the ruined tower
Inventory: Torch, iron key
Threat level: Medium

STARTING SCENE:
The traveller steps through the cracked stone doorway of the ruined
tower. Rain beats against the entrance behind them while a narrow
staircase coils upward into darkness. An old guard sits beside an
extinguished brazier. He warns that something has been moving on the
upper floors since the tower's beacon began glowing again. The
traveller carries a torch and an iron key recovered outside. Nothing
has yet attacked, but scraping noises occasionally echo down the
staircase.

PLAYER ACTION:
The traveller lights the torch, asks the guard what he saw upstairs,
and then begins climbing the staircase.
""".strip()


def word_count(text: str) -> int:
    """
    Match SekAI's application word-count rule.
    """
    return len(re.findall(r"\b[\w'-]+\b", text))


def ns_to_seconds(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000


def build_user_prompt(target_words: int) -> str:
    return f"""
{STORY_CONTEXT}

LENGTH REQUIREMENT:
Write approximately {target_words} words.

The requested length is part of the test. Do not explain the word
count and do not output anything except the story continuation.
""".strip()


def installed_models(client: httpx.Client) -> set[str]:
    response = client.get(f"{OLLAMA_URL}/api/tags")
    response.raise_for_status()

    return {
        model["name"]
        for model in response.json().get("models", [])
        if model.get("name")
    }


def generate(
    client: httpx.Client,
    model: str,
    target_words: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(target_words),
            },
        ],
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
    }

    started = time.perf_counter()

    response = client.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
    )

    wall_seconds = time.perf_counter() - started

    response.raise_for_status()
    body = response.json()

    text = (
        body.get("message", {})
        .get("content", "")
        .strip()
    )

    eval_seconds = ns_to_seconds(body.get("eval_duration"))
    output_tokens = body.get("eval_count")

    tokens_per_second = None

    if (
        eval_seconds
        and output_tokens is not None
        and eval_seconds > 0
    ):
        tokens_per_second = output_tokens / eval_seconds

    actual_words = word_count(text)
    word_error = actual_words - target_words

    tolerance = max(
        5,
        round(target_words * 0.10),
    )

    return {
        "text": text,
        "wall_seconds": wall_seconds,
        "total_seconds": ns_to_seconds(
            body.get("total_duration")
        ),
        "load_seconds": ns_to_seconds(
            body.get("load_duration")
        ),
        "prompt_eval_seconds": ns_to_seconds(
            body.get("prompt_eval_duration")
        ),
        "generation_seconds": eval_seconds,
        "prompt_tokens": body.get("prompt_eval_count"),
        "output_tokens": output_tokens,
        "tokens_per_second": tokens_per_second,
        "actual_words": actual_words,
        "word_error": word_error,
        "absolute_word_error": abs(word_error),
        "within_10_percent": (
            abs(word_error) <= tolerance
        ),
        "within_sekai_bounds": (
            50 <= actual_words <= 150
        ),
    }


def make_conditions() -> list[tuple[int, int]]:
    conditions = [
        (target, repetition)
        for target in WORD_TARGETS
        for repetition in range(1, REPETITIONS + 1)
    ]

    # Use a fixed shuffle so every run can be reproduced.
    random.Random(20260921).shuffle(conditions)

    return conditions


def save_rows(rows: list[dict]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "phase",
        "model",
        "target_words",
        "repetition",
        "temperature",
        "seed",
        "num_predict",
        "num_ctx",
        "wall_seconds",
        "total_seconds",
        "load_seconds",
        "prompt_eval_seconds",
        "generation_seconds",
        "prompt_tokens",
        "output_tokens",
        "tokens_per_second",
        "actual_words",
        "word_error",
        "absolute_word_error",
        "within_10_percent",
        "within_sekai_bounds",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                field: row.get(field)
                for field in fieldnames
            })


def save_output(record: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def print_summary(rows: list[dict]) -> None:
    measured = [
        row
        for row in rows
        if row["phase"] == "measured"
    ]

    print("\n")
    print("=" * 90)
    print("QWEN QUANTIZATION SPEED SUMMARY")
    print("=" * 90)

    for model in MODELS:
        model_rows = [
            row
            for row in measured
            if row["model"] == model
        ]

        print(f"\nMODEL: {model}")

        for target in WORD_TARGETS:
            target_rows = [
                row
                for row in model_rows
                if row["target_words"] == target
            ]

            if not target_rows:
                continue

            wall_times = [
                row["wall_seconds"]
                for row in target_rows
            ]

            throughputs = [
                row["tokens_per_second"]
                for row in target_rows
                if row["tokens_per_second"] is not None
            ]

            actual_words = [
                row["actual_words"]
                for row in target_rows
            ]

            mean_wall = statistics.mean(wall_times)
            median_wall = statistics.median(wall_times)
            mean_words = statistics.mean(actual_words)

            mean_tps = (
                statistics.mean(throughputs)
                if throughputs
                else 0.0
            )

            length_successes = sum(
                row["within_10_percent"]
                for row in target_rows
            )

            print(
                f"  Target {target:>3} words | "
                f"mean={mean_wall:>7.2f}s | "
                f"median={median_wall:>7.2f}s | "
                f"tok/s={mean_tps:>6.2f} | "
                f"actual words={mean_words:>6.1f} | "
                f"length success={length_successes}/{len(target_rows)}"
            )


def main() -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Start a new raw-output file for this benchmark.
    OUTPUT_PATH.write_text(
        "",
        encoding="utf-8",
    )

    timeout = httpx.Timeout(
        300.0,
        connect=5.0,
    )

    rows: list[dict] = []

    with httpx.Client(timeout=timeout) as client:
        available = installed_models(client)

        missing = [
            model
            for model in MODELS
            if model not in available
        ]

        if missing:
            print("The following benchmark models are not installed:")

            for model in missing:
                print(f"  - {model}")

            print(
                "\nRun 'ollama list' and update MODELS "
                "with the exact installed model tags."
            )

            raise SystemExit(1)

        conditions = make_conditions()

        for model in MODELS:
            print("\n" + "=" * 90)
            print(f"MODEL: {model}")
            print("=" * 90)

            # ---------------------------------------------
            # Warm-up run.
            # This loads the model but is not included
            # in the primary timing comparison.
            # ---------------------------------------------
            print("Warm-up generation...")

            warmup = generate(
                client,
                model,
                50,
            )

            warmup_row = {
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "phase": "warmup",
                "model": model,
                "target_words": 50,
                "repetition": 0,
                "temperature": TEMPERATURE,
                "seed": SEED,
                "num_predict": NUM_PREDICT,
                "num_ctx": NUM_CTX,
                **warmup,
            }

            rows.append(warmup_row)
            save_output(warmup_row)

            print(
                f"Warm-up: "
                f"{warmup['wall_seconds']:.2f}s, "
                f"{warmup['actual_words']} words"
            )

            # ---------------------------------------------
            # Controlled measured runs.
            # ---------------------------------------------
            for target_words, repetition in conditions:
                print(
                    f"Testing {target_words} words "
                    f"(repeat {repetition}/{REPETITIONS})..."
                )

                result = generate(
                    client,
                    model,
                    target_words,
                )

                row = {
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "phase": "measured",
                    "model": model,
                    "target_words": target_words,
                    "repetition": repetition,
                    "temperature": TEMPERATURE,
                    "seed": SEED,
                    "num_predict": NUM_PREDICT,
                    "num_ctx": NUM_CTX,
                    **result,
                }

                rows.append(row)
                save_output(row)

                print(
                    f"  {result['wall_seconds']:.2f}s | "
                    f"{result['tokens_per_second'] or 0:.2f} tok/s | "
                    f"{result['actual_words']} words"
                )

                # Save continuously in case the full benchmark
                # is interrupted.
                save_rows(rows)

    save_rows(rows)
    print_summary(rows)

    print("\nResults written to:")
    print(f"  {CSV_PATH}")
    print(f"  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

