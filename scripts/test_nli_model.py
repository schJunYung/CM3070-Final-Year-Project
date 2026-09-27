from __future__ import annotations

from pathlib import Path

from sentence_transformers import CrossEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "nli-deberta-v3-small"
LABELS = ["contradiction", "entailment", "neutral"]


def main() -> int:
    if not MODEL_DIR.exists():
        print("[ERROR] Local model folder does not exist.")
        print("Run scripts/download_nli_model.py first.")
        return 1

    model = CrossEncoder(
        str(MODEL_DIR),
        device="cpu",
        local_files_only=True,
    )

    pairs = [
        (
            "The player enters the dungeon and closes the gate behind them.",
            "The current location is the dungeon.",
        ),
        (
            "The player enters the dungeon and closes the gate behind them.",
            "The current location is the village.",
        ),
        (
            "The player studies a glowing altar in silence.",
            "The current inventory contains a silver key.",
        ),
    ]

    scores = model.predict(
        pairs,
        apply_softmax=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        device="cpu",
    )

    for pair, row in zip(pairs, scores):
        label_index = int(row.argmax())
        print("\nPremise:   ", pair[0])
        print("Hypothesis:", pair[1])
        print("Label:     ", LABELS[label_index])
        print(
            "Scores:    ",
            {
                label: round(float(row[index]), 4)
                for index, label in enumerate(LABELS)
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
