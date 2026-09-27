from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import image_service


def load_scenes(path: Path) -> list[dict]:
    """Load a non-empty JSON array of benchmark scenes."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, list) or not payload:
        raise ValueError("Dataset must be a non-empty JSON array.")

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark SekAI's local OpenVINO image model on fixed scenes."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(image_service.MODEL_SPECS),
        default=["sd15-int8"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 41])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("image_model_benchmark.csv"),
    )
    args = parser.parse_args()

    scenes = load_scenes(args.dataset)
    rows: list[dict[str, object]] = []

    for scene in scenes:
        for model_key in args.models:
            for seed in args.seeds:
                result = image_service.generate_scene_image(
                    project_id="benchmark",
                    source_kind="manual",
                    model_key=model_key,
                    title=str(scene["title"]),
                    story_text=str(scene["story_text"]),
                    genre=str(scene.get("genre", "Fantasy")),
                    tone=str(scene.get("tone", "Adventurous")),
                    memory_snapshot=dict(scene.get("memory_snapshot", {})),
                    prompt_override=str(scene.get("prompt_override", "")),
                    seed=seed,
                )

                row = {
                    "scene_id": scene.get("id", scene["title"]),
                    "model_key": result["model_key"],
                    "model_id": result["model_id"],
                    "device": result["device"],
                    "precision": result["precision"],
                    "seed": result["seed"],
                    "steps": result["steps"],
                    "guidance_scale": result["guidance_scale"],
                    "model_load_seconds": result["model_load_seconds"],
                    "inference_seconds": result["inference_seconds"],
                    "total_seconds": result["total_seconds"],
                    "cold_start": result["cold_start"],
                    "prompt": result["prompt"],
                    "image_url": result["image_url"],
                    "scene_relevance_1_to_5": "",
                    "key_element_coverage_0_to_1": "",
                    "visual_quality_1_to_5": "",
                    "notes": "",
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))

    if not rows:
        raise RuntimeError("Benchmark generated no result rows.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
