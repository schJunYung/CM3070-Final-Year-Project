from __future__ import annotations
from pathlib import Path

import openvino_genai as ov_genai
from PIL import Image


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "image"
    / "sd15-int8-ov"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "generated_images"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main() -> int:
    print(f"[LOAD] {MODEL_DIR}")

    pipe = (
        ov_genai.Text2ImagePipeline(
            str(MODEL_DIR),
            "CPU",
        )
    )

    prompt = (
        "fantasy tabletop RPG scene, "
        "ancient stone tower interior, "
        "two adventurers climbing a "
        "spiral staircase, torchlight, "
        "stone guardian waiting above, "
        "cinematic composition"
    )
    print("[GENERATING]")

    tensor = pipe.generate(
        prompt,
        num_inference_steps=20,
    )
    image = Image.fromarray(tensor.data[0])
    output = (
        OUTPUT_DIR / "sd15_test.png"
    )
    image.save(output)
    print(f"[READY] {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
