from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import IMAGE_MODEL_ROOT
from backend.image_service import MODEL_SPECS


def export_model(model_key: str) -> None:
    from optimum.intel import (
        OVStableDiffusionPipeline,
        OVWeightQuantizationConfig,
    )

    spec = MODEL_SPECS[model_key]
    target = IMAGE_MODEL_ROOT / spec.local_dir_name
    target.mkdir(parents=True, exist_ok=True)

    print(f"[MODEL]  {model_key}")
    print(f"[SOURCE] {spec.model_id}")
    print(f"[TARGET] {target}")

    kwargs = {
        "export": True,
        "compile": False,
    }

    if model_key == "sd15-int8":
        # This is INT8 WEIGHT COMPRESSION, not full activation INT8.
        # It is deliberately the easy reproducible baseline.
        kwargs["quantization_config"] = OVWeightQuantizationConfig(
            bits=8,
        )

    pipeline = OVStableDiffusionPipeline.from_pretrained(
        spec.model_id,
        **kwargs,
    )
    pipeline.save_pretrained(target)
    print(f"[READY] Saved OpenVINO model: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export SekAI image candidates to local OpenVINO IR."
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_SPECS, "all"],
        default="all",
    )
    args = parser.parse_args()

    selected = (
        list(MODEL_SPECS)
        if args.model == "all"
        else [args.model]
    )

    for model_key in selected:
        export_model(model_key)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
