from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import transcription_service

AUDIO_SUFFIXES = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".mp4"}


def normalise_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold())


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    # Dynamic-programming WER with operation counts: substitutions, deletions, insertions.
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, cols):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            sub = dp[i - 1][j - 1]
            delete = dp[i - 1][j]
            insert = dp[i][j - 1]
            options = [
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ]
            dp[i][j] = min(options, key=lambda value: value[0])
    _, substitutions, deletions, insertions = dp[-1][-1]
    return substitutions, deletions, insertions


def word_error_rate(reference_text: str, hypothesis_text: str) -> float | None:
    reference = normalise_words(reference_text)
    if not reference:
        return None
    hypothesis = normalise_words(hypothesis_text)
    s, d, i = edit_counts(reference, hypothesis)
    return (s + d + i) / len(reference)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark SekAI's configured STT backend on fixed audio/reference pairs."
    )
    parser.add_argument("dataset", type=Path, help="Folder containing audio files and matching .txt references")
    parser.add_argument("--output", type=Path, default=Path("stt_benchmark.csv"))
    args = parser.parse_args()

    audio_files = sorted(
        path for path in args.dataset.iterdir()
        if path.is_file() and path.suffix.casefold() in AUDIO_SUFFIXES
    )
    if not audio_files:
        raise SystemExit("No supported audio files found.")

    rows: list[dict[str, object]] = []
    for audio_path in audio_files:
        reference_path = audio_path.with_suffix(".txt")
        reference = reference_path.read_text(encoding="utf-8").strip() if reference_path.exists() else ""
        result = transcription_service.transcribe_file(audio_path)
        wer = word_error_rate(reference, result["text"]) if reference else None
        row = {
            "file": audio_path.name,
            "engine": result["engine"],
            "model": result["model"],
            "compute_type": result["compute_type"],
            "audio_seconds": result["audio_seconds"],
            "model_load_seconds": result["model_load_seconds"],
            "inference_seconds": result["inference_seconds"],
            "total_seconds": result["total_seconds"],
            "real_time_factor": result["real_time_factor"],
            "cold_start": result["cold_start"],
            "reference": reference,
            "transcript": result["text"],
            "wer": round(wer, 4) if wer is not None else None,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rtf_values = [float(row["real_time_factor"]) for row in rows if row["real_time_factor"] is not None]
    wer_values = [float(row["wer"]) for row in rows if row["wer"] is not None]
    summary = {
        "samples": len(rows),
        "mean_rtf": round(statistics.mean(rtf_values), 4) if rtf_values else None,
        "median_rtf": round(statistics.median(rtf_values), 4) if rtf_values else None,
        "mean_wer": round(statistics.mean(wer_values), 4) if wer_values else None,
        "median_wer": round(statistics.median(wer_values), 4) if wer_values else None,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
