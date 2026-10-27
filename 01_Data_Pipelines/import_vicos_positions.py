#!/usr/bin/env python3
"""Build a descriptive BJJ position-transition model from ViCoS annotations.

The source contains sparsely selected labeled frames, not continuous state
annotations. Outputs therefore describe transitions between observations and
same-label observation spans; they are not exact dwell-time estimates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from statistics import median
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "vicos_bjj_positions"
TASK_ID = "bjj_position_transition_analysis"
SOURCE_PAGE = "https://www2.vicos.si/resources/jiujitsu/"
ANNOTATION_URL = "https://data.vicos.si/datasets/JuiJuitsu/annotations.json"
LICENSE_ID = "CC-BY-NC-SA-4.0"
NOMINAL_FPS = 30.0
CHUNK_SIZE = 1024 * 1024


def _iter_json_array(path: Path) -> Iterator[dict[str, object]]:
    decoder = JSONDecoder()
    with path.open("r", encoding="utf-8-sig") as source:
        buffer = ""
        position = 0
        started = False
        while True:
            if position >= len(buffer):
                chunk = source.read(CHUNK_SIZE)
                if not chunk:
                    if not started:
                        raise ValueError("Annotation file is empty")
                    return
                buffer, position = chunk, 0

            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                continue

            if not started:
                if buffer[position] != "[":
                    raise ValueError("Expected a top-level JSON array")
                started = True
                position += 1
                continue

            if buffer[position] == ",":
                position += 1
                continue
            if buffer[position] == "]":
                return

            try:
                item, end = decoder.raw_decode(buffer, position)
            except JSONDecodeError:
                chunk = source.read(CHUNK_SIZE)
                if not chunk:
                    raise ValueError("Annotation file ended inside a JSON object")
                buffer = buffer[position:] + chunk
                position = 0
                continue

            if not isinstance(item, dict):
                raise ValueError("Each annotation entry must be a JSON object")
            yield item
            position = end
            if position >= CHUNK_SIZE:
                buffer, position = buffer[position:], 0


def _load_registry(registry_path: Path) -> dict[str, object]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("policy") != "deny_by_default":
        raise PermissionError("Dataset registry must retain its deny-by-default policy")
    entries = registry.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("Registry must contain a 'datasets' list")
    matches = [entry for entry in entries if entry.get("dataset_id") == DATASET_ID]
    if len(matches) != 1:
        raise PermissionError(f"Registry must contain exactly one {DATASET_ID!r} entry")
    entry = matches[0]
    approval = entry.get("analysis_approval", {})
    if TASK_ID not in approval.get("approved_tasks", []):
        raise PermissionError(f"Registry does not approve the {TASK_ID!r} analysis")
    if entry.get("license_id") != LICENSE_ID:
        raise PermissionError("Registry license does not match the verified ViCoS license")
    return entry


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterator[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_annotations(annotations_path: Path, registry_path: Path, output_dir: Path) -> dict[str, object]:
    registry_entry = _load_registry(registry_path)
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations not found: {annotations_path}")

    frames: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    annotation_count = 0
    duplicate_count = 0
    conflict_count = 0

    for item in _iter_json_array(annotations_path):
        annotation_count += 1
        image_id = str(item.get("image", "")).strip()
        position = str(item.get("position", "")).strip().lower()
        if len(image_id) < 7 or not image_id.isdigit() or not position:
            raise ValueError(f"Invalid image or position label at annotation {annotation_count}")
        frame = int(item.get("frame", -1))
        sequence_id = image_id[:2]
        if frame < 0 or int(image_id[-5:]) != frame:
            raise ValueError(f"Image/frame mismatch for {image_id!r} at annotation {annotation_count}")

        sequence_frames = frames[sequence_id]
        existing = sequence_frames.get(frame)
        if existing is None:
            sequence_frames[frame] = {"image_id": image_id, "position": position, "ambiguous": False}
        elif existing["position"] == position and not existing["ambiguous"]:
            duplicate_count += 1
        else:
            if not existing["ambiguous"]:
                conflict_count += 1
            existing["ambiguous"] = True
            existing["position"] = ""

    output_dir.mkdir(parents=True, exist_ok=True)
    observations: list[dict[str, object]] = []
    intervals: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    transition_counts: Counter[tuple[str, str]] = Counter()
    state_observation_counts: Counter[str] = Counter()
    states: set[str] = set()

    def finish_run(sequence_id: str, run: list[tuple[int, str]]) -> None:
        if not run:
            return
        first_frame, position = run[0]
        last_frame = run[-1][0]
        runs.append({
            "sequence_id": sequence_id,
            "position": position,
            "first_observed_frame": first_frame,
            "last_observed_frame": last_frame,
            "observation_count": len(run),
            "same_label_observation_span_seconds": (last_frame - first_frame) / NOMINAL_FPS,
            "span_is_positive": len(run) > 1,
            "duration_is_exact_dwell": False,
        })

    sequence_ids = sorted(frames)
    for sequence_id in sequence_ids:
        ordered = sorted(frames[sequence_id].items())
        previous: tuple[int, str, str] | None = None
        current_run: list[tuple[int, str]] = []
        for frame, record in ordered:
            position = str(record["position"])
            ambiguous = bool(record["ambiguous"])
            observations.append({
                "dataset_id": DATASET_ID,
                "sequence_id": sequence_id,
                "frame": frame,
                "timestamp_ms_approx": round(frame * 1000 / NOMINAL_FPS, 3),
                "image_id": record["image_id"],
                "position": position,
                "label_status": "ambiguous_duplicate_frame" if ambiguous else "labeled",
                "source_id": DATASET_ID,
                "license_id": LICENSE_ID,
            })
            if ambiguous:
                finish_run(sequence_id, current_run)
                current_run = []
                previous = None
                continue

            states.add(position)
            state_observation_counts[position] += 1
            image_id = str(record["image_id"])
            if previous is None:
                current_run = [(frame, position)]
            else:
                previous_frame, previous_position, previous_image_id = previous
                gap_frames = frame - previous_frame
                intervals.append({
                    "sequence_id": sequence_id,
                    "frame_from": previous_frame,
                    "frame_to": frame,
                    "gap_frames": gap_frames,
                    "gap_seconds_approx": round(gap_frames / NOMINAL_FPS, 6),
                    "state_from": previous_position,
                    "state_to": position,
                    "label_changed": previous_position != position,
                    "image_from": previous_image_id,
                    "image_to": image_id,
                    "transition_time_is_exact": False,
                })
                transition_counts[(previous_position, position)] += 1
                if position == previous_position:
                    current_run.append((frame, position))
                else:
                    finish_run(sequence_id, current_run)
                    current_run = [(frame, position)]
            previous = (frame, position, image_id)
        finish_run(sequence_id, current_run)

    states_sorted = sorted(states)
    counts_by_from = Counter()
    for (state_from, _), count in transition_counts.items():
        counts_by_from[state_from] += count

    count_rows = (
        {"state_from": state_from, "state_to": state_to, "count": transition_counts[(state_from, state_to)]}
        for state_from in states_sorted
        for state_to in states_sorted
        if transition_counts[(state_from, state_to)]
    )
    matrix_rows = (
        {
            "state_from": state_from,
            **{
                state_to: (
                    transition_counts[(state_from, state_to)] / counts_by_from[state_from]
                    if counts_by_from[state_from] else 0.0
                )
                for state_to in states_sorted
            },
        }
        for state_from in states_sorted
    )

    spans_by_state: dict[str, list[float]] = defaultdict(list)
    singleton_by_state: Counter[str] = Counter()
    for run in runs:
        state = str(run["position"])
        if run["span_is_positive"]:
            spans_by_state[state].append(float(run["same_label_observation_span_seconds"]))
        else:
            singleton_by_state[state] += 1
    dwell_rows = (
        {
            "position": state,
            "runs_with_multiple_observations": len(spans_by_state[state]),
            "single_observation_runs_excluded": singleton_by_state[state],
            "median_same_label_span_seconds": median(spans_by_state[state]) if spans_by_state[state] else None,
            "p90_same_label_span_seconds": _percentile(spans_by_state[state], 0.9),
            "interpretation": "same-label span between sampled annotations; not exact or uninterrupted dwell time",
        }
        for state in states_sorted
    )

    observation_fields = [
        "dataset_id", "sequence_id", "frame", "timestamp_ms_approx", "image_id",
        "position", "label_status", "source_id", "license_id",
    ]
    _write_csv(output_dir / "position_observations.csv", observation_fields, iter(observations))
    _write_csv(
        output_dir / "position_transition_intervals.csv",
        [
            "sequence_id", "frame_from", "frame_to", "gap_frames", "gap_seconds_approx",
            "state_from", "state_to", "label_changed", "image_from", "image_to",
            "transition_time_is_exact",
        ],
        iter(intervals),
    )
    _write_csv(
        output_dir / "transition_counts.csv",
        ["state_from", "state_to", "count"],
        count_rows,
    )
    _write_csv(
        output_dir / "transition_matrix.csv",
        ["state_from", *states_sorted],
        matrix_rows,
    )
    _write_csv(
        output_dir / "same_label_span_runs.csv",
        [
            "sequence_id", "position", "first_observed_frame", "last_observed_frame",
            "observation_count", "same_label_observation_span_seconds",
            "span_is_positive", "duration_is_exact_dwell",
        ],
        iter(runs),
    )
    _write_csv(
        output_dir / "same_label_span_summary.csv",
        [
            "position", "runs_with_multiple_observations", "single_observation_runs_excluded",
            "median_same_label_span_seconds", "p90_same_label_span_seconds", "interpretation",
        ],
        dwell_rows,
    )

    summary = {
        "dataset_id": DATASET_ID,
        "analysis": TASK_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_page": SOURCE_PAGE,
        "annotation_download": ANNOTATION_URL,
        "license_id": LICENSE_ID,
        "license_note": "Non-commercial use; attribution and share-alike terms apply.",
        "paper_citation": "Hudovernik, V. & Skočaj, D. (2022). Video-Based Detection of Combat Positions and Automatic Scoring in Jiu-jitsu. ACM Multimedia Workshop MMSports’22.",
        "annotation_sha256": _sha256(annotations_path),
        "annotation_file": str(annotations_path),
        "annotation_records": annotation_count,
        "unique_frames": sum(len(sequence_frames) for sequence_frames in frames.values()),
        "identical_duplicate_annotations_ignored": duplicate_count,
        "ambiguous_duplicate_frames_excluded_from_transitions": conflict_count,
        "source_reported_sparring_sequences": 6,
        "encoded_sequence_prefix_count": len(sequence_ids),
        "position_observation_counts": dict(sorted(state_observation_counts.items())),
        "position_class_count": len(states_sorted),
        "position_classes": states_sorted,
        "observation_interval_count": len(intervals),
        "observed_transition_pair_count": sum(transition_counts.values()),
        "observed_label_changes": sum(count for (source, target), count in transition_counts.items() if source != target),
        "same_label_observation_pairs": sum(count for (source, target), count in transition_counts.items() if source == target),
        "same_label_runs": len(runs),
        "nominal_fps_assumption": NOMINAL_FPS,
        "images_downloaded": False,
        "pose_landmarks_used": False,
        "model_trained": False,
        "limitations": [
            "This is a descriptive transition model over consecutive sampled annotations, not a supervised image model.",
            "The source contains selected labeled frames from six sparring sequences; annotations are not continuous.",
            f"The annotation image prefixes encode {len(sequence_ids)} distinct sequence IDs; do not assume these equal independent matches.",
            "Frame-rate timing is approximate (the source describes videos as roughly 30 fps).",
            "Observed label changes may hide intermediate positions; transition times and true dwell durations are unknown.",
            "Same-label observation spans are not guaranteed to be uninterrupted dwell episodes.",
            "Results are specific to this limited dataset and are not general performance estimates.",
            "Use and redistribution must comply with CC BY-NC-SA 4.0, including attribution and share-alike terms.",
        ],
        "generated_files": [
            "position_observations.csv",
            "position_transition_intervals.csv",
            "transition_counts.csv",
            "transition_matrix.csv",
            "same_label_span_runs.csv",
            "same_label_span_summary.csv",
            "analysis_summary.json",
        ],
        "registry_status": registry_entry.get("status"),
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create non-commercial descriptive BJJ position-transition outputs from ViCoS annotations."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "data" / "raw" / DATASET_ID / "annotations.json",
    )
    parser.add_argument("--registry", type=Path, default=ROOT / "data" / "dataset_registry.json")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "vicos_position_analysis",
    )
    parser.add_argument(
        "--confirm-noncommercial-use",
        action="store_true",
        help="Confirm use is non-commercial and attribution/share-alike terms will be followed.",
    )
    args = parser.parse_args()
    if not args.confirm_noncommercial_use:
        parser.error("ViCoS CC BY-NC-SA 4.0 requires --confirm-noncommercial-use")
    summary = process_annotations(args.annotations, args.registry, args.out_dir)
    print(json.dumps({
        "dataset_id": summary["dataset_id"],
        "annotations": summary["annotation_records"],
        "sequence_prefixes": summary["encoded_sequence_prefix_count"],
        "position_classes": summary["position_class_count"],
        "observed_pairs": summary["observed_transition_pair_count"],
        "output_dir": str(args.out_dir),
        "warning": "Descriptive sparse-label analysis only; no image model trained.",
    }, indent=2))


if __name__ == "__main__":
    main()
