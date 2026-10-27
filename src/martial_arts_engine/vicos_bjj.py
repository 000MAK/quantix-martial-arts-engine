from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd


DISCIPLINE_ID = "ma-027-brazilian-jiu-jitsu-no-gi"
DATASET_URL = "https://www2.vicos.si/resources/jiujitsu/"
LICENSE_ID = "CC-BY-NC-SA-4.0"
POSITION_ALIASES = {
    "standing": "standing",
    "takedown": "takedown",
    "open_guard": "open_guard",
    "half_guard": "half_guard",
    "closed_guard": "closed_guard",
    "50_50": "50_50_guard",
    "50_50_guard": "50_50_guard",
    "5050": "50_50_guard",
    "5050_guard": "50_50_guard",
    "fifty_fifty_guard": "50_50_guard",
    "side_control": "side_control",
    "mount": "mount",
    "back": "back",
    "back_control": "back",
    "turtle": "turtle",
}


def _position_state(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every ViCoS annotation needs a non-empty Position label")
    label = value.strip().lower().replace("-", "_").replace(" ", "_")
    label = re.sub(r"(?:_)?[12]$", "", label)
    if label not in POSITION_ALIASES:
        raise ValueError(f"Unrecognized ViCoS Position label: {value!r}")
    return POSITION_ALIASES[label]


def read_vicos_annotations(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("annotations")
    if not isinstance(payload, list) or not payload:
        raise ValueError("ViCoS annotations must be a non-empty JSON array")

    rows = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Annotation {index} must be a JSON object")
        fields = {str(key).casefold(): value for key, value in item.items()}
        image = str(fields.get("image", "")).strip()
        match = re.match(r"(\d{2})", Path(image).stem)
        if not match:
            raise ValueError(f"Annotation {index} has an invalid Image sequence prefix: {image!r}")
        try:
            frame_number = float(fields["frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Annotation {index} needs a non-negative integer Frame") from exc
        if not math.isfinite(frame_number) or not frame_number.is_integer():
            raise ValueError(f"Annotation {index} needs a non-negative integer Frame")
        frame = int(frame_number)
        if frame < 0:
            raise ValueError(f"Annotation {index} has a negative Frame")
        position = fields.get("position")
        rows.append({
            "sequence_id": f"vicos-{match.group(1)}",
            "frame_index": frame,
            "timestamp_ms": frame * 1000.0 / 30.0,
            "source_image": image,
            "source_position": str(position or ""),
            "state": _position_state(position),
            "discipline_id": DISCIPLINE_ID,
        })

    observations = pd.DataFrame(rows).sort_values(
        ["sequence_id", "frame_index"], kind="stable"
    ).reset_index(drop=True)
    duplicated = observations.duplicated(["sequence_id", "frame_index"], keep=False)
    if duplicated.any():
        row = observations.loc[duplicated].iloc[0]
        raise ValueError(
            "ViCoS has duplicate annotations for "
            f"{row.sequence_id} frame {row.frame_index}"
        )
    return observations


def analyze_vicos_positions(
    observations: pd.DataFrame, fps: float = 30.0
) -> dict[str, pd.DataFrame]:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be greater than zero")
    required = {"sequence_id", "frame_index", "state"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError("Missing position columns: " + ", ".join(sorted(missing)))
    if observations.empty:
        raise ValueError("No ViCoS position observations supplied")

    ordered = observations.sort_values(
        ["sequence_id", "frame_index"], kind="stable"
    ).copy()
    if ordered.duplicated(["sequence_id", "frame_index"]).any():
        raise ValueError("Each sequence/frame pair must have exactly one position label")

    episodes: list[dict] = []
    transitions: list[dict] = []
    for sequence_id, group in ordered.groupby("sequence_id", sort=True):
        group = group.reset_index(drop=True)
        run_starts = [0]
        for i in range(1, len(group)):
            if group.at[i, "state"] != group.at[i - 1, "state"]:
                run_starts.append(i)
        for run_number, start_index in enumerate(run_starts):
            end_index = (
                run_starts[run_number + 1] - 1
                if run_number + 1 < len(run_starts)
                else len(group) - 1
            )
            start = group.iloc[start_index]
            end = group.iloc[end_index]
            has_next = run_number + 1 < len(run_starts)
            next_start = group.iloc[run_starts[run_number + 1]] if has_next else None
            span = (int(end.frame_index) - int(start.frame_index)) / fps
            episode = {
                "sequence_id": str(sequence_id),
                "episode_index": run_number,
                "state": str(start.state),
                "start_frame": int(start.frame_index),
                "end_frame": int(end.frame_index),
                "observed_frames": int(end_index - start_index + 1),
                "observed_label_span_seconds": float(span),
                "next_observed_state": str(next_start.state) if has_next else None,
                "next_observed_frame": int(next_start.frame_index) if has_next else None,
                "first_observed_episode": run_number == 0,
                "terminal_right_censored": not has_next,
            }
            episodes.append(episode)
            if has_next:
                transitions.append({
                    "sequence_id": str(sequence_id),
                    "from_state": str(start.state),
                    "to_state": str(next_start.state),
                    "last_from_frame": int(end.frame_index),
                    "first_to_frame": int(next_start.frame_index),
                    "observation_gap_seconds": (
                        int(next_start.frame_index) - int(end.frame_index)
                    ) / fps,
                })

    episode_frame = pd.DataFrame(episodes)
    transition_frame = pd.DataFrame(transitions, columns=[
        "sequence_id", "from_state", "to_state", "last_from_frame",
        "first_to_frame", "observation_gap_seconds",
    ])
    states = sorted(ordered.state.astype(str).unique())
    counts = pd.crosstab(
        transition_frame["from_state"], transition_frame["to_state"]
    ) if not transition_frame.empty else pd.DataFrame()
    counts = counts.reindex(index=states, columns=states, fill_value=0)
    matrix = counts.div(counts.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0.0)

    summaries = []
    for state, group in episode_frame.groupby("state", sort=True):
        observed_spans = group["observed_label_span_seconds"]
        summaries.append({
            "state": state,
            "observed_episodes": int(len(group)),
            "sequences": int(group["sequence_id"].nunique()),
            "median_observed_label_span_seconds": float(observed_spans.median()),
            "p90_observed_label_span_seconds": float(observed_spans.quantile(0.9)),
            "first_observed_episodes": int(group["first_observed_episode"].sum()),
            "terminal_right_censored_episodes": int(group["terminal_right_censored"].sum()),
            "duration_interpretation": "sampled-label span, not true position dwell time",
        })
    return {
        "episodes": episode_frame,
        "transitions": transition_frame,
        "transition_counts": counts,
        "transition_matrix": matrix,
        "dwell_summary": pd.DataFrame(summaries),
    }


def write_vicos_analysis(
    observations: pd.DataFrame, out_dir: Path, fps: float = 30.0
) -> dict[str, Path]:
    results = analyze_vicos_positions(observations, fps)
    observations = observations.copy()
    observations["timestamp_ms"] = observations["frame_index"] * 1000.0 / fps
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "observations": out_dir / "position_observations.csv",
        "episodes": out_dir / "position_episodes.csv",
        "transitions": out_dir / "position_transitions.csv",
        "transition_counts": out_dir / "transition_counts.csv",
        "transition_matrix": out_dir / "transition_matrix.csv",
        "dwell_summary": out_dir / "observed_label_span_summary.csv",
        "metadata": out_dir / "analysis_metadata.json",
    }
    observations.to_csv(outputs["observations"], index=False)
    results["episodes"].to_csv(outputs["episodes"], index=False)
    results["transitions"].to_csv(outputs["transitions"], index=False)
    results["transition_counts"].to_csv(outputs["transition_counts"])
    results["transition_matrix"].to_csv(outputs["transition_matrix"])
    results["dwell_summary"].to_csv(outputs["dwell_summary"], index=False)
    outputs["metadata"].write_text(json.dumps({
        "analysis": "descriptive_bjj_position_transitions",
        "dataset": "ViCoS Brazilian Jiu-Jitsu Positions Dataset",
        "source_url": DATASET_URL,
        "license": LICENSE_ID,
        "attribution": "Hudovernik and Skočaj, Video-Based Detection of Combat Positions and Automatic Scoring in Jiu-jitsu, MMSports 2022.",
        "discipline_id": DISCIPLINE_ID,
        "fps_assumption": fps,
        "observations": int(len(observations)),
        "sequences": int(observations["sequence_id"].nunique()),
        "transition_rows": int(len(results["transitions"])),
        "interpretation": [
            "Position labels are collapsed across athlete-side suffixes; top/bottom identity is not modeled.",
            "The source contains selected labeled frames from six sparring sequences, not continuous event logs.",
            "Transition counts describe changes between consecutive observed labels and can miss unobserved intermediate states.",
            "Reported label spans are sampled-frame spans, not validated physical dwell times.",
            "Descriptive pilot only; no model training, universal score, or generalization claim.",
            "Non-commercial, attribution, and share-alike terms apply; review the dataset license before sharing outputs.",
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return outputs
