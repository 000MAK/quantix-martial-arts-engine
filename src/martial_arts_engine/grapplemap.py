"""Decode GrappleMap into a position/transition graph.

GrappleMap (Eelis van der Weegen, public domain) is a hand-built database of
grappling positions and transitions animated with stick figures. Decoding follows
its source (src/persistence.cpp, positions.cpp, graph.cpp):

* each frame is 2 players x 23 joints x (x, y, z), every coordinate two base-62
  digits / 1000, with x and z offset by -2 (y is height);
* a sequence with one frame is a named position, with several frames a transition;
* a transition connects the positions matching its first and last frame, where two
  frames match if, after rotating about the vertical axis, translating, optionally
  mirroring and optionally swapping the players, every joint lies within 4 cm.

Difference from GrappleMap's own code: the rotation/translation is found by
least-squares 2-D Procrustes alignment instead of its canonical orientation; the
4 cm joint tolerance and the head-to-head pre-check are the same.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE62 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DIGIT = {c: i for i, c in enumerate(BASE62)}
JOINTS = ["LeftToe", "RightToe", "LeftHeel", "RightHeel", "LeftAnkle", "RightAnkle", "LeftKnee", "RightKnee",
          "LeftHip", "RightHip", "LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow", "LeftWrist", "RightWrist",
          "LeftHand", "RightHand", "LeftFingers", "RightFingers", "Core", "Neck", "Head"]
N = len(JOINTS)
HEAD = JOINTS.index("Head")
# left/right pairs swapped by GrappleMap's mirror (all paired joints)
MIRROR_PERM = []
for j in JOINTS:
    MIRROR_PERM.append(JOINTS.index(j.replace("Left", "Right")) if j.startswith("Left")
                       else JOINTS.index(j.replace("Right", "Left")) if j.startswith("Right") else JOINTS.index(j))
MAX_JOINT_DIST_SQ = 0.0016   # GrappleMap basicallySame: 4 cm
HEAD2HEAD_TOL = 0.05         # GrappleMap is_reoriented pre-check on squared head distance
ENCODED = 2 * N * 3 * 2


def decode_frame(encoded: str) -> np.ndarray:
    """(2, 23, 3) joint coordinates in metres."""
    s = re.sub(r"\s+", "", encoded)
    if len(s) != ENCODED:
        raise ValueError(f"Encoded frame has {len(s)} characters, expected {ENCODED}")
    v = np.array([(DIGIT[s[i]] * 62 + DIGIT[s[i + 1]]) / 1000 for i in range(0, ENCODED, 2)])
    p = v.reshape(2, N, 3)
    p[..., 0] -= 2
    p[..., 2] -= 2
    return p


def parse(text: str) -> list[dict]:
    """Sequences in file order: description lines and decoded frames."""
    seqs, desc, frame_lines = [], [], []

    def flush_frames():
        if frame_lines:
            for k in range(0, len(frame_lines), 4):
                seqs[-1]["frames"].append(decode_frame("".join(frame_lines[k:k + 4])))
            frame_lines.clear()

    for line in text.splitlines():
        if line.startswith(" "):
            if not frame_lines and (not seqs or seqs[-1]["desc"] is not desc or seqs[-1]["frames"]):
                seqs.append({"desc": desc, "frames": []})
                desc = []
            frame_lines.append(line)
        else:
            flush_frames()
            if line.strip():
                desc.append(line)
    flush_frames()
    for s in seqs:
        s["frames"] = np.stack(s["frames"])
        name_lines = [l for l in s["desc"] if not re.match(r"^(tags|properties|ref):", l)]
        s["name"] = " ".join(name_lines).replace("\\n", " ").strip() or None
        s["tags"] = sorted({t for l in s["desc"] if l.startswith("tags:") for t in l[5:].split()})
        s["properties"] = sorted({t for l in s["desc"] if l.startswith("properties:") for t in l[11:].split()})
        s["refs"] = [l[4:].strip() for l in s["desc"] if l.startswith("ref:")]
    return seqs


def _head2head(p: np.ndarray) -> float:
    return float(np.sum((p[0, HEAD] - p[1, HEAD]) ** 2))


def _variants(p: np.ndarray):
    for swap in (False, True):
        q = p[::-1] if swap else p
        yield q
        m = q[:, MIRROR_PERM].copy()
        m[..., 0] *= -1
        yield m


def _aligned_max_dist_sq(a: np.ndarray, b: np.ndarray) -> float:
    """Align b onto a by rotation about the vertical (y) axis and x/z translation; max squared joint distance."""
    A, B = a.reshape(-1, 3), b.reshape(-1, 3)
    axz, bxz = A[:, [0, 2]], B[:, [0, 2]]
    ca, cb = axz.mean(0), bxz.mean(0)
    h = (bxz - cb).T @ (axz - ca)
    theta = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    r = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    moved = (bxz - cb) @ r + ca
    d = np.sum((moved - axz) ** 2, axis=1) + (B[:, 1] - A[:, 1]) ** 2
    return float(d.max())


def same_position(a: np.ndarray, b: np.ndarray) -> bool:
    if abs(_head2head(a) - _head2head(b)) > HEAD2HEAD_TOL:
        return False
    return any(_aligned_max_dist_sq(a, v) <= MAX_JOINT_DIST_SQ for v in _variants(b))


def build_graph(seqs: list[dict]) -> tuple[list[dict], list[dict]]:
    nodes, frames, h2h = [], [], []

    def find_or_add(frame: np.ndarray) -> int:
        hh = _head2head(frame)
        for i, (f, h) in enumerate(zip(frames, h2h)):
            if abs(h - hh) <= HEAD2HEAD_TOL and same_position(f, frame):
                return i
        nodes.append({"node_id": len(nodes), "name": None, "tags": [], "properties": [], "refs": []})
        frames.append(frame)
        h2h.append(hh)
        return len(nodes) - 1

    for s in seqs:
        if len(s["frames"]) == 1:
            i = find_or_add(s["frames"][0])
            if nodes[i]["name"] is None:
                nodes[i].update(name=s["name"], tags=s["tags"], properties=s["properties"], refs=s["refs"])
    edges = []
    for s in seqs:
        if len(s["frames"]) >= 2:
            edges.append({"edge_id": len(edges), "name": s["name"], "from": find_or_add(s["frames"][0]),
                          "to": find_or_add(s["frames"][-1]), "frames": int(len(s["frames"])), "tags": s["tags"],
                          "properties": s["properties"], "bidirectional": "bidirectional" in s["properties"], "refs": s["refs"]})
    for n, f in zip(nodes, frames):
        n["keyframe"] = np.round(f, 3).tolist()
    return nodes, edges


def graph_summary(nodes: list[dict], edges: list[dict]) -> dict:
    out_deg, in_deg = defaultdict(int), defaultdict(int)
    adj = defaultdict(set)
    for e in edges:
        out_deg[e["from"]] += 1
        in_deg[e["to"]] += 1
        adj[e["from"]].add(e["to"])
        if e["bidirectional"]:
            adj[e["to"]].add(e["from"])
    named = sum(n["name"] is not None for n in nodes)
    top = sorted(nodes, key=lambda n: -(in_deg[n["node_id"]] + out_deg[n["node_id"]]))[:10]
    return {"nodes": len(nodes), "named_positions": named, "unnamed_endpoint_nodes": len(nodes) - named,
            "transitions": len(edges), "bidirectional_transitions": sum(e["bidirectional"] for e in edges),
            "isolated_named_positions": sum(1 for n in nodes if in_deg[n["node_id"]] + out_deg[n["node_id"]] == 0),
            "most_connected": [{"node_id": n["node_id"], "name": n["name"], "in": in_deg[n["node_id"]], "out": out_deg[n["node_id"]]} for n in top]}


def export(grapplemap_txt: Path, out_dir: Path, source_commit: str | None = None) -> dict:
    seqs = parse(grapplemap_txt.read_text(encoding="utf-8"))
    nodes, edges = build_graph(seqs)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "positions.jsonl").open("w", encoding="utf-8") as f:
        for n in nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    with (out_dir / "transitions.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    summary = {**graph_summary(nodes, edges), "source": "https://github.com/Eelis/GrappleMap", "source_commit": source_commit,
               "license": "Public domain (GrappleMap LICENSE)", "frame_encoding": "2 players x 23 joints x (x, y, z) metres; y is height",
               "joints": JOINTS}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "README.md").write_text(dataset_card(summary), encoding="utf-8")
    return summary


def dataset_card(s: dict) -> str:
    top = "\n".join(f"| {m['name']} | {m['in']} | {m['out']} |" for m in s["most_connected"][:8])
    return f"""---
license: other
license_name: public-domain-dedication
license_link: https://github.com/Eelis/GrappleMap/blob/master/LICENSE
pretty_name: GrappleMap BJJ position-transition graph
tags:
- grappling
- brazilian-jiu-jitsu
- graph
- pose
configs:
- config_name: positions
  data_files: positions.jsonl
- config_name: transitions
  data_files: transitions.jsonl
---

# GrappleMap BJJ position-transition graph

A machine-readable graph of grappling positions and transitions, decoded from
[GrappleMap](https://github.com/Eelis/GrappleMap) by **Eelis van der Weegen** (source commit `{s['source_commit']}`).
All credit for the positions, transitions, names, tags and technique references goes to GrappleMap and its contributors;
this dataset only converts its text database into JSON Lines and resolves which transitions connect which positions.

## Contents

- `positions.jsonl`: {s['nodes']} positions ({s['named_positions']} named in GrappleMap, {s['unnamed_endpoint_nodes']} unnamed transition endpoints), with tags, properties, references and one keyframe.
- `transitions.jsonl`: {s['transitions']} transitions (`from`, `to`, frame count, tags, properties, references); {s['bidirectional_transitions']} are bidirectional.
- `summary.json`: counts and the joint order.

Keyframes are 2 players x 23 joints x (x, y, z) in metres, y up; joints: {', '.join(s['joints'])}.

| Most connected position | In | Out |
|---|---|---|
{top}

## How the graph was built

Decoding follows GrappleMap's source (`persistence.cpp`, `positions.cpp`, `graph.cpp`): two base-62 digits per coordinate divided by 1000, x and z offset by -2.
A transition connects the positions matching its first and last frame; two frames match when, after rotation about the vertical
axis, translation, optional mirroring and optional player swap, every joint lies within 4 cm (GrappleMap's tolerance).
Unlike GrappleMap, the rotation and translation are found by least-squares Procrustes alignment, so a few borderline
matches may differ from the GrappleMap website. All {s['named_positions']} named positions were kept distinct.

## Limitations

GrappleMap is a hand-authored technique map: it shows which transitions are possible in its author's model, not how
often they happen in real matches, and it covers a subset of BJJ. Edge counts are not probabilities.

## Licence

GrappleMap's contributors released their work into the public domain; this conversion is released under the same terms.
Code: [quantix-martial-arts-engine](https://github.com/000MAK/quantix-martial-arts-engine) (`grapplemap.py`, MIT).
"""
