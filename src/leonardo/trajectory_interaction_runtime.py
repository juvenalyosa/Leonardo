"""Pure helpers for trajectory interaction analysis.

The PyMOL controller owns structure loading and plotting.  This module keeps
frame selection, occupancy accounting, and per-frame interaction matching
deterministic and testable without a live PyMOL session.
"""

from __future__ import annotations

import math

import numpy as np

CLUSTER_INTERACTION_TYPES = ("Hydrophobic", "Van der Waals")
RING_INTERACTION_TYPES = ("Pi-Pi Stacking", "T-Shaped Pi-Pi", "Cation-Pi", "Anion-Pi")


def trajectory_normalize_step(step=1):
    """Return a safe positive stride for 1-based PyMOL trajectory frames."""
    try:
        value = int(step)
    except Exception:
        value = 1
    return int(max(1, value))


def trajectory_frame_indices(n_states=0, step=1, include_last=True):
    """
    Build the 1-based frame list used by trajectory analyses.

    PyMOL states are 1-based.  Including the last state avoids silently missing
    the final conformation when `step` does not divide `n_states - 1`.
    """
    try:
        n = int(n_states or 0)
    except Exception:
        n = 0
    n = max(0, int(n))
    if n <= 0:
        return []
    stride = trajectory_normalize_step(step)
    frames = list(range(1, n + 1, stride))
    if bool(include_last) and frames and frames[-1] != n:
        frames.append(n)
    return [int(i) for i in frames]


def trajectory_normalize_max_distance(value=None):
    """Return a usable Angstrom cutoff, or None when the filter is disabled."""
    try:
        dist = float(value)
    except Exception:
        return None
    if not np.isfinite(dist) or dist <= 0.0:
        return None
    return float(dist)


def trajectory_interaction_key(inter):
    """Strict atom-level key for one interaction record."""
    data = dict(inter or {}) if isinstance(inter, dict) else {}
    return (
        str(data.get("type", "") or ""),
        str(data.get("prot_res", "") or ""),
        str(data.get("prot_atom", "") or ""),
        str(data.get("lig_atom", "") or ""),
    )


def trajectory_interaction_occ_key(inter, coarse=False):
    """
    Stable key for frame-to-frame occupancy.

    Cluster/ring interactions can move between equivalent atoms as coordinates
    fluctuate, so coarse mode aggregates them at the residue or ring level.
    """
    data = dict(inter or {}) if isinstance(inter, dict) else {}
    itype = str(data.get("type", "") or "")
    prot_res = str(data.get("prot_res", "") or "")
    prot_atom = str(data.get("prot_atom", "") or "")
    lig_atom = str(data.get("lig_atom", "") or "")
    details = str(data.get("details", "") or "")
    if not bool(coarse):
        return ("atom", itype, prot_res, prot_atom, lig_atom)
    if details == "Cluster" or itype in CLUSTER_INTERACTION_TYPES:
        return ("cluster", itype, prot_res)
    if itype in RING_INTERACTION_TYPES:
        return ("ring", itype, prot_res)
    return ("atom", itype, prot_res, prot_atom, lig_atom)


def trajectory_interaction_label(inter):
    """Compact label for occupancy plots."""
    data = dict(inter or {}) if isinstance(inter, dict) else {}
    itype = str(data.get("type", "") or "")
    prot_res = str(data.get("prot_res", "") or "")
    lig_atom = str(data.get("lig_atom", "") or "")
    if str(data.get("details", "") or "") == "Cluster" or itype in CLUSTER_INTERACTION_TYPES:
        return f"{prot_res}-Cluster"
    if itype in RING_INTERACTION_TYPES and lig_atom == "Ring":
        return f"{prot_res}-Ring"
    return f"{prot_res}-{lig_atom}"


def _safe_distance(value):
    try:
        dist = float(value)
    except Exception:
        return math.nan
    return float(dist) if np.isfinite(dist) else math.nan


def trajectory_collect_interaction_occupancy(
    frame_interactions=None,
    *,
    max_distance=None,
    top_n=20,
    coarse=True,
    color_map=None,
):
    """
    Aggregate per-frame interaction records into occupancy percentages.

    `frame_interactions` is expected to be an iterable where each item contains
    the interaction dictionaries detected in one sampled frame.  Empty frames
    still count toward occupancy denominators.
    """
    frames = list(frame_interactions or [])
    frames_analyzed = int(len(frames))
    cutoff = trajectory_normalize_max_distance(max_distance)
    cmap = dict(color_map or {})
    try:
        top_n_i = int(top_n or 20)
    except Exception:
        top_n_i = 20
    top_n_i = max(1, int(top_n_i))

    counts = {}
    meta = {}
    skipped_distance = 0
    skipped_invalid = 0
    for frame in frames:
        seen_this_frame = set()
        for inter in list(frame or []):
            if not isinstance(inter, dict):
                skipped_invalid += 1
                continue
            dist = _safe_distance(inter.get("distance", math.nan))
            if not np.isfinite(dist):
                skipped_invalid += 1
                continue
            if cutoff is not None and dist > float(cutoff):
                skipped_distance += 1
                continue
            key = trajectory_interaction_occ_key(inter, coarse=bool(coarse))
            # A residue-level interaction appearing multiple times in the same
            # frame should contribute one occupancy hit, not one hit per atom.
            if key in seen_this_frame:
                continue
            seen_this_frame.add(key)
            counts[key] = int(counts.get(key, 0)) + 1
            if key not in meta:
                meta[key] = {
                    "type": str(inter.get("type", "") or ""),
                    "label": trajectory_interaction_label(inter),
                    "example": dict(inter),
                }

    if frames_analyzed <= 0:
        return {
            "ok": False,
            "reason": "no_frames_analyzed",
            "frames_analyzed": 0,
            "counts": {},
            "meta": {},
            "sorted_items": [],
            "top_items": [],
            "top_labels": [],
            "top_freqs": [],
            "top_colors": [],
            "all_occ": np.asarray([], dtype=float),
            "skipped_distance": int(skipped_distance),
            "skipped_invalid": int(skipped_invalid),
        }

    sorted_items = sorted(
        counts.items(),
        key=lambda item: (
            -int(item[1]),
            str(meta.get(item[0], {}).get("type", "")),
            str(meta.get(item[0], {}).get("label", "")),
            repr(item[0]),
        ),
    )
    top_items = sorted_items[: min(top_n_i, len(sorted_items))]
    top_labels = [str(meta.get(k, {}).get("label", str(k))) for k, _v in top_items]
    top_freqs = [100.0 * float(v) / float(frames_analyzed) for _k, v in top_items]
    top_colors = [str(cmap.get(str(meta.get(k, {}).get("type", "")), "gray")) for k, _v in top_items]
    all_occ = np.asarray([100.0 * float(v) / float(frames_analyzed) for _k, v in sorted_items], dtype=float)

    return {
        "ok": bool(len(sorted_items) > 0),
        "reason": "ready" if sorted_items else "no_interactions",
        "frames_analyzed": int(frames_analyzed),
        "counts": dict(counts),
        "meta": dict(meta),
        "sorted_items": list(sorted_items),
        "top_items": list(top_items),
        "top_n": int(len(top_items)),
        "top_labels": list(top_labels),
        "top_freqs": [float(x) for x in top_freqs],
        "top_colors": list(top_colors),
        "all_occ": all_occ,
        "skipped_distance": int(skipped_distance),
        "skipped_invalid": int(skipped_invalid),
        "max_distance": cutoff,
    }


def trajectory_collect_presence_counts(frame_interactions=None):
    """
    Count residue and interaction-type presence once per sampled frame.

    This is used for ABFE post-processing summaries where per-frame persistence,
    not atom-level multiplicity, is the desired statistic.
    """
    residue_counts = {}
    type_counts = {}
    frames = list(frame_interactions or [])
    for frame in frames:
        residues = set()
        types = set()
        for inter in list(frame or []):
            if not isinstance(inter, dict):
                continue
            pres = str(inter.get("prot_res", "") or "").strip()
            if pres:
                residues.add(pres)
            itype = str(inter.get("type", "") or "").strip()
            if itype:
                types.add(itype)
        for res in residues:
            residue_counts[str(res)] = int(residue_counts.get(str(res), 0)) + 1
        for itype in types:
            type_counts[str(itype)] = int(type_counts.get(str(itype), 0)) + 1
    return {
        "ok": True,
        "frames_analyzed": int(len(frames)),
        "residue_counts": dict(residue_counts),
        "type_counts": dict(type_counts),
    }


def trajectory_run_interaction_analysis(
    *,
    n_states=0,
    step=1,
    calculate_interactions_fn=None,
    set_frame_fn=None,
    get_current_frame_fn=None,
    max_distance=None,
    top_n=20,
    color_map=None,
    progress_set_fn=None,
    progress_canceled_fn=None,
    include_last=True,
):
    """
    Execute the frame loop for trajectory interaction analysis.

    The UI supplies PyMOL callbacks; this helper owns the control flow so frame
    sampling, cancellation, aggregation, and state restoration are covered by
    unit tests.
    """
    if not callable(calculate_interactions_fn):
        return {
            "ok": False,
            "reason": "calculate_interactions_fn_missing",
            "frame_plan": [],
            "sampled_frames": [],
            "frame_interactions": [],
            "frames_analyzed": 0,
            "canceled": False,
        }
    frame_plan = trajectory_frame_indices(n_states, step=step, include_last=include_last)
    if not frame_plan:
        return {
            "ok": False,
            "reason": "no_frames_analyzed",
            "frame_plan": [],
            "sampled_frames": [],
            "frame_interactions": [],
            "frames_analyzed": 0,
            "canceled": False,
        }

    original_frame = None
    if callable(get_current_frame_fn):
        try:
            original_frame = int(get_current_frame_fn())
        except Exception:
            original_frame = None

    frame_interactions = []
    sampled_frames = []
    canceled = False
    try:
        for frame_pos, frame_idx in enumerate(frame_plan, start=1):
            if callable(progress_canceled_fn):
                try:
                    if bool(progress_canceled_fn()):
                        canceled = True
                        break
                except Exception:
                    pass
            if callable(progress_set_fn):
                try:
                    progress_set_fn(int(frame_pos - 1))
                except Exception:
                    pass
            if callable(set_frame_fn):
                set_frame_fn(int(frame_idx))
            current = calculate_interactions_fn(int(frame_idx))
            frame_interactions.append(list(current or []))
            sampled_frames.append(int(frame_idx))
        if callable(progress_set_fn):
            try:
                progress_set_fn(int(len(sampled_frames)))
            except Exception:
                pass
    finally:
        if original_frame is not None and callable(set_frame_fn):
            try:
                set_frame_fn(max(1, int(original_frame)))
            except Exception:
                pass

    payload = trajectory_collect_interaction_occupancy(
        frame_interactions,
        max_distance=max_distance,
        top_n=top_n,
        coarse=True,
        color_map=color_map,
    )
    payload["frame_plan"] = [int(x) for x in frame_plan]
    payload["sampled_frames"] = [int(x) for x in sampled_frames]
    payload["frame_interactions"] = list(frame_interactions)
    payload["canceled"] = bool(canceled)
    payload["original_frame"] = original_frame
    if canceled and not bool(payload.get("ok", False)) and not sampled_frames:
        payload["reason"] = "analysis_canceled"
    return payload


def trajectory_pick_matching_interaction(current_interactions=None, target_inter=None):
    """
    Find the best current-frame match for a previously selected interaction.

    The strict atom key is tried first.  If it fails, coarse occupancy keys allow
    cluster and ring interactions to remain trackable when representative atoms
    change between frames.
    """
    target = dict(target_inter or {}) if isinstance(target_inter, dict) else {}
    current = [x for x in list(current_interactions or []) if isinstance(x, dict)]
    if not target or not current:
        return None
    target_key = trajectory_interaction_key(target)
    by_key = {trajectory_interaction_key(x): x for x in current}
    hit = by_key.get(target_key)
    if hit is not None:
        return dict(hit)

    target_occ_key = trajectory_interaction_occ_key(target, coarse=True)
    relaxed = [x for x in current if trajectory_interaction_occ_key(x, coarse=True) == target_occ_key]
    if not relaxed:
        return None
    ref_d = _safe_distance(target.get("distance", math.nan))

    def _score(candidate):
        d = _safe_distance(candidate.get("distance", math.nan))
        if np.isfinite(ref_d) and np.isfinite(d):
            return (0, abs(float(d) - float(ref_d)), float(d))
        if np.isfinite(d):
            return (1, float(d), 0.0)
        return (2, 1.0e9, 0.0)

    return dict(min(relaxed, key=_score))


def trajectory_build_selected_trace_payload(frames=None, present=None, distances=None):
    """Normalize trace arrays and compute occupancy/distance summary."""
    frame_arr = np.asarray(list(frames or []), dtype=int)
    present_arr = np.asarray(list(present or []), dtype=float)
    dist_arr = np.asarray(list(distances or []), dtype=float)
    n = int(min(frame_arr.size, present_arr.size, dist_arr.size))
    frame_arr = frame_arr[:n]
    present_arr = present_arr[:n]
    dist_arr = dist_arr[:n]
    present_arr = np.where(np.isfinite(present_arr), present_arr, 0.0)
    present_arr = np.clip(present_arr, 0.0, 1.0)
    valid_dist = dist_arr[np.isfinite(dist_arr)]
    occupancy_pct = 100.0 * float(np.mean(present_arr)) if n > 0 else math.nan
    return {
        "ok": bool(n > 0),
        "reason": "ready" if n > 0 else "no_frames_analyzed",
        "frames": frame_arr,
        "present": present_arr,
        "distances": dist_arr,
        "dist_valid": valid_dist,
        "occupancy_pct": float(occupancy_pct) if np.isfinite(occupancy_pct) else math.nan,
        "distance_mean": float(np.mean(valid_dist)) if valid_dist.size else math.nan,
        "distance_median": float(np.median(valid_dist)) if valid_dist.size else math.nan,
        "distance_min": float(np.min(valid_dist)) if valid_dist.size else math.nan,
        "distance_max": float(np.max(valid_dist)) if valid_dist.size else math.nan,
        "n_frames": int(n),
        "n_present": int(np.sum(present_arr > 0.5)) if n > 0 else 0,
    }
