"""Candidate window generation, multi-track feature extraction, and 3-lane pruning.

Extracts candidate windows strictly within 20.0s to 60.0s duration bounds,
computes multi-track dramatic features, and selects 8-12 high-diversity candidates
per episode using Temporal Non-Maximum Suppression (NMS).
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field

from kuku_promo_system.config import (
    MIN_PROMO_DURATION,
    MAX_PROMO_DURATION,
    TEMPORAL_NMS_IOU_THRESHOLD,
    LANE_QUOTAS,
    validate_duration,
)
from kuku_promo_system.timeline import AtomicBeat, format_seconds


class CandidateWindow(BaseModel):
    """A valid 20-60 second promotional candidate window."""
    candidate_id: str
    episode_num: int
    scene_start: int
    scene_end: int
    start_beat_id: int
    end_beat_id: int
    start_sec: float
    end_sec: float
    start_str: str
    end_str: str
    duration_sec: float
    lane: str
    boundary_score: float = 0.0
    development_score: float = 0.0
    anomaly_score: float = 0.0
    prune_rank_score: float = 0.0
    reaches_episode_end: bool = False
    window_text: str = Field(description="Text slice for Cold-Viewer evaluation")
    lookahead_text: str = Field(default="", description="Next atomic beat text for resolution checking")
    beats: List[AtomicBeat] = Field(default_factory=list)

    def validate_bounds(self) -> bool:
        return validate_duration(self.duration_sec)


# High-stakes thematic keywords for Lane 3 (Anomaly / Subtle Drama)
THEMATIC_KEYWORDS = re.compile(
    r"\b(betray|betrayal|abandon|abandoned|altar|wedding|mother|bridal|gown|fianc[eé]|"
    r"jewel|sister|bastard|worthless|plainness|"
    r"ruin|ruined|secret|mistress|arrest|theft|liar|kill|murder|poison|disown|disowned)\b",
    re.IGNORECASE,
)

# Kinetic action verbs for visual hooks
KINETIC_VERBS = re.compile(
    r"\b(slap|slaps|strike|strikes|tear|tears|clench|clenches|freeze|freezes|draw|draws|"
    r"confront|confronts|pause|pauses|enter|enters|grab|grabs|drop|drops|push|pushes|"
    r"shatter|shatters|collapse|collapses)\b",
    re.IGNORECASE,
)

# High-intensity emotional delivery markers
INTENSE_PARENTHTICALS = re.compile(
    r"\b(cold|commanding|desperate|unbelieving|trembling|pleading|blunt|unforgiving|"
    r"arrogant|dismissive|resolute|defiance|furious|whisper|whispers|sobbing|screaming)\b",
    re.IGNORECASE,
)

# Terminal cues indicating end of episode
TERMINAL_CUE = re.compile(
    r"cut to black|fade to black|fade out|black screen|to be continued",
    re.IGNORECASE,
)


def is_terminal_only_beat(beat: AtomicBeat) -> bool:
    """True if the beat has no DIALOGUE event and at least one event's content matches TERMINAL_CUE."""
    has_dialogue = any(e.track == "DIALOGUE" for e in beat.events)
    if has_dialogue:
        return False
    return any(bool(TERMINAL_CUE.search(e.content)) for e in beat.events)


def reaches_episode_end(end_index: int, beats: List[AtomicBeat]) -> bool:
    """True if every beat after end_index is terminal-only. The last beat is therefore always True."""
    after_beats = beats[end_index + 1 :]
    if not after_beats:
        return True
    return all(is_terminal_only_beat(b) for b in after_beats)


def compute_temporal_iou(w1: CandidateWindow, w2: CandidateWindow) -> float:
    """Compute 1D Temporal Intersection-over-Union between two time windows."""
    intersection_start = max(w1.start_sec, w2.start_sec)
    intersection_end = min(w1.end_sec, w2.end_sec)
    intersection = max(0.0, intersection_end - intersection_start)

    union_start = min(w1.start_sec, w2.start_sec)
    union_end = max(w1.end_sec, w2.end_sec)
    union = max(0.001, union_end - union_start)

    return round(intersection / union, 4)


def apply_temporal_nms(candidates: List[CandidateWindow], iou_threshold: float = TEMPORAL_NMS_IOU_THRESHOLD) -> List[CandidateWindow]:
    """Suppress overlapping candidates that share the same temporal region."""
    if not candidates:
        return []

    # Sort descending by prune_rank_score
    sorted_cands = sorted(candidates, key=lambda c: c.prune_rank_score, reverse=True)
    selected: List[CandidateWindow] = []

    for cand in sorted_cands:
        is_suppressed = False
        for kept in selected:
            if compute_temporal_iou(cand, kept) > iou_threshold:
                is_suppressed = True
                break
        if not is_suppressed:
            selected.append(cand)

    return selected


def score_boundary_features(window_beats: List[AtomicBeat]) -> float:
    """Calculate Lane 1 score: Opening Hook Salience + Ending Cliffhanger Salience."""
    if not window_beats:
        return 0.0

    score = 0.0
    first_beat = window_beats[0]
    last_beat = window_beats[-1]

    # --- Hook Salience (First Beat) ---
    for ev in first_beat.events:
        content_lower = ev.content.lower()
        if ev.track == "CAMERA":
            if any(k in content_lower for k in ("close-up", "extreme close-up", "push-in", "tight insert")):
                score += 1.5
        elif ev.track == "ACTION":
            if KINETIC_VERBS.search(content_lower):
                score += 1.5
        elif ev.track == "DIALOGUE":
            if ev.parenthetical and INTENSE_PARENTHTICALS.search(ev.parenthetical):
                score += 1.0
        elif ev.track in ("SOUND_FX", "MUSIC"):
            score += 1.5

    # --- Cliffhanger Salience (Last Beat) ---
    for ev in last_beat.events:
        content_lower = ev.content.lower()
        if "cut to black" in content_lower or "cliffhanger peak" in content_lower or "drone swells" in content_lower:
            score += 2.5
        if ev.track == "DIALOGUE":
            # Ends with question, ellipsis or exclamation
            stripped = ev.content.strip().rstrip('"').strip()
            if stripped.endswith("?") or stripped.endswith("...") or stripped.endswith("!"):
                score += 1.5
            if ev.parenthetical and INTENSE_PARENTHTICALS.search(ev.parenthetical):
                score += 1.0

    return round(score, 2)


def score_development_features(window_beats: List[AtomicBeat], duration_sec: float) -> float:
    """Calculate Lane 2 score: Dynamic progression, pacing acceleration, and speaker alternation."""
    if len(window_beats) < 2 or duration_sec <= 0:
        return 0.0

    duration_min = duration_sec / 60.0
    mid_sec = window_beats[0].start_sec + (duration_sec / 2.0)

    # 1. Pacing Acceleration (Camera cuts in 2nd half minus 1st half)
    cuts_first_half = sum(1 for b in window_beats if b.end_sec <= mid_sec for e in b.events if e.track == "CAMERA")
    cuts_second_half = sum(1 for b in window_beats if b.start_sec >= mid_sec for e in b.events if e.track == "CAMERA")
    acceleration = max(0, cuts_second_half - cuts_first_half)

    # 2. Speaker Alternation Velocity
    speakers_in_window = [b.speakers[0] for b in window_beats if b.speakers]
    speaker_switches = 0
    for i in range(len(speakers_in_window) - 1):
        if speakers_in_window[i] != speakers_in_window[i + 1]:
            speaker_switches += 1

    switch_rate = speaker_switches / duration_min

    # 3. Total Event Density
    total_events = sum(len(b.events) for b in window_beats)
    event_density = total_events / duration_min

    score = (acceleration * 1.5) + (switch_rate * 1.0) + (event_density * 0.2)
    return round(score, 2)


def score_anomaly_features(window_beats: List[AtomicBeat], duration_sec: float) -> float:
    """Calculate Lane 3 score: High-stakes thematic keywords and intimate visual inserts."""
    if not window_beats or duration_sec <= 0:
        return 0.0

    full_text = " ".join(b.primary_text for b in window_beats)
    keyword_matches = len(THEMATIC_KEYWORDS.findall(full_text))

    insert_bonus = 0.0
    for b in window_beats:
        for e in b.events:
            c_low = e.content.lower()
            if "tight insert" in c_low or "extreme close-up" in c_low:
                insert_bonus += 2.0

    score = (keyword_matches * 1.2) + insert_bonus
    return round(score, 2)


def generate_all_valid_windows(beats: List[AtomicBeat], episode_num: int) -> List[CandidateWindow]:
    """Generate all valid candidate windows satisfying 20.0s <= duration <= 60.0s."""
    candidates: List[CandidateWindow] = []
    n_beats = len(beats)

    for i in range(n_beats):
        for j in range(i, n_beats):
            # Candidate window spanning beats[i] through beats[j]
            start_sec = beats[i].start_sec
            end_sec = beats[j].end_sec
            dur = round(end_sec - start_sec, 2)

            if dur < MIN_PROMO_DURATION:
                continue
            if dur > MAX_PROMO_DURATION:
                # Since beats are chronological, extending j further will only increase duration
                break

            window_beats = beats[i : j + 1]
            window_text = "\n\n".join(b.primary_text for b in window_beats)

            # Lookahead text: the immediate next atomic beat
            lookahead_text = ""
            if j + 1 < n_beats:
                lookahead_text = beats[j + 1].primary_text

            cand = CandidateWindow(
                candidate_id=f"EP{episode_num:02d}_CAND_{i+1:02d}_{j+1:02d}",
                episode_num=episode_num,
                scene_start=beats[i].scene_num,
                scene_end=beats[j].scene_num,
                start_beat_id=beats[i].beat_id,
                end_beat_id=beats[j].beat_id,
                start_sec=start_sec,
                end_sec=end_sec,
                start_str=format_seconds(start_sec),
                end_str=format_seconds(end_sec),
                duration_sec=dur,
                lane="unassigned",
                boundary_score=score_boundary_features(window_beats),
                development_score=score_development_features(window_beats, dur),
                anomaly_score=score_anomaly_features(window_beats, dur),
                reaches_episode_end=reaches_episode_end(j, beats),
                window_text=window_text,
                lookahead_text=lookahead_text,
                beats=window_beats,
            )
            assert cand.validate_bounds(), f"Invalid duration {dur}s in generated candidate"
            candidates.append(cand)

    return candidates


def prune_candidates_3_lane(
    all_candidates: List[CandidateWindow],
    target_count: int = 10,
) -> List[CandidateWindow]:
    """Select 8-12 diverse, high-recall candidates across the 3 lanes using NMS."""
    if not all_candidates:
        return []

    # If already fewer than target_count, return all sorted by boundary score
    if len(all_candidates) <= target_count:
        for c in all_candidates:
            c.lane = "boundary"
            c.prune_rank_score = c.boundary_score
        return sorted(all_candidates, key=lambda c: c.boundary_score, reverse=True)

    # 1. Lane 1: Boundary-Driven
    lane1_pool = [c.model_copy() for c in all_candidates]
    for c in lane1_pool:
        c.lane = "boundary"
        c.prune_rank_score = c.boundary_score
    lane1_selected = apply_temporal_nms(lane1_pool, TEMPORAL_NMS_IOU_THRESHOLD)[: LANE_QUOTAS["boundary"]]

    # 2. Lane 2: Development-Driven
    lane2_pool = [c.model_copy() for c in all_candidates]
    for c in lane2_pool:
        c.lane = "development"
        c.prune_rank_score = c.development_score
    lane2_selected = apply_temporal_nms(lane2_pool, TEMPORAL_NMS_IOU_THRESHOLD)[: LANE_QUOTAS["development"]]

    # 3. Lane 3: Anomaly-Driven
    lane3_pool = [c.model_copy() for c in all_candidates]
    for c in lane3_pool:
        c.lane = "anomaly"
        c.prune_rank_score = c.anomaly_score
    lane3_selected = apply_temporal_nms(lane3_pool, TEMPORAL_NMS_IOU_THRESHOLD)[: LANE_QUOTAS["anomaly"]]

    # Combine selected candidates from all 3 lanes
    combined: List[CandidateWindow] = []
    seen_ids = set()

    for cand_list in (lane1_selected, lane2_selected, lane3_selected):
        for c in cand_list:
            if c.candidate_id not in seen_ids:
                seen_ids.add(c.candidate_id)
                combined.append(c)

    # Final cross-lane NMS to prevent duplicates across lanes
    final_shortlist = apply_temporal_nms(combined, TEMPORAL_NMS_IOU_THRESHOLD)

    # If NMS pruned below 8 and we have more candidates, backfill from remaining pool
    if len(final_shortlist) < 8 and len(all_candidates) >= 8:
        remaining = [c for c in all_candidates if c.candidate_id not in seen_ids]
        remaining_sorted = sorted(remaining, key=lambda c: (c.boundary_score + c.anomaly_score), reverse=True)
        for r in remaining_sorted:
            is_dup = any(compute_temporal_iou(r, kept) > TEMPORAL_NMS_IOU_THRESHOLD for kept in final_shortlist)
            if not is_dup:
                r.lane = "backfill"
                r.prune_rank_score = r.boundary_score
                final_shortlist.append(r)
                if len(final_shortlist) >= target_count:
                    break

    # Ensure at least one candidate reaches episode end
    if not any(c.reaches_episode_end for c in final_shortlist):
        flagged = [c for c in all_candidates if c.reaches_episode_end]
        flagged_sorted = sorted(flagged, key=lambda c: c.boundary_score, reverse=True)
        chosen = None
        for cand in flagged_sorted:
            if all(compute_temporal_iou(cand, sc) <= TEMPORAL_NMS_IOU_THRESHOLD for sc in final_shortlist):
                chosen = cand.model_copy(update={"lane": "episode_end"})
                final_shortlist.append(chosen)
                break
        if chosen is None and flagged_sorted:
            top_window = flagged_sorted[0]
            highest_iou = -1.0
            highest_idx = 0
            for idx, sc in enumerate(final_shortlist):
                iou = compute_temporal_iou(top_window, sc)
                if iou > highest_iou:
                    highest_iou = iou
                    highest_idx = idx
            final_shortlist[highest_idx] = top_window.model_copy(update={"lane": "episode_end"})

    return sorted(final_shortlist, key=lambda c: c.start_sec)
