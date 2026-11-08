"""Unit tests for Candidate Window generation and 3-Lane Pruning."""

from pathlib import Path
import os
import pytest

from kuku_promo_system.config import (
    DEFAULT_DATA_DIR,
    MIN_PROMO_DURATION,
    MAX_PROMO_DURATION,
    TEMPORAL_NMS_IOU_THRESHOLD,
)
from kuku_promo_system.parser import parse_screenplay_file
from kuku_promo_system.timeline import build_episode_timeline
from kuku_promo_system.pruner import (
    CandidateWindow,
    compute_temporal_iou,
    apply_temporal_nms,
    generate_all_valid_windows,
    prune_candidates_3_lane,
)


def test_temporal_iou_calculation():
    """Verify 1D temporal IoU overlap computation."""
    c1 = CandidateWindow(
        candidate_id="C1",
        episode_num=1,
        scene_start=1,
        scene_end=1,
        start_beat_id=1,
        end_beat_id=3,
        start_sec=10.0,
        end_sec=35.0,
        start_str="00:10.00",
        end_str="00:35.00",
        duration_sec=25.0,
        lane="boundary",
        window_text="sample",
    )
    # Identical window -> IoU = 1.0
    assert compute_temporal_iou(c1, c1) == 1.0

    # Non-overlapping window -> IoU = 0.0
    c2 = c1.model_copy(update={"start_sec": 40.0, "end_sec": 70.0, "duration_sec": 30.0})
    assert compute_temporal_iou(c1, c2) == 0.0

    # Partial overlap: [10, 35] and [20, 45] -> Intersection=[20,35]=15s, Union=[10,45]=35s -> 15/35 = 0.4286
    c3 = c1.model_copy(update={"start_sec": 20.0, "end_sec": 45.0, "duration_sec": 25.0})
    assert abs(compute_temporal_iou(c1, c3) - (15.0 / 35.0)) < 0.01


@pytest.mark.skipif(not DEFAULT_DATA_DIR.exists(), reason="DEFAULT_DATA_DIR does not exist")
def test_candidate_duration_bounds():
    """Verify that every generated candidate satisfies 20.0s <= duration <= 60.0s."""
    fpath = list(DEFAULT_DATA_DIR.glob("episode-001-*.txt"))[0]
    ep = parse_screenplay_file(fpath)
    beats = build_episode_timeline(ep)

    raw_cands = generate_all_valid_windows(beats, episode_num=ep.episode_num)
    assert len(raw_cands) > 10, f"Expected >10 valid windows, got {len(raw_cands)}"

    for c in raw_cands:
        assert MIN_PROMO_DURATION <= c.duration_sec <= MAX_PROMO_DURATION
        assert c.validate_bounds() is True


@pytest.mark.skipif(not DEFAULT_DATA_DIR.exists(), reason="DEFAULT_DATA_DIR does not exist")
def test_3_lane_pruning_on_episode_001():
    """Verify high-recall 3-lane pruning on real Episode 1."""
    fpath = list(DEFAULT_DATA_DIR.glob("episode-001-*.txt"))[0]
    ep = parse_screenplay_file(fpath)
    beats = build_episode_timeline(ep)

    raw_cands = generate_all_valid_windows(beats, episode_num=ep.episode_num)
    shortlist = prune_candidates_3_lane(raw_cands, target_count=10)

    # Verify shortlist size is in the approved 8 to 12 range
    assert 8 <= len(shortlist) <= 12, f"Expected 8-12 candidates, got {len(shortlist)}"

    # Verify no two candidates exceed IoU threshold
    for i in range(len(shortlist)):
        for j in range(i + 1, len(shortlist)):
            iou = compute_temporal_iou(shortlist[i], shortlist[j])
            assert iou <= TEMPORAL_NMS_IOU_THRESHOLD, f"NMS failed: {shortlist[i].candidate_id} and {shortlist[j].candidate_id} have IoU {iou}"

    # Verify lanes are represented
    lanes = set(c.lane for c in shortlist)
    assert len(lanes) >= 2, f"Expected diversity across lanes, got {lanes}"

    # Verify hermetic window text and lookahead text
    for c in shortlist:
        assert len(c.window_text) > 30, "Window text must not be empty"
        assert c.duration_sec >= 20.0 and c.duration_sec <= 60.0


@pytest.mark.skipif(not DEFAULT_DATA_DIR.exists(), reason="DEFAULT_DATA_DIR does not exist")
def test_pruning_on_short_and_long_episodes():
    """Verify pruning works consistently on short (Ep 12) and long (Ep 3) episodes."""
    for ep_num in (3, 12):
        fpath = list(DEFAULT_DATA_DIR.glob(f"episode-{ep_num:03d}-*.txt"))[0]
        ep = parse_screenplay_file(fpath)
        beats = build_episode_timeline(ep)
        raw_cands = generate_all_valid_windows(beats, episode_num=ep.episode_num)
        shortlist = prune_candidates_3_lane(raw_cands, target_count=10)

        assert len(shortlist) >= 4, f"Episode {ep_num} produced too few candidates: {len(shortlist)}"
        for c in shortlist:
            assert 20.0 <= c.duration_sec <= 60.0


def test_synthetic_reaches_episode_end_before_cut_to_black():
    """Verify that a candidate window ending before a terminal-only 'Cut to black' beat has reaches_episode_end=True."""
    from kuku_promo_system.parser import TrackEvent
    from kuku_promo_system.timeline import AtomicBeat
    from kuku_promo_system.pruner import reaches_episode_end, is_terminal_only_beat

    b1 = AtomicBeat(
        beat_id=1,
        episode_num=1,
        scene_num=1,
        start_sec=0.0,
        end_sec=15.0,
        start_str="00:00.00",
        end_str="00:15.00",
        events=[
            TrackEvent(
                event_id="E1",
                track="DIALOGUE",
                speaker="HERO",
                content="This is the end.",
                start_sec=0.0,
                end_sec=15.0,
                start_str="00:00.00",
                end_str="00:15.00",
                scene_num=1,
                source_file="ep.txt",
                line_number=1,
            )
        ],
    )
    b2 = AtomicBeat(
        beat_id=2,
        episode_num=1,
        scene_num=1,
        start_sec=15.0,
        end_sec=35.0,
        start_str="00:15.00",
        end_str="00:35.00",
        events=[
            TrackEvent(
                event_id="E2",
                track="DIALOGUE",
                speaker="VILLAIN",
                content="Not yet it isn't.",
                start_sec=15.0,
                end_sec=35.0,
                start_str="00:15.00",
                end_str="00:35.00",
                scene_num=1,
                source_file="ep.txt",
                line_number=5,
            )
        ],
    )
    b3 = AtomicBeat(
        beat_id=3,
        episode_num=1,
        scene_num=1,
        start_sec=35.0,
        end_sec=36.0,
        start_str="00:35.00",
        end_str="00:36.00",
        events=[
            TrackEvent(
                event_id="E3",
                track="CAMERA",
                content="Cut to black.",
                start_sec=35.0,
                end_sec=36.0,
                start_str="00:35.00",
                end_str="00:36.00",
                scene_num=1,
                source_file="ep.txt",
                line_number=10,
            )
        ],
    )

    beats = [b1, b2, b3]
    assert is_terminal_only_beat(b3) is True
    assert reaches_episode_end(1, beats) is True
    assert reaches_episode_end(2, beats) is True

    windows = generate_all_valid_windows(beats, episode_num=1)
    # The window spanning b1 and b2 (duration 35s) ends at index 1 before the terminal beat b3
    w_before_cut = next((w for w in windows if w.end_beat_id == 2), None)
    assert w_before_cut is not None
    assert w_before_cut.reaches_episode_end is True


@pytest.mark.skipif(not DEFAULT_DATA_DIR.exists(), reason="DEFAULT_DATA_DIR does not exist")
def test_real_data_every_episode_shortlist_has_flagged_candidate():
    """Verify that every episode's shortlist across the canonical 50 episodes contains at least one flagged candidate."""
    import re
    files = sorted(
        [DEFAULT_DATA_DIR / f for f in os.listdir(DEFAULT_DATA_DIR) if f.endswith(".txt")],
        key=lambda p: int(re.search(r'Episode\s*(\d+)', p.name, re.I).group(1))
    )
    assert len(files) == 50

    for fpath in files:
        ep = parse_screenplay_file(fpath)
        beats = build_episode_timeline(ep)
        windows = generate_all_valid_windows(beats, episode_num=ep.episode_num)
        shortlist = prune_candidates_3_lane(windows, target_count=10)
        has_flagged = any(c.reaches_episode_end for c in shortlist)
        assert has_flagged, f"Episode {ep.episode_num} shortlist does not have any candidate with reaches_episode_end=True"
