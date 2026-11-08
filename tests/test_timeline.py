"""Unit tests for multi-track timeline synchronization and Atomic Beat clustering."""

from pathlib import Path
import pytest

from kuku_promo_system.config import DEFAULT_DATA_DIR
from kuku_promo_system.parser import TrackEvent, Scene, parse_screenplay_file
from kuku_promo_system.timeline import (
    format_seconds,
    should_split_beat_cluster,
    cluster_scene_events,
    build_episode_timeline,
    AtomicBeat,
)


def test_format_seconds():
    """Verify conversion of seconds to MM:SS.ss strings."""
    assert format_seconds(0.0) == "00:00.00"
    assert format_seconds(4.8) == "00:04.80"
    assert format_seconds(75.25) == "01:15.25"
    assert format_seconds(138.5) == "02:18.50"


def test_concurrent_events_merge():
    """Verify that simultaneous Camera and Action events merge into a single beat."""
    e1 = TrackEvent(
        event_id="E1",
        track="CAMERA",
        speaker=None,
        parenthetical=None,
        content="Camera push in on church doors",
        start_sec=0.0,
        end_sec=4.80,
        start_str="00:00.00",
        end_str="00:04.80",
        scene_num=1,
        source_file="test.txt",
        line_number=1,
    )
    e2 = TrackEvent(
        event_id="E2",
        track="ACTION",
        speaker=None,
        parenthetical=None,
        content="Juliana enters cathedral",
        start_sec=1.0,
        end_sec=4.80,
        start_str="00:01.00",
        end_str="00:04.80",
        scene_num=1,
        source_file="test.txt",
        line_number=5,
    )

    scene = Scene(
        scene_num=1,
        heading="INT. CATHEDRAL",
        start_sec=0.0,
        end_sec=10.0,
        start_str="00:00.00",
        end_str="00:10.00",
        events=[e1, e2],
    )

    beats = cluster_scene_events(scene, episode_num=1)
    assert len(beats) == 1, "Concurrent camera and action events must merge into one beat"
    assert beats[0].start_sec == 0.0
    assert beats[0].end_sec == 4.80
    assert beats[0].has_camera_cut is True
    assert beats[0].has_action is True
    assert len(beats[0].events) == 2


def test_speaker_alternation_splits_beat():
    """Verify that dialogue turn between two characters splits beats even with zero gap."""
    e1 = TrackEvent(
        event_id="E1",
        track="DIALOGUE",
        speaker="MATRIARCH",
        parenthetical="Cold",
        content='"Turn around, Juliana."',
        start_sec=11.30,
        end_sec=13.10,
        start_str="00:11.30",
        end_str="00:13.10",
        scene_num=1,
        source_file="test.txt",
        line_number=10,
    )
    e2 = TrackEvent(
        event_id="E2",
        track="DIALOGUE",
        speaker="JULIANA",
        parenthetical="Breathless",
        content='"Why?"',
        start_sec=13.20,
        end_sec=14.30,
        start_str="00:13.20",
        end_str="00:14.30",
        scene_num=1,
        source_file="test.txt",
        line_number=15,
    )

    assert should_split_beat_cluster([e1], e2) is True

    scene = Scene(
        scene_num=1,
        heading="INT. CATHEDRAL",
        start_sec=0.0,
        end_sec=20.0,
        start_str="00:00.00",
        end_str="00:20.00",
        events=[e1, e2],
    )
    beats = cluster_scene_events(scene, episode_num=1)
    assert len(beats) == 2, "Speaker alternation must produce two separate Atomic Beats"
    assert beats[0].speakers == ["MATRIARCH"]
    assert beats[1].speakers == ["JULIANA"]


@pytest.mark.skipif(not DEFAULT_DATA_DIR.exists(), reason="DEFAULT_DATA_DIR does not exist")
def test_real_episode_001_timeline():
    """Verify timeline clustering on actual Episode 1."""
    fpath = list(DEFAULT_DATA_DIR.glob("episode-001-*.txt"))[0]
    ep = parse_screenplay_file(fpath)
    beats = build_episode_timeline(ep)

    # Episode 1 has 95 raw events; clustering should yield 15-35 coherent dramatic beats
    assert 15 <= len(beats) <= 35, f"Expected 15-35 beats, got {len(beats)}"

    # Check chronological monotonicity
    for i in range(len(beats) - 1):
        assert beats[i].beat_id == i + 1
        assert beats[i].end_sec >= beats[i].start_sec
        assert beats[i + 1].start_sec >= beats[i].start_sec


@pytest.mark.skipif(not DEFAULT_DATA_DIR.exists(), reason="DEFAULT_DATA_DIR does not exist")
def test_scene_boundary_isolation():
    """Verify that Atomic Beats never cross scene boundaries."""
    files = list(DEFAULT_DATA_DIR.glob("episode-003-*.txt"))
    ep = parse_screenplay_file(files[0])
    beats = build_episode_timeline(ep)

    scene_nums = set(b.scene_num for b in beats)
    assert len(scene_nums) >= 2, "Episode 3 should have multiple scenes"

    for b in beats:
        for ev in b.events:
            assert ev.scene_num == b.scene_num
