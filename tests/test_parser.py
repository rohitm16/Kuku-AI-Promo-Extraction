"""Unit and integration tests for the deterministic Screenplay Parser."""

from pathlib import Path
import pytest

from kuku_promo_system.config import DEFAULT_DATA_DIR
from kuku_promo_system.parser import (
    parse_timestamp_str,
    classify_track,
    parse_screenplay_text,
    parse_screenplay_file,
    ParsedEpisode,
)

SAMPLE_SNIPPET = """EPISODE NUMBER: Episode 99
EPISODE LOGLINE: A test logline describing dramatic events.
CONTINUITY NOTES: Test continuity notes for validation.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. GREAT HALL - NIGHT
Scene start: 00:00.00
Scene end: 01:30.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:03.50)
Wide establishing shot of the banquet hall.

**ACTION:**
Start: (00:01.00) | End: (00:04.00)
Lady Juliana enters abruptly, tears in her eyes.

**MATRIARCH (Cold, Commanding):**
Start: (00:05.00) | End: (00:08.20)
"How dare you return here, Juliana?"

**SOUND FX:**
Start: (00:08.00) | End: (00:10.50)
Thunder cracks outside the cathedral windows.

--- SCRIPT END ---
"""


def test_timestamp_parsing():
    """Verify conversion of MM:SS.ss to floating-point seconds."""
    assert parse_timestamp_str("00:00.00") == 0.0
    assert parse_timestamp_str("00:04.80") == 4.80
    assert parse_timestamp_str("01:30.00") == 90.0
    assert parse_timestamp_str("02:18.50") == 138.50
    assert parse_timestamp_str("(02:18.50)") == 138.50

    with pytest.raises(ValueError):
        parse_timestamp_str("invalid_time")


def test_track_classification():
    """Verify mapping of raw tags to track types and character metadata."""
    t1, s1, p1 = classify_track("CAMERA:")
    assert t1 == "CAMERA" and s1 is None and p1 is None

    t2, s2, p2 = classify_track("ACTION:")
    assert t2 == "ACTION" and s2 is None and p2 is None

    t3, s3, p3 = classify_track("SOUND FX:")
    assert t3 == "SOUND_FX" and s3 is None and p3 is None

    t4, s4, p4 = classify_track("MATRIARCH (Cold, Commanding):")
    assert t4 == "DIALOGUE"
    assert s4 == "MATRIARCH"
    assert p4 == "Cold, Commanding"

    t5, s5, p5 = classify_track("LADY JULIANA:")
    assert t5 == "DIALOGUE"
    assert s5 == "LADY JULIANA"
    assert p5 is None


def test_parse_snippet():
    """Verify parsing of a synthetic multi-track screenplay snippet."""
    ep = parse_screenplay_text(SAMPLE_SNIPPET, filename="episode-099-test.txt")
    assert ep.episode_num == 99
    assert "test logline" in ep.logline
    assert len(ep.scenes) == 1

    scene = ep.scenes[0]
    assert scene.scene_num == 1
    assert scene.heading == "INT. GREAT HALL - NIGHT"
    assert scene.start_sec == 0.0
    assert scene.end_sec == 90.0

    events = scene.events
    assert len(events) == 4

    # Verify event tracks and timing
    assert events[0].track == "CAMERA"
    assert events[0].start_sec == 0.0
    assert events[0].end_sec == 3.50

    assert events[1].track == "ACTION"
    assert events[1].start_sec == 1.0
    assert events[1].end_sec == 4.0

    assert events[2].track == "DIALOGUE"
    assert events[2].speaker == "MATRIARCH"
    assert events[2].parenthetical == "Cold, Commanding"
    assert '"How dare you return here, Juliana?"' in events[2].content

    assert events[3].track == "SOUND_FX"
    assert events[3].start_sec == 8.0
    assert events[3].end_sec == 10.50


@pytest.mark.skipif(not DEFAULT_DATA_DIR.exists(), reason="DEFAULT_DATA_DIR does not exist")
def test_parse_real_episode_001():
    """Verify parser execution on actual Episode 1 from the input pack."""
    files = list(DEFAULT_DATA_DIR.glob("episode-001-*.txt"))
    assert len(files) == 1, "Episode 1 file must exist"

    ep = parse_screenplay_file(files[0])
    assert ep.episode_num == 1
    assert "bridal gown" in ep.logline.lower()
    assert len(ep.scenes) >= 1

    all_evs = ep.all_events
    assert len(all_evs) > 30, f"Expected >30 events in Ep 1, got {len(all_evs)}"

    # Assert timestamp validity across all events
    for ev in all_evs:
        assert ev.end_sec >= ev.start_sec, f"Negative duration in event {ev.event_id}"
        assert ev.start_sec >= 0.0
        assert ev.source_file == files[0].name


@pytest.mark.skipif(not DEFAULT_DATA_DIR.exists(), reason="DEFAULT_DATA_DIR does not exist")
def test_parse_all_50_episodes_without_errors():
    """Integration test: parse all 50 screenplay files in the dataset."""
    files = sorted(list(DEFAULT_DATA_DIR.glob("episode-*.txt")))
    assert len(files) == 50, f"Expected 50 files, got {len(files)}"

    total_events = 0
    total_scenes = 0

    for fpath in files:
        ep = parse_screenplay_file(fpath)
        assert ep.episode_num >= 1 and ep.episode_num <= 50
        assert len(ep.logline) > 10, f"Missing logline in {fpath.name}"
        assert len(ep.scenes) >= 1, f"No scenes parsed in {fpath.name}"
        total_scenes += len(ep.scenes)
        total_events += len(ep.all_events)

    assert total_scenes >= 50
    assert total_events > 3000, f"Expected >3000 total events, got {total_events}"
