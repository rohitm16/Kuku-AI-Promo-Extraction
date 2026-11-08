"""Deterministic screenplay parser for Kuku microdrama scripts.

Parses multi-track production scripts into structured scenes and events
with centisecond timestamp precision and source traceability.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


TrackType = Literal["CAMERA", "ACTION", "DIALOGUE", "SOUND_FX", "MUSIC", "OTHER"]


class TrackEvent(BaseModel):
    """An atomic production event occurring on a specific track timeline."""
    event_id: str = Field(description="Unique identifier for the event, e.g., 'EP01_S01_E005'")
    track: TrackType = Field(description="Production track: CAMERA, ACTION, DIALOGUE, SOUND_FX, MUSIC")
    speaker: Optional[str] = Field(default=None, description="Character name if dialogue")
    parenthetical: Optional[str] = Field(default=None, description="Acting direction/emotion")
    content: str = Field(description="Spoken dialogue or visual/acoustic description")
    start_sec: float = Field(description="Timestamp in seconds from episode start")
    end_sec: float = Field(description="Timestamp in seconds from episode start")
    start_str: str = Field(description="Original timestamp string 'MM:SS.ss'")
    end_str: str = Field(description="Original timestamp string 'MM:SS.ss'")
    scene_num: int = Field(description="Scene index within the episode")
    source_file: str = Field(description="Source screenplay filename")
    line_number: int = Field(description="Starting line number in source file")

    @property
    def duration(self) -> float:
        return round(self.end_sec - self.start_sec, 2)


class Scene(BaseModel):
    """A distinct screenplay scene with environment heading and duration."""
    scene_num: int
    heading: str
    start_sec: float
    end_sec: float
    start_str: str
    end_str: str
    events: List[TrackEvent] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(self.end_sec - self.start_sec, 2)


class ParsedEpisode(BaseModel):
    """Complete structured representation of an episode."""
    episode_num: int
    episode_id: str
    logline: str
    continuity_notes: str
    source_file: str
    scenes: List[Scene] = Field(default_factory=list)

    @property
    def all_events(self) -> List[TrackEvent]:
        """Flat list of all events across all scenes in chronological order."""
        events: List[TrackEvent] = []
        for s in self.scenes:
            events.extend(s.events)
        return sorted(events, key=lambda e: (e.start_sec, e.end_sec))

    @property
    def total_duration_sec(self) -> float:
        """Total duration of the episode based on the final scene/event end time."""
        if not self.scenes:
            return 0.0
        return max(s.end_sec for s in self.scenes)


def parse_timestamp_str(time_str: str) -> float:
    """Convert 'MM:SS.ss' timestamp into seconds (float).

    Example: '02:18.50' -> 138.50
    """
    clean = time_str.strip().strip("()")
    parts = clean.split(":")
    if len(parts) != 2:
        raise ValueError(f"Malformed timestamp string: '{time_str}'")
    minutes = int(parts[0])
    seconds = float(parts[1])
    return round(minutes * 60 + seconds, 2)


def classify_track(raw_tag: str) -> tuple[TrackType, Optional[str], Optional[str]]:
    """Classify a markdown tag into track type, speaker name, and parenthetical emotion.

    Examples:
        'CAMERA:' -> ('CAMERA', None, None)
        'ACTION:' -> ('ACTION', None, None)
        'SOUND FX:' -> ('SOUND_FX', None, None)
        'MATRIARCH (Cold, Commanding):' -> ('DIALOGUE', 'MATRIARCH', 'Cold, Commanding')
        'LADY JULIANA:' -> ('DIALOGUE', 'LADY JULIANA', None)
    """
    clean_tag = raw_tag.strip().rstrip(":")
    tag_upper = clean_tag.upper()

    if tag_upper in ("CAMERA", "CAMERA:"):
        return "CAMERA", None, None
    if tag_upper in ("ACTION", "ACTION:"):
        return "ACTION", None, None
    if tag_upper in ("SOUND FX", "SOUND FX:", "SOUND EFFECT", "SOUND EFFECTS"):
        return "SOUND_FX", None, None
    if tag_upper in ("MUSIC", "MUSIC:"):
        return "MUSIC", None, None

    # Check for speaker with parenthetical: "SPEAKER (Emotion)"
    match = re.match(r"^([^(]+)(?:\(([^)]+)\))?", clean_tag)
    if match:
        speaker = match.group(1).strip()
        parenthetical = match.group(2).strip() if match.group(2) else None
        return "DIALOGUE", speaker, parenthetical

    return "OTHER", clean_tag, None


def parse_screenplay_text(text: str, filename: str = "unknown") -> ParsedEpisode:
    """Parse raw screenplay text into a fully structured ParsedEpisode object."""
    lines = text.splitlines()

    # 1. Parse Header Metadata
    ep_num = 0
    ep_id = ""
    logline = ""
    continuity_notes = ""

    # Extract episode number from filename or header
    fn_match = re.search(r"episode-(\d+)", filename, re.IGNORECASE)
    if fn_match:
        ep_num = int(fn_match.group(1))

    for line in lines[:30]:
        line_clean = line.strip()
        if line_clean.startswith("EPISODE NUMBER:"):
            header_val = line_clean.replace("EPISODE NUMBER:", "").strip()
            # If not yet extracted from filename, parse from header
            if ep_num == 0:
                num_m = re.search(r"(\d+)", header_val)
                if num_m:
                    ep_num = int(num_m.group(1))
            ep_id = header_val
        elif line_clean.startswith("EPISODE LOGLINE:"):
            logline = line_clean.replace("EPISODE LOGLINE:", "").strip()
        elif line_clean.startswith("CONTINUITY NOTES:"):
            continuity_notes = line_clean.replace("CONTINUITY NOTES:", "").strip()

    # 2. Extract Scenes and Script Content
    scenes: List[Scene] = []
    current_scene: Optional[Scene] = None
    event_counter = 0

    # Pattern for Scene Headers
    scene_pattern = re.compile(r"^SCENE\s+(\d+)", re.IGNORECASE)
    # Pattern for Timestamps: Start: (MM:SS.ss) | End: (MM:SS.ss)
    timestamp_pattern = re.compile(
        r"Start:\s*\(?(\d{2}:\d{2}\.\d{2})\)?\s*\|\s*End:\s*\(?(\d{2}:\d{2}\.\d{2})\)?"
    )

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()

        # Check for Scene Start
        sc_match = scene_pattern.match(line)
        if sc_match:
            sc_num = int(sc_match.group(1))
            heading = ""
            sc_start_str = "00:00.00"
            sc_end_str = "00:00.00"
            sc_start_sec = 0.0
            sc_end_sec = 0.0

            # Scan next few lines for heading and start/end
            look_idx = idx + 1
            while look_idx < min(idx + 10, len(lines)):
                look_line = lines[look_idx].strip()
                if look_line.startswith("Scene heading:"):
                    heading = look_line.replace("Scene heading:", "").strip()
                elif look_line.startswith("Scene start:"):
                    sc_start_str = look_line.replace("Scene start:", "").strip()
                    sc_start_sec = parse_timestamp_str(sc_start_str)
                elif look_line.startswith("Scene end:"):
                    sc_end_str = look_line.replace("Scene end:", "").strip()
                    sc_end_sec = parse_timestamp_str(sc_end_str)
                elif look_line.startswith("Actions and dialogue:"):
                    look_idx += 1
                    break
                look_idx += 1

            current_scene = Scene(
                scene_num=sc_num,
                heading=heading,
                start_sec=sc_start_sec,
                end_sec=sc_end_sec,
                start_str=sc_start_str,
                end_str=sc_end_str,
                events=[],
            )
            scenes.append(current_scene)
            idx = look_idx
            continue

        # Check for Tagged Production Event: **TAG:**
        if line.startswith("**") and "**" in line[2:] and current_scene is not None:
            end_tag_pos = line.find("**", 2)
            raw_tag = line[2:end_tag_pos].strip()
            line_num = idx + 1

            # Next line should contain timestamps
            if idx + 1 < len(lines):
                ts_line = lines[idx + 1].strip()
                ts_match = timestamp_pattern.search(ts_line)
                if ts_match:
                    start_str = ts_match.group(1)
                    end_str = ts_match.group(2)
                    start_sec = parse_timestamp_str(start_str)
                    end_sec = parse_timestamp_str(end_str)

                    # Next lines contain content until empty line, next tag, or script end
                    content_lines = []
                    content_idx = idx + 2
                    while content_idx < len(lines):
                        c_line = lines[content_idx].strip()
                        if not c_line:
                            content_idx += 1
                            break
                        if c_line.startswith("**") or c_line.startswith("SCENE ") or c_line.startswith("---"):
                            break
                        content_lines.append(c_line)
                        content_idx += 1

                    content = " ".join(content_lines).strip()
                    track, speaker, parenthetical = classify_track(raw_tag)
                    event_counter += 1
                    event_id = f"EP{ep_num:02d}_S{current_scene.scene_num:02d}_E{event_counter:03d}"

                    event = TrackEvent(
                        event_id=event_id,
                        track=track,
                        speaker=speaker,
                        parenthetical=parenthetical,
                        content=content,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        start_str=start_str,
                        end_str=end_str,
                        scene_num=current_scene.scene_num,
                        source_file=filename,
                        line_number=line_num,
                    )
                    current_scene.events.append(event)
                    idx = content_idx
                    continue

        idx += 1

    return ParsedEpisode(
        episode_num=ep_num,
        episode_id=ep_id or f"Episode {ep_num}",
        logline=logline,
        continuity_notes=continuity_notes,
        source_file=filename,
        scenes=scenes,
    )


def parse_screenplay_file(filepath: Path | str) -> ParsedEpisode:
    """Read and parse a screenplay file from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Screenplay file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_screenplay_text(text, filename=path.name)
