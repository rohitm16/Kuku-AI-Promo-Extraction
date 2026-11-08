"""Multi-track timeline synchronization and Atomic Beat clustering.

Fuses concurrent CAMERA, ACTION, DIALOGUE, and SOUND FX events into
cohesive narrative beats while preventing beat bloat via micro-beat splitters.
"""

from __future__ import annotations

from typing import List, Optional, Set
from pydantic import BaseModel, Field

from kuku_promo_system.config import BEAT_MERGE_GAP_SEC
from kuku_promo_system.parser import TrackEvent, Scene, ParsedEpisode, TrackType


def format_seconds(seconds: float) -> str:
    """Format floating-point seconds into 'MM:SS.ss' string."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


class AtomicBeat(BaseModel):
    """An atomic unit of dramatic action/reaction within a scene."""
    beat_id: int
    episode_num: int
    scene_num: int
    start_sec: float
    end_sec: float
    start_str: str
    end_str: str
    events: List[TrackEvent] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(self.end_sec - self.start_sec, 2)

    @property
    def speakers(self) -> List[str]:
        return sorted(list(set(e.speaker for e in self.events if e.speaker)))

    @property
    def has_camera_cut(self) -> bool:
        return any(e.track == "CAMERA" for e in self.events)

    @property
    def has_action(self) -> bool:
        return any(e.track == "ACTION" for e in self.events)

    @property
    def has_dialogue(self) -> bool:
        return any(e.track == "DIALOGUE" for e in self.events)

    @property
    def has_sound_fx(self) -> bool:
        return any(e.track in ("SOUND_FX", "MUSIC") for e in self.events)

    @property
    def primary_text(self) -> str:
        """Formatted text representation of all events in this beat for Cold-Viewer evaluation."""
        lines = []
        for e in self.events:
            if e.track == "DIALOGUE" and e.speaker:
                parenthetical = f" ({e.parenthetical})" if e.parenthetical else ""
                lines.append(f"{e.speaker}{parenthetical}: {e.content}")
            elif e.track == "ACTION":
                lines.append(f"[ACTION] {e.content}")
            elif e.track == "CAMERA":
                lines.append(f"[CAMERA] {e.content}")
            elif e.track in ("SOUND_FX", "MUSIC"):
                lines.append(f"[AUDIO] {e.content}")
            else:
                lines.append(e.content)
        return "\n".join(lines)


def should_split_beat_cluster(
    current_events: List[TrackEvent],
    next_event: TrackEvent,
    max_gap: float = BEAT_MERGE_GAP_SEC,
    max_beat_duration: float = 12.0,
) -> bool:
    """Determine whether next_event should start a new Atomic Beat.

    Splits occur if:
    1. Temporal gap: The time between current events' end and next_event exceeds max_gap.
    2. Speaker alternation: A new character speaks when the current beat already has dialogue.
    3. Terminal sound cues or explicit cut-to-black directives.
    4. Duration cap: The current beat exceeds max_beat_duration (preventing beat bloat).
    5. Post-dialogue transition: When dialogue in current beat has finished and a new camera/action sequence starts.
    """
    if not current_events:
        return False

    curr_start = min(e.start_sec for e in current_events)
    curr_end = max(e.end_sec for e in current_events)
    curr_speakers: Set[str] = set(e.speaker for e in current_events if e.speaker)

    # 1. Temporal gap check
    if next_event.start_sec - curr_end > max_gap:
        return True

    # 2. Speaker alternation check: new speaker arrives
    if next_event.speaker and curr_speakers and (next_event.speaker not in curr_speakers):
        return True

    # 3. Explicit terminal cue in current beat
    for e in current_events:
        c_low = e.content.lower()
        if "cut to black" in c_low or "cliffhanger peak" in c_low:
            return True

    # 4. Duration cap: beat is already >= max_beat_duration
    if next_event.start_sec - curr_start >= max_beat_duration:
        return True

    # 5. Post-dialogue transition: dialogue ended and an independent camera or action starts >= 3s into the beat
    dialogue_ends = [e.end_sec for e in current_events if e.track == "DIALOGUE"]
    if dialogue_ends:
        max_diag_end = max(dialogue_ends)
        if (
            next_event.track in ("CAMERA", "ACTION")
            and next_event.start_sec >= max_diag_end - 0.2
            and (next_event.start_sec - curr_start >= 3.0)
        ):
            return True

    return False


def cluster_scene_events(scene: Scene, episode_num: int, start_beat_id: int = 1) -> List[AtomicBeat]:
    """Cluster events within a single scene into Atomic Beats."""
    if not scene.events:
        return []

    # Sort events strictly by start time, then end time
    sorted_events = sorted(scene.events, key=lambda e: (e.start_sec, e.end_sec))

    beats: List[AtomicBeat] = []
    current_events: List[TrackEvent] = [sorted_events[0]]

    for next_ev in sorted_events[1:]:
        if should_split_beat_cluster(current_events, next_ev):
            # Finalize current beat
            b_start = min(e.start_sec for e in current_events)
            b_end = max(e.end_sec for e in current_events)
            beat = AtomicBeat(
                beat_id=start_beat_id + len(beats),
                episode_num=episode_num,
                scene_num=scene.scene_num,
                start_sec=round(b_start, 2),
                end_sec=round(b_end, 2),
                start_str=format_seconds(b_start),
                end_str=format_seconds(b_end),
                events=list(current_events),
            )
            beats.append(beat)
            current_events = [next_ev]
        else:
            current_events.append(next_ev)

    # Finalize trailing beat
    if current_events:
        b_start = min(e.start_sec for e in current_events)
        b_end = max(e.end_sec for e in current_events)
        beat = AtomicBeat(
            beat_id=start_beat_id + len(beats),
            episode_num=episode_num,
            scene_num=scene.scene_num,
            start_sec=round(b_start, 2),
            end_sec=round(b_end, 2),
            start_str=format_seconds(b_start),
            end_str=format_seconds(b_end),
            events=list(current_events),
        )
        beats.append(beat)

    return beats


def build_episode_timeline(episode: ParsedEpisode) -> List[AtomicBeat]:
    """Transform an entire ParsedEpisode into a synchronized sequence of Atomic Beats."""
    all_beats: List[AtomicBeat] = []
    beat_counter = 1

    for scene in episode.scenes:
        scene_beats = cluster_scene_events(scene, episode_num=episode.episode_num, start_beat_id=beat_counter)
        all_beats.extend(scene_beats)
        beat_counter += len(scene_beats)

    return all_beats
