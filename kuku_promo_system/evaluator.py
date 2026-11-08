"""Cold-Viewer evaluation module with batch packaging and model-aware checkpointing.

Evaluates candidate promo windows strictly within their isolated time slices,
enforcing multi-modal hook grading, Next-Beat lookahead checking, and Pydantic validation.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Callable
from pydantic import BaseModel, Field

from kuku_promo_system.config import (
    DEFAULT_OUTPUT_DIR,
    OPENAI_OUTPUT_DIR,
    BATCH_SIZE,
    validate_duration,
)
from kuku_promo_system.logger import logger
from kuku_promo_system.pruner import CandidateWindow
from kuku_promo_system.llm_client import LLMClientAdapter


class SinglePromoEvaluation(BaseModel):
    """Structured semantic evaluation for an individual candidate promo."""
    candidate_id: str
    hook_score: float = Field(ge=1.0, le=10.0, description="Scroll-stopping shock in first 3s (1.0 to 10.0)")
    hook_type: str = Field(description="visual | audio_verbal | audiovisual_sync")
    hook_description: str = Field(description="Exact opening event that arrests attention")
    cliffhanger_score: float = Field(ge=1.0, le=10.0, description="Dramatic unresolved tension at cut point (1.0 to 10.0)")
    cliffhanger_type: str = Field(description="scene_terminal | intra_scene")
    cliffhanger_question: str = Field(description="Urgent unresolved question left in the viewer's mind")
    is_immediately_resolved: bool = Field(description="True if lookahead text immediately deflates the question")
    standalone_clarity_score: float = Field(ge=1.0, le=10.0, description="Self-sufficiency for a new viewer (1.0 to 10.0)")
    identified_conflict: str = Field(description="Central dramatic conflict discernible from this slice alone")
    requires_prior_lore: bool = Field(description="True if clip is incomprehensible without earlier episodes")
    editorial_rationale: str = Field(description="One concise line explaining why this promo works")
    is_simulated: bool = Field(default=False, description="True if generated via mock testing mode")
    evaluation_source: str = Field(default="unknown", description="'openai' | 'openrouter' | 'mock'")
    model_id: str = Field(default="", description="Model ID used for evaluation")
    prompt_tokens: Optional[int] = Field(default=None, description="Input tokens used")
    completion_tokens: Optional[int] = Field(default=None, description="Output tokens generated")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens consumed")
    latency_sec: Optional[float] = Field(default=None, description="Request duration in seconds")
    prompt_version: str = Field(default="v1", description="Evaluation prompt version")
    evaluated_at: Optional[str] = Field(default=None, description="ISO timestamp of evaluation")


class BatchPromoEvaluation(BaseModel):
    """Structured response container for a batch of candidate evaluations."""
    evaluations: List[SinglePromoEvaluation]


class EvaluatedCandidate(BaseModel):
    """Full operational record of a candidate window with its evaluation and composite score."""
    candidate: CandidateWindow
    evaluation: Optional[SinglePromoEvaluation] = None
    status: str = Field(default="pending", description="'evaluated' | 'evaluation_failed'")
    error_message: Optional[str] = None
    composite_score: float = 0.0

    def compute_composite_score(self) -> float:
        """Calculate composite narrative potential score with penalty multipliers."""
        if not self.evaluation or self.status != "evaluated":
            self.composite_score = 0.0
            return 0.0

        ev = self.evaluation
        # Base weighted score: Hook (35%) + Cliffhanger (35%) + Standalone Clarity (30%)
        base = (0.35 * ev.hook_score) + (0.35 * ev.cliffhanger_score) + (0.30 * ev.standalone_clarity_score)

        # Lookahead Penalty: if cliffhanger is immediately resolved in the next beat, apply 30% penalty
        p_lookahead = 0.70 if ev.is_immediately_resolved else 1.0

        # Lore Penalty: if clip is incomprehensible without prior episodes, apply 50% penalty
        p_lore = 0.50 if ev.requires_prior_lore else 1.0

        self.composite_score = round(base * p_lookahead * p_lore, 2)
        return self.composite_score


COLD_VIEWER_SYSTEM_PROMPT = """You are a senior video editor and microdrama marketing strategist for mobile short-form video (TikTok, Instagram Reels, YouTube Shorts).
Your job is to evaluate candidate promotional clips (20-60 seconds) extracted from screenplay production scripts.

CRITICAL OPERATIONAL RULES:
1. HERMETIC COLD-VIEWER ISOLATION:
   You are an uninitiated viewer who has NEVER watched this show before.
   You must evaluate each candidate strictly and independently using ONLY the provided text slice between its start and end timestamps.
   Do not use knowledge from other candidates or external lore. If an uninitiated stranger cannot understand who is fighting or what is at stake from this slice alone, flag `requires_prior_lore: true` and rate `standalone_clarity_score` low.

2. MULTI-MODAL HOOK EVALUATION:
   Evaluate the first 3 seconds of the clip.
   Identify whether the hook is 'visual' (slap, strike, dramatic entrance, camera push), 'audio_verbal' (shocking dialogue, insult, threat), or 'audiovisual_sync' (shocking line combined with kinetic action or close-up).

3. RESOLUTION LOOKAHEAD AUDIT:
   You are provided with a 'Lookahead Context' showing what happens immediately after the cut point.
   Determine whether the dramatic question posed at the cut point is immediately resolved or deflated in the lookahead. If it is immediately resolved, set `is_immediately_resolved: true`.

4. STRICT JSON OUTPUT FORMAT:
   Return a valid JSON object containing an "evaluations" array with an object for EVERY candidate in the batch.
   EVERY field below is strictly required for each evaluation:
   {
     "evaluations": [
       {
         "candidate_id": "EP01_CAND_01_09",
         "hook_score": 8.5,
         "hook_type": "visual",
         "hook_description": "Precise description of opening hook event in first 3s",
         "cliffhanger_score": 9.0,
         "cliffhanger_type": "scene_terminal",
         "cliffhanger_question": "Precise unresolved question left in viewer's mind",
         "is_immediately_resolved": false,
         "standalone_clarity_score": 7.0,
         "identified_conflict": "Clear statement of the central dramatic conflict",
         "requires_prior_lore": false,
         "editorial_rationale": "One concise sentence on why this promo works"
       }
     ]
   }
   Valid hook_type values: "visual", "audio_verbal", "audiovisual_sync".
   Valid cliffhanger_type values: "scene_terminal", "intra_scene".
"""


def build_batch_prompt(candidates: List[CandidateWindow]) -> str:
    """Format a batch of candidates into an isolated, structured user prompt."""
    prompt_sections = []
    prompt_sections.append(f"Please evaluate the following {len(candidates)} candidate promotional clips independently:\n")

    for idx, c in enumerate(candidates, 1):
        section = [
            f"--- CANDIDATE {idx} ---",
            f"Candidate ID: {c.candidate_id}",
            f"Episode: {c.episode_num}",
            f"Timestamps: {c.start_str} to {c.end_str} (Duration: {c.duration_sec:.2f}s)",
            f"Lane: {c.lane}",
            "\n[ISOLATED CANDIDATE TRANSCRIPT SLICE]:",
            c.window_text,
        ]
        if c.lookahead_text:
            section.extend([
                "\n[IMMEDIATE NEXT-BEAT LOOKAHEAD CONTEXT (AFTER CUT)]:",
                c.lookahead_text,
            ])
        else:
            section.append("\n[IMMEDIATE NEXT-BEAT LOOKAHEAD CONTEXT (AFTER CUT)]: (Scene or Episode End - Cut to Black)")

        prompt_sections.append("\n".join(section))

    return "\n\n".join(prompt_sections)


class PromoEvaluator:
    """Orchestrates batch evaluations with model-aware persistent checkpointing."""

    def __init__(
        self,
        llm_client: Optional[LLMClientAdapter] = None,
        checkpoint_dir: Path = OPENAI_OUTPUT_DIR,
        checkpoint_filename: str = "eval_checkpoint.jsonl",
        batch_size: int = BATCH_SIZE,
    ):
        self.llm_client = llm_client or LLMClientAdapter()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_file = self.checkpoint_dir / checkpoint_filename
        self.batch_size = batch_size
        self._ensure_checkpoint_dir()

    def _ensure_checkpoint_dir(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_checkpoints(self) -> Dict[str, EvaluatedCandidate]:
        """Load previously evaluated candidates from disk matching the active model and configuration."""
        cached: Dict[str, EvaluatedCandidate] = {}
        if not self.checkpoint_file.exists():
            return cached

        with open(self.checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        cand_record = EvaluatedCandidate.model_validate(data)

                        # Always skip simulated records
                        if cand_record.evaluation and cand_record.evaluation.is_simulated:
                            continue

                        # Only records with status == 'evaluated' are considered completed
                        if cand_record.status != "evaluated":
                            continue

                        # MODEL-AWARE CHECKPOINT VALIDATION:
                        # Do not reuse cached evaluations from a different model or provider!
                        if cand_record.evaluation:
                            if cand_record.evaluation.evaluation_source != self.llm_client.provider:
                                continue
                            if cand_record.evaluation.model_id != self.llm_client.model:
                                continue

                        cached[cand_record.candidate.candidate_id] = cand_record
                    except Exception as e:
                        logger.warning(f"Skipping corrupt checkpoint line: {e}")
        return cached

    def append_checkpoint(self, record: EvaluatedCandidate):
        """Append a single evaluated candidate to the checkpoint file."""
        with open(self.checkpoint_file, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def evaluate_candidates(
        self,
        candidates: List[CandidateWindow],
        progress_callback: Optional[Callable[[int, int, List[EvaluatedCandidate], str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> List[EvaluatedCandidate]:
        """Evaluate candidate windows in batches with checkpointing and real-time progress callbacks."""
        cached_records = self.load_checkpoints()
        results: List[EvaluatedCandidate] = []
        pending: List[CandidateWindow] = []

        # Check which candidates have already been evaluated for this configuration
        rejected_mismatched = 0
        for c in candidates:
            if c.candidate_id in cached_records:
                cached_rec = cached_records[c.candidate_id]
                if (
                    cached_rec.candidate.window_text == c.window_text
                    and cached_rec.candidate.lookahead_text == c.lookahead_text
                ):
                    results.append(cached_rec)
                else:
                    rejected_mismatched += 1
                    pending.append(c)
            else:
                pending.append(c)

        if rejected_mismatched > 0:
            logger.info(
                f"Rejected {rejected_mismatched} checkpoint record(s) with mismatched window_text or lookahead_text."
            )

        if not pending:
            logger.info(f"All {len(candidates)} candidates resolved from local checkpoint.")
            if progress_callback:
                progress_callback(1, 1, results, f"All {len(candidates)} candidates loaded from cache.")
            return results

        total_batches = (len(pending) + self.batch_size - 1) // self.batch_size
        logger.info(
            f"Evaluating {len(pending)} pending candidates in {total_batches} batches of {self.batch_size} "
            f"(Provider: {self.llm_client.provider}, Model: {self.llm_client.model})..."
        )

        for batch_num, i in enumerate(range(0, len(pending), self.batch_size), 1):
            # Check user cancellation
            if is_cancelled and is_cancelled():
                logger.info("Pipeline execution paused by user cancellation request.")
                break

            batch = pending[i : i + self.batch_size]
            user_prompt = build_batch_prompt(batch)

            messages = [
                {"role": "system", "content": COLD_VIEWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            batch_resp, is_simulated, err, usage_dict, latency = self.llm_client.generate_structured_completion(
                messages=messages,
                response_model=BatchPromoEvaluation,
                temperature=0.0,
            )

            # Map responses back to candidate windows
            resp_map = {}
            if batch_resp and batch_resp.evaluations:
                p_tokens = usage_dict.get("prompt_tokens")
                c_tokens = usage_dict.get("completion_tokens")
                t_tokens = usage_dict.get("total_tokens")
                now_iso = datetime.now(timezone.utc).isoformat()

                for ev in batch_resp.evaluations:
                    ev.is_simulated = is_simulated
                    ev.evaluation_source = self.llm_client.provider
                    ev.model_id = self.llm_client.model
                    ev.prompt_tokens = p_tokens // len(batch) if p_tokens else None
                    ev.completion_tokens = c_tokens // len(batch) if c_tokens else None
                    ev.total_tokens = t_tokens // len(batch) if t_tokens else None
                    ev.latency_sec = round(latency / len(batch), 2) if latency else None
                    ev.evaluated_at = now_iso
                    resp_map[ev.candidate_id] = ev

            batch_results: List[EvaluatedCandidate] = []
            for c in batch:
                if c.candidate_id in resp_map:
                    ev_res = resp_map[c.candidate_id]
                    record = EvaluatedCandidate(
                        candidate=c,
                        evaluation=ev_res,
                        status="evaluated",
                    )
                    record.compute_composite_score()
                else:
                    logger.error(f"Evaluation failed for candidate {c.candidate_id}: {err or 'Missing from batch response'}")
                    record = EvaluatedCandidate(
                        candidate=c,
                        evaluation=None,
                        status="evaluation_failed",
                        error_message=err or "Missing from batch response",
                        composite_score=0.0,
                    )

                self.append_checkpoint(record)
                results.append(record)
                batch_results.append(record)

            logger.info(
                f"[Batch {batch_num}/{total_batches}] Evaluated {len(batch)} candidates in {latency:.2f}s | "
                f"Tokens: {usage_dict.get('total_tokens', 'N/A')} | Simulated: {is_simulated}"
            )

            if progress_callback:
                progress_callback(
                    batch_num,
                    total_batches,
                    batch_results,
                    f"Batch {batch_num}/{total_batches} ({len(batch)} candidates) evaluated in {latency:.2f}s",
                )

            if getattr(self.llm_client, "_circuit_broken", False):
                logger.error("Circuit breaker tripped; aborting remaining evaluation batches.")
                break

            if batch_num < total_batches:
                time.sleep(0.5)

        return results
