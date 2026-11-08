"""Automated 12-case benchmark evaluation suite and failure analysis engine.

Evaluates pipeline performance on curated test cases representing:
- 4 Clear Ground-Truth Hits
- 2 Weak / Mundane Setups
- 2 Lore-Heavy / Cold-Viewer Failure Traps
- 2 Artificial Cliffhangers (Immediately Deflated)
- 2 Subtle / Quiet High-Stakes Moments
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel

from kuku_promo_system.config import DEFAULT_OUTPUT_DIR, validate_duration
from kuku_promo_system.logger import logger
from kuku_promo_system.pruner import CandidateWindow
from kuku_promo_system.evaluator import (
    SinglePromoEvaluation,
    EvaluatedCandidate,
    PromoEvaluator,
)
from kuku_promo_system.llm_client import LLMClientAdapter


class BenchmarkCase(BaseModel):
    case_id: str
    category: str  # "ground_truth_hit" | "mundane_setup" | "lore_heavy" | "artificial_cliffhanger" | "subtle_moment"
    description: str
    candidate: CandidateWindow
    expected_outcome: str  # "high_rank" | "low_rank" | "lore_flagged" | "lookahead_flagged"


def create_12_case_benchmark() -> List[BenchmarkCase]:
    """Assemble the 12 curated benchmark test cases."""
    cases: List[BenchmarkCase] = []

    # 1. Ground Truth Hit: Altar Abandonment (Ep 1)
    cases.append(BenchmarkCase(
        case_id="HIT_01_ALTAR_BETRAYAL",
        category="ground_truth_hit",
        description="Matriarch stops Juliana at cathedral doors; Gerald abandons her at altar.",
        candidate=CandidateWindow(
            candidate_id="BENCH_HIT_01",
            episode_num=1,
            scene_start=1,
            scene_end=1,
            start_beat_id=1,
            end_beat_id=5,
            start_sec=11.30,
            end_sec=43.90,
            start_str="00:11.30",
            end_str="00:43.90",
            duration_sec=32.60,
            lane="boundary",
            window_text="MATRIARCH (Cold, Commanding): \"You should turn around, Juliana.\"\nLADY JULIANA: \"Why? Gerald is waiting for me.\"\nMATRIARCH: \"Not anymore.\"\n[ACTION] Juliana walks down the nave in shock.\nLADY IN GOLD GOWN: \"Because he is marrying me.\"",
            lookahead_text="LADY JULIANA (Trembling): \"No... this cannot be true.\"",
        ),
        expected_outcome="high_rank",
    ))

    # 2. Ground Truth Hit: Gerald Disowning Juliana
    cases.append(BenchmarkCase(
        case_id="HIT_02_GERALD_DISMISSAL",
        category="ground_truth_hit",
        description="Gerald announces Juliana's low value while holding Clarissa's hand.",
        candidate=CandidateWindow(
            candidate_id="BENCH_HIT_02",
            episode_num=1,
            scene_start=1,
            scene_end=1,
            start_beat_id=15,
            end_beat_id=20,
            start_sec=124.0,
            end_sec=138.0,
            start_str="02:04.00",
            end_str="02:18.00",
            duration_sec=24.00,
            lane="boundary",
            window_text="GERALD (Arrogant, Dismissive): \"I cannot marry a woman of such plainness and low value when Clarissa is the true jewel of the county.\"\n[ACTION] Juliana's hand clenches fiercely, knuckles whitening with suppressed fury.\n[CAMERA] Push-in on Juliana's tearful blue eyes hardening into cold, resolute defiance.\n[AUDIO] Solemn low-pitched string drone swells to an ominous cliffhanger peak.\n[CAMERA] Cut to black.",
            lookahead_text="",
        ),
        expected_outcome="high_rank",
    ))

    # 3. Ground Truth Hit: Arrest Confrontation (Ep 12)
    cases.append(BenchmarkCase(
        case_id="HIT_03_ARREST_WARRANT",
        category="ground_truth_hit",
        description="Constable confronts Alexander with theft warrant before Duke reveals authority.",
        candidate=CandidateWindow(
            candidate_id="BENCH_HIT_03",
            episode_num=12,
            scene_start=1,
            scene_end=1,
            start_beat_id=2,
            end_beat_id=8,
            start_sec=15.0,
            end_sec=48.0,
            start_str="00:15.00",
            end_str="00:48.00",
            duration_sec=33.00,
            lane="boundary",
            window_text="CONSTABLE (Gruff, Aggressive): \"By order of the magistrate, you are under arrest for grand theft.\"\nCLARISSA (Sneering): \"He is nothing but a crippled soldier with stolen coins.\"\nALEXANDER (Cold, Razor-sharp): \"Look closely at the crest on this ring, Constable. Dare you lay a hand upon me?\"",
            lookahead_text="CONSTABLE (Pale, Stuttering): \"Y-Your Grace... forgive me.\"",
        ),
        expected_outcome="high_rank",
    ))

    # 4. Ground Truth Hit: Duke Banishing Rivals
    cases.append(BenchmarkCase(
        case_id="HIT_04_DUKE_PROMISE",
        category="ground_truth_hit",
        description="Alexander promises retribution against Juliana's abusive family.",
        candidate=CandidateWindow(
            candidate_id="BENCH_HIT_04",
            episode_num=3,
            scene_start=2,
            scene_end=2,
            start_beat_id=4,
            end_beat_id=10,
            start_sec=30.0,
            end_sec=65.0,
            start_str="00:30.00",
            end_str="01:05.00",
            duration_sec=35.00,
            lane="development",
            window_text="[ACTION] Alexander stands between the guards and Juliana, eyes blazing.\nALEXANDER: \"Touch her again, and you answer to the Crown.\"\nJULIANA: \"Why are you defending someone everyone else discarded?\"\nALEXANDER: \"Because you are now my Duchess.\"",
            lookahead_text="GUARDS (Retreating in panic): \"Stand down!\"",
        ),
        expected_outcome="high_rank",
    ))

    # 5. Mundane Setup: Carriage Seating Conversation
    cases.append(BenchmarkCase(
        case_id="MUNDANE_01_CARRIAGE_RIDE",
        category="mundane_setup",
        description="Routine polite travel conversation with zero dramatic stakes.",
        candidate=CandidateWindow(
            candidate_id="BENCH_MUNDANE_01",
            episode_num=5,
            scene_start=1,
            scene_end=1,
            start_beat_id=1,
            end_beat_id=4,
            start_sec=5.0,
            end_sec=32.0,
            start_str="00:05.00",
            end_str="00:32.00",
            duration_sec=27.00,
            lane="boundary",
            window_text="[CAMERA] Exterior carriage moving down a cobblestone road.\nJULIANA (Calm): \"The afternoon breeze is quite pleasant.\"\nMAID: \"Yes, my lady. The coachman believes we shall arrive before nightfall.\"\nJULIANA: \"That is reassuring.\"",
            lookahead_text="[CAMERA] Carriage rolls past town square.",
        ),
        expected_outcome="low_rank",
    ))

    # 6. Mundane Setup: Tea Service Logistics
    cases.append(BenchmarkCase(
        case_id="MUNDANE_02_TEA_SERVICE",
        category="mundane_setup",
        description="Polite household servant instructions.",
        candidate=CandidateWindow(
            candidate_id="BENCH_MUNDANE_02",
            episode_num=6,
            scene_start=1,
            scene_end=1,
            start_beat_id=1,
            end_beat_id=4,
            start_sec=10.0,
            end_sec=35.0,
            start_str="00:10.00",
            end_str="00:35.00",
            duration_sec=25.00,
            lane="boundary",
            window_text="BUTLER: \"The silver has been polished as requested, my lord.\"\nLORD: \"Very good, Jenkins. Ensure the library fireplace is lit at six.\"\nBUTLER: \"Immediately, sir.\"",
            lookahead_text="BUTLER exits quietly.",
        ),
        expected_outcome="low_rank",
    ))

    # 7. Lore-Heavy Failure Trap: Obscure Lineage Debate
    cases.append(BenchmarkCase(
        case_id="LORE_01_OBSCURE_DEED",
        category="lore_heavy",
        description="Argument referring to unseen third-party contracts from 15 episodes ago.",
        candidate=CandidateWindow(
            candidate_id="BENCH_LORE_01",
            episode_num=18,
            scene_start=1,
            scene_end=1,
            start_beat_id=3,
            end_beat_id=7,
            start_sec=20.0,
            end_sec=50.0,
            start_str="00:20.00",
            end_str="00:50.00",
            duration_sec=30.00,
            lane="boundary",
            window_text="SOLICITOR: \"According to Clause 4 of the 1812 codicil signed in Sussex, Lady Margaret's third cousin retains the reversionary lease over Blackwood.\"\nPRUDENCE: \"But what of Uncle Harold's codicil in Kent?\"\nSOLICITOR: \"Voided upon probate in York.\"",
            lookahead_text="SOLICITOR adjusts spectacles.",
        ),
        expected_outcome="lore_flagged",
    ))

    # 8. Lore-Heavy Failure Trap: Unnamed Cousin Mention
    cases.append(BenchmarkCase(
        case_id="LORE_02_FAMILY_TREE",
        category="lore_heavy",
        description="Dialogue naming characters never introduced on screen.",
        candidate=CandidateWindow(
            candidate_id="BENCH_LORE_02",
            episode_num=22,
            scene_start=1,
            scene_end=1,
            start_beat_id=2,
            end_beat_id=6,
            start_sec=15.0,
            end_sec=42.0,
            start_str="00:15.00",
            end_str="00:42.00",
            duration_sec=27.00,
            lane="boundary",
            window_text="NOBLEMAN: \"If Eustace hears that Beatrice gave the ledger to Barnaby, all is lost.\"\nLADY: \"Barnaby would never betray Eustace's nephew.\"",
            lookahead_text="NOBLEMAN: \"We can only pray.\"",
        ),
        expected_outcome="lore_flagged",
    ))

    # 9. Artificial Cliffhanger: Banal Immediate Answer
    cases.append(BenchmarkCase(
        case_id="FAKE_CLIFF_01_BANAL_ANSWER",
        category="artificial_cliffhanger",
        description="Accusation cut prematurely, but next beat immediately deflates tension.",
        candidate=CandidateWindow(
            candidate_id="BENCH_FAKE_CLIFF_01",
            episode_num=8,
            scene_start=1,
            scene_end=1,
            start_beat_id=3,
            end_beat_id=7,
            start_sec=12.0,
            end_sec=38.0,
            start_str="00:12.00",
            end_str="00:38.00",
            duration_sec=26.00,
            lane="boundary",
            window_text="JULIANA (Furious): \"Where did you go last night, Gerald?! Whose carriage was outside the gates?\"",
            lookahead_text="GERALD (Sighing, Flat): \"The physician's carriage. Mother had a coughing fit and I fetched his tonic.\"",
        ),
        expected_outcome="lookahead_flagged",
    ))

    # 10. Artificial Cliffhanger: Accidental Document Drop
    cases.append(BenchmarkCase(
        case_id="FAKE_CLIFF_02_FALSE_STAKES",
        category="artificial_cliffhanger",
        description="Dramatic gasp cut before mundane clarification.",
        candidate=CandidateWindow(
            candidate_id="BENCH_FAKE_CLIFF_02",
            episode_num=14,
            scene_start=1,
            scene_end=1,
            start_beat_id=5,
            end_beat_id=9,
            start_sec=25.0,
            end_sec=51.0,
            start_str="00:25.00",
            end_str="00:51.00",
            duration_sec=26.00,
            lane="boundary",
            window_text="[ACTION] Clarissa drops the sealed envelope, gasping in horror.\nCLARISSA: \"The Duke... he knows everything!\"",
            lookahead_text="CLARISSA (Examining envelope closer): \"Wait... this is merely an invitation to the seasonal garden ball.\"",
        ),
        expected_outcome="lookahead_flagged",
    ))

    # 11. Subtle High Stakes: Whispered Coercion
    cases.append(BenchmarkCase(
        case_id="SUBTLE_01_WHISPERED_THREAT",
        category="subtle_moment",
        description="Quiet psychological intimidation without raised voices or fast camera cuts.",
        candidate=CandidateWindow(
            candidate_id="BENCH_SUBTLE_01",
            episode_num=25,
            scene_start=1,
            scene_end=1,
            start_beat_id=8,
            end_beat_id=14,
            start_sec=40.0,
            end_sec=72.0,
            start_str="00:40.00",
            end_str="01:12.00",
            duration_sec=32.00,
            lane="anomaly",
            window_text="[CAMERA] Extreme close-up on Matriarch leaning into Juliana's ear.\nMATRIARCH (Quiet, Merciless whisper): \"Sign the renunciation, Juliana. Or your mother's grave will be plowed into the sheep pasture before Sunday.\"\n[ACTION] Juliana freezes, breath catching in her throat, tears frozen in disbelief.\nMATRIARCH: \"Choose now.\"",
            lookahead_text="JULIANA (Dignified): \"Plow it. Her soul is with God, not under your boots.\"",
        ),
        expected_outcome="high_rank",
    ))

    # 12. Subtle High Stakes: Silent Ring Exchange
    cases.append(BenchmarkCase(
        case_id="SUBTLE_02_RING_REMOVAL",
        category="subtle_moment",
        description="Quiet visual defiance where a betrothal ring is dropped into wine.",
        candidate=CandidateWindow(
            candidate_id="BENCH_SUBTLE_02",
            episode_num=2,
            scene_start=1,
            scene_end=1,
            start_beat_id=6,
            end_beat_id=11,
            start_sec=30.0,
            end_sec=58.0,
            start_str="00:30.00",
            end_str="00:58.00",
            duration_sec=28.00,
            lane="anomaly",
            window_text="[CAMERA] Tight insert on Juliana's trembling fingers slowly sliding Gerald's diamond ring from her finger.\nJULIANA (Composed, Chillingly calm): \"Keep your county, Gerald. And keep my sister.\"\n[ACTION] Juliana drops the ring into his goblet of red wine with a dull splash.\n[CAMERA] Close-up on Gerald's stunned face as Juliana turns her back.",
            lookahead_text="GERALD (Whispering): \"Juliana... wait.\"",
        ),
        expected_outcome="high_rank",
    ))

    return cases


def run_benchmark_audit(evaluator: PromoEvaluator) -> Dict[str, Any]:
    """Execute evaluation on all 12 benchmark cases and produce a diagnostic report."""
    benchmark_cases = create_12_case_benchmark()
    candidates = [b.candidate for b in benchmark_cases]

    logger.info(f"Running automated Quality Audit on {len(candidates)} benchmark cases...")
    evaluated_records = evaluator.evaluate_candidates(candidates)

    report: Dict[str, Any] = {
        "model_id": evaluator.llm_client.model,
        "is_live": True,
        "total_cases": len(benchmark_cases),
        "duration_compliance_rate": 0.0,
        "ground_truth_recall_rate": 0.0,
        "lore_heavy_rejection_rate": 0.0,
        "lookahead_catch_rate": 0.0,
        "subtle_moment_recall_rate": 0.0,
        "diagnostics": [],
    }

    # Evaluate compliance metrics
    valid_durations = sum(1 for c in candidates if validate_duration(c.duration_sec))
    report["duration_compliance_rate"] = round((valid_durations / len(candidates)) * 100.0, 1)

    eval_by_id = {r.candidate.candidate_id: r for r in evaluated_records}

    hits_passed = 0
    lore_penalized = 0
    lookahead_caught = 0
    subtle_passed = 0

    for b in benchmark_cases:
        rec = eval_by_id.get(b.candidate.candidate_id)
        if not rec or not rec.evaluation:
            report["diagnostics"].append({
                "case_id": b.case_id,
                "status": "EVAL_FAILED",
                "notes": "Candidate failed evaluation or was missing from output",
            })
            continue

        ev = rec.evaluation
        score = rec.composite_score
        notes = f"Score: {score:.2f} (Hook: {ev.hook_score}, Cliff: {ev.cliffhanger_score}, Clarity: {ev.standalone_clarity_score})"

        if b.category == "ground_truth_hit":
            if score >= 7.0 and not ev.requires_prior_lore:
                hits_passed += 1
                status = "PASS"
            else:
                status = "FAIL"
                notes += " -> Expected high rank (>=7.0)"

        elif b.category == "mundane_setup":
            if score < 6.5:
                status = "PASS"
            else:
                status = "FAIL"
                notes += " -> Mundane clip scored too high"

        elif b.category == "lore_heavy":
            if ev.requires_prior_lore or ev.standalone_clarity_score <= 5.5:
                lore_penalized += 1
                status = "PASS"
            else:
                status = "FAIL"
                notes += " -> Lore dependency was not detected"

        elif b.category == "artificial_cliffhanger":
            if ev.is_immediately_resolved:
                lookahead_caught += 1
                status = "PASS"
            else:
                status = "FAIL"
                notes += " -> Artificial cliffhanger was not caught by lookahead"

        elif b.category == "subtle_moment":
            if score >= 6.5:
                subtle_passed += 1
                status = "PASS"
            else:
                status = "FAIL"
                notes += " -> Subtle high-stakes moment was penalized too severely"

        report["diagnostics"].append({
            "case_id": b.case_id,
            "category": b.category,
            "status": status,
            "notes": notes,
        })

    report["ground_truth_recall_rate"] = round((hits_passed / 4.0) * 100.0, 1)
    report["lore_heavy_rejection_rate"] = round((lore_penalized / 2.0) * 100.0, 1)
    report["lookahead_catch_rate"] = round((lookahead_caught / 2.0) * 100.0, 1)
    report["subtle_moment_recall_rate"] = round((subtle_passed / 2.0) * 100.0, 1)

    logger.info("Quality Audit Complete:")
    logger.info(f"  Duration Compliance:  {report['duration_compliance_rate']}%")
    logger.info(f"  Ground-Truth Recall:  {report['ground_truth_recall_rate']}%")
    logger.info(f"  Lore-Heavy Catch:     {report['lore_heavy_rejection_rate']}%")
    logger.info(f"  Lookahead Catch Rate: {report['lookahead_catch_rate']}%")
    logger.info(f"  Subtle Beat Recall:   {report['subtle_moment_recall_rate']}%")

    return report
