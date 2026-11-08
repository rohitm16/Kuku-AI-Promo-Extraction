"""Quality Audit and 12-Case Benchmark view with synthetic case inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd

from kuku_promo_system.llm_client import LLMClientAdapter
from kuku_promo_system.evaluator import PromoEvaluator
from kuku_promo_system.audit import run_benchmark_audit
from ui.components import render_kpi_cards


def render_audit_view(
    api_key: Optional[str] = None,
    model_id: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
):
    """Render the Quality Audit & 12-Case Benchmark inspector."""
    st.markdown("## 🛡️ Algorithmic Quality Audit & 12-Case Benchmark")
    st.caption("Verifying cold-viewer comprehension, duration compliance, and adversarial robustness.")

    # Mandatory Synthetic Disclosure Banner
    st.warning("These 12 cases are hand-written test fixtures. The earlier 100% result came from a simulator and is invalid. Results appear here only after a live model run.")

    has_key = bool(api_key and api_key.strip() and "PASTE_" not in api_key and api_key not in ("your-api-key-here", "your_api_key_here"))

    report_data = st.session_state.get("audit_report_data")

    # Control bar to re-run audit
    col_stat, col_btn = st.columns([3, 1])
    with col_stat:
        if report_data:
            st.markdown(f"**Live Session Audit:** Model `{report_data.get('model_id', model_id)}` (12/12 Cases Audited)")
        else:
            st.info("No live benchmark audit run executed in this session. Click 'Re-run Quality Audit' below to execute.")

    with col_btn:
        run_now = st.button("🧪 Re-run Quality Audit", use_container_width=True, disabled=not has_key)

    if run_now:
        with st.spinner("Executing 12-case benchmark evaluation suite..."):
            adapter = LLMClientAdapter(
                api_key=api_key,
                model=model_id,
                base_url=base_url or None,
            )
            evaluator = PromoEvaluator(llm_client=adapter)
            report_data = run_benchmark_audit(evaluator)
            st.session_state["audit_report_data"] = report_data
            st.success("Quality audit complete!")

    if not report_data:
        return

    # Render KPI Cards
    render_kpi_cards([
        ("Duration Compliance", f"{report_data.get('duration_compliance_rate', 0.0):.1f}%", "20.0s - 60.0s Rule"),
        ("Ground Truth Recall", f"{report_data.get('ground_truth_recall_rate', 0.0):.1f}%", "High-Stakes Climax"),
        ("Lore-Trap Rejection", f"{report_data.get('lore_heavy_rejection_rate', 0.0):.1f}%", "Inscrutable Setup Filter"),
        ("Fake Cliff Catch", f"{report_data.get('lookahead_catch_rate', 0.0):.1f}%", "Immediate Resolution Check"),
        ("Subtle Tension Recall", f"{report_data.get('subtle_moment_recall_rate', 0.0):.1f}%", "Micro-Moment Sensitivity"),
    ])

    st.markdown("---")
    st.markdown("### 📋 Detailed Benchmark Case Diagnostics")

    diagnostics = report_data.get("diagnostics", [])
    if diagnostics:
        rows = []
        for d in diagnostics:
            rows.append({
                "Case Identifier": d.get("case_id"),
                "Stress-Test Category": d.get("category", "").replace("_", " ").title(),
                "Status": d.get("status"),
                "Evaluator Score & Diagnostics": d.get("notes"),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("### 🔍 Category Deep-Dive")
        col_cat1, col_cat2 = st.columns(2)

        with col_cat1:
            with st.expander("🎯 Ground Truth Hits (4 Cases) - High-Stakes Climaxes"):
                st.write(
                    "High-drama, emotionally charged moments that are expected to score 7.0 or higher. "
                    "Tests whether high-stakes narrative turnpoints (e.g., altar abandonment, public arrest) "
                    "are reliably identified by the LLM cold viewer."
                )
                for d in diagnostics:
                    if d.get("category") == "ground_truth_hit":
                        st.markdown(f"- **`{d['case_id']}`**: `{d['status']}` &bull; {d['notes']}")

            with st.expander("🚫 Lore-Heavy Traps (2 Cases) - Inscrutable Exposition"):
                st.write(
                    "Expositional dialogues discussing fictional kingdoms, obscure genealogies, or boundary disputes. "
                    "Tests the Standalone Clarity rubric: cold viewers without prior episode knowledge "
                    "should find these confusing, triggering heavy score penalties."
                )
                for d in diagnostics:
                    if d.get("category") == "lore_heavy":
                        st.markdown(f"- **`{d['case_id']}`**: `{d['status']}` &bull; {d['notes']}")

        with col_cat2:
            with st.expander("⏳ Fake Cliffhangers (2 Cases) - Immediate Resolution"):
                st.write(
                    "Windows that end on what appears to be a cliffhanger, but where the lookahead window "
                    "reveals an immediate banal answer or anticlimax within 10 seconds. "
                    "The pipeline multiplies these scores by 0.70."
                )
                for d in diagnostics:
                    if d.get("category") == "artificial_cliffhanger":
                        st.markdown(f"- **`{d['case_id']}`**: `{d['status']}` &bull; {d['notes']}")

            with st.expander("🤫 Subtle Micro-Moments (2 Cases) - Quiet Dramatic Tension"):
                st.write(
                    "Moments lacking loud yelling or violent actions, but containing intense psychological tension "
                    "(e.g., quiet whispered threats, removing a wedding ring). Tests that the prompt does not "
                    "blindly over-weight sound effects or overt combat."
                )
                for d in diagnostics:
                    if d.get("category") == "subtle_moment":
                        st.markdown(f"- **`{d['case_id']}`**: `{d['status']}` &bull; {d['notes']}")
