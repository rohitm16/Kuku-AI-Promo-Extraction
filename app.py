"""Kuku AI Promo Discovery System - Executive Streamlit Web Application

Modular, interactive frontend for automated promotional clip extraction from multi-track screenplays.
Supports official OpenAI SDK integration, model-aware checkpointing,
and full deliverable exploration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
import streamlit as st

# Configure page setup - must be the first Streamlit command
st.set_page_config(
    page_title="Kuku AI Promo Discovery System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from kuku_promo_system.config import (
    DEFAULT_DATA_DIR,
    RAW_OUTPUTS_DIR,
    OPENAI_OUTPUT_DIR,
    TARGET_CANDIDATES_PER_EPISODE,
    mask_secret,
)
from kuku_promo_system.llm_client import MissingAPIKeyError, infer_provider
from ui.components import render_header, render_kpi_cards
from ui.model_settings import render_model_settings_sidebar
from ui.file_manager import ScreenplayFileManager
from ui.run_controller import estimate_pipeline_cost, execute_pipeline_ui
from ui.results_view import render_results_view
from ui.audit_view import render_audit_view
from ui.export_view import render_export_view


def main():
    # 1. Sidebar Configuration
    api_key, selected_model_id, max_tokens, base_url = render_model_settings_sidebar()

    # Sidebar System Info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ System Environment")
    builtin_files = ScreenplayFileManager.get_builtin_dataset_files()
    st.sidebar.caption(
        f"**Workspace:** `{Path.cwd().name}`<br>"
        f"**Built-in Episodes:** {len(builtin_files)} screenplays available<br>"
        f"**Active Engine:** Official `openai` SDK<br>"
        f"**Execution Mode:** Live ({selected_model_id})",
        unsafe_allow_html=True,
    )

    # 2. Executive Header
    render_header(
        active_provider=infer_provider(base_url),
        active_model=selected_model_id,
    )

    # 3. Main Navigation Tabs
    tab_overview, tab_runner, tab_results, tab_benchmark, tab_audit, tab_exports = st.tabs([
        "🏛️ System Overview",
        "🚀 Ingestion & Runner",
        "📊 Discovery Dashboard",
        "📜 Historical Benchmark",
        "🛡️ Quality Audit",
        "📥 Exports & Deliverables",
    ])

    # -------------------------------------------------------------
    # TAB 1: System Overview & Architecture
    # -------------------------------------------------------------
    with tab_overview:
        st.markdown("### Executive Summary: Automated Promotional Extraction")
        st.markdown(
            """
            The **Kuku AI Promo Discovery System** turns timestamped screenplays into
            **20-to-60 second promo candidates** for an editor to review.
            
            It combines deterministic parsing
            with LLM scoring to suggest hooks, cliffhangers,
            and self-contained conflicts.
            """
        )

        render_kpi_cards([
            ("Canonical Corpus", "50 Episodes", "Full Drama Season"),
            ("Shortlisted Candidates", "463 Windows", "3-Lane Pruner"),
            ("Duration Compliance", "100.0%", "Strict 20.0s - 60.0s"),
            ("Ranked Output Promos", "282 Promos", "Diversity-Adjusted"),
            ("List-Price Cost", "$0.0785 USD", "Not billed usage"),
        ])

        st.markdown("---")
        st.markdown("### 🔄 4-Stage Algorithmic Architecture")

        col_arch1, col_arch2 = st.columns([1.5, 1])

        with col_arch1:
            st.markdown(
                """
                1. **Multi-Track Screenplay Ingestion & Timeline Assembly:**
                   - Parses multi-track screenplay markers (`[CAMERA]`, `[ACTION]`, `[SFX]`, dialogue lines, and scene boundaries).
                   - Constructs a unified, continuous timeline converting timestamps to relative seconds.
                   - Groups synchronous atomic events into narrative **beats** to preserve dramatic cohesion.
                
                2. **Valid Window Generation & 3-Lane Pruning:**
                   - Generates every valid contiguous window strictly between **20.0s and 60.0s**.
                   - Evaluates candidates across three independent heuristic lanes:
                     - **Boundary:** Scores opening and closing beats using simple text cues.
                     - **Development Complexity:** Scores dialogue turns and action event density.
                     - **Keyword:** Counts high-stakes words and close-up shots.
                   - Reduces 8,730 windows to about 9-10 candidates per episode.
                
                3. **Cold-Viewer LLM Evaluation:**
                   - Evaluates batches of 3 candidates simultaneously using structured JSON schemas.
                   - Applies a **Cold-Viewer Persona**: No assumptions of external season lore or character backgrounds.
                   - Evaluates three primary axes:
                     - **Hook Strength (1-10):** Asks the model to judge the first 3 seconds. The prompt has no per-event timing.
                     - **Cliffhanger Potency (1-10):** High unresolved stakes. Verifies anti-climaxes via lookahead context.
                     - **Standalone Clarity (1-10):** Comprehensible without prior episode context.
                
                4. **Diversity-Aware Ranking & Deduplication:**
                   - Limits each episode to 1 promo in the top 10 and 2 in the top 25.
                   - Removes overlaps within an episode at IoU above 0.50.
                """
            )

        with col_arch2:
            st.markdown(
                """
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
                    <h4 style="margin-top:0; color: #1E3A8A;">Implemented Checks</h4>
                    <ul style="font-size: 0.9rem; line-height: 1.6; color: #334155;">
                        <li><b>Strict Duration Invariant:</b> 100% of generated promos adhere strictly to 20.0s - 60.0s.</li>
                        <li><b>Cold-Viewer Comprehension:</b> Penalizes heavy lore and inscrutable dialogue.</li>
                        <li><b>Lookahead:</b> Shows the model the next beat after the cut (median 5 seconds).</li>
                        <li><b>Model-Aware Checkpointing:</b> Checkpoint cache records model IDs, preventing cross-model contamination.</li>
                        <li><b>Safe Ingestion:</b> Comprehensive Zip-Slip path traversal protection on uploads.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------
    # TAB 2: Screenplay Ingestion & Pipeline Runner
    # -------------------------------------------------------------
    with tab_runner:
        st.markdown("### 🚀 Execute Promotional Extraction Pipeline")
        st.caption("Ingest screenplay files, review estimated costs, and execute end-to-end extraction.")

        # Ingestion source radio
        ingest_mode = st.radio(
            "Select Screenplay Ingestion Source",
            options=[
                "Built-in 50-Episode Dataset (episode-001 to episode-050)",
                "Upload Custom Screenplay Files (.txt)",
                "Upload ZIP Archive (.zip)",
            ],
            index=0,
            horizontal=True,
        )

        screenplay_files_to_run: List[Path] = []

        if ingest_mode == "Built-in 50-Episode Dataset (episode-001 to episode-050)":
            builtin_files = ScreenplayFileManager.get_builtin_dataset_files()
            if builtin_files:
                st.success(f"Loaded {len(builtin_files)} canonical screenplay files from `{DEFAULT_DATA_DIR}`.")
                screenplay_files_to_run = builtin_files
            else:
                st.error(f"Built-in screenplays not found at `{DEFAULT_DATA_DIR}`.")

        elif ingest_mode == "Upload Custom Screenplay Files (.txt)":
            uploaded_txts = st.file_uploader(
                "Upload one or more multi-track screenplay .txt files",
                type=["txt"],
                accept_multiple_files=True,
            )
            if uploaded_txts:
                upload_tmp_dir = Path("tmp_uploads")
                upload_tmp_dir.mkdir(exist_ok=True)
                valid_files = []
                for uf in uploaded_txts:
                    content = uf.read().decode("utf-8", errors="replace")
                    is_valid, err = ScreenplayFileManager.validate_screenplay_text(content, uf.name)
                    if is_valid:
                        dest = upload_tmp_dir / uf.name
                        dest.write_text(content, encoding="utf-8")
                        valid_files.append(dest)
                    else:
                        st.warning(f"Skipping {uf.name}: {err}")
                screenplay_files_to_run = valid_files
                st.info(f"Ingested {len(screenplay_files_to_run)} valid custom screenplay files.")

        elif ingest_mode == "Upload ZIP Archive (.zip)":
            uploaded_zip = st.file_uploader("Upload ZIP archive containing screenplays", type=["zip"])
            if uploaded_zip:
                zip_extract_dir = Path("tmp_zip_extracted")
                zip_extract_dir.mkdir(exist_ok=True)
                extracted, errors = ScreenplayFileManager.extract_zip_safely(
                    uploaded_zip.read(),
                    zip_extract_dir,
                )
                if errors:
                    for err in errors:
                        st.warning(err)
                if extracted:
                    st.success(f"Safely extracted {len(extracted)} valid screenplay files (Zip-Slip protected).")
                    screenplay_files_to_run = extracted

        # Run Configuration & Cost Estimation
        st.markdown("---")
        st.markdown("### 💰 Pre-Run Resource & Cost Estimation")

        num_episodes = len(screenplay_files_to_run)
        cands_per_ep = st.slider(
            "Shortlisted Candidates per Episode",
            min_value=5,
            max_value=15,
            value=TARGET_CANDIDATES_PER_EPISODE,
            help="Number of top candidates per episode to pass to LLM cold evaluation.",
        )

        cost_est = estimate_pipeline_cost(num_episodes, cands_per_ep, selected_model_id)

        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.metric("Total Episodes", str(num_episodes))
        with col_c2:
            st.metric("Candidate Windows", str(cost_est["candidate_count"]))
        with col_c3:
            st.metric("Estimated Tokens", f"~{cost_est['total_tokens']:,}")
        with col_c4:
            cost_str = f"~${cost_est['estimated_cost_usd']:.4f} USD"
            st.metric("Estimated Cost", cost_str)

        st.caption(
            f"Pricing Basis: `{selected_model_id}` &bull; "
            f"\\${cost_est['pricing_basis']['input_per_m']:.2f}/1M prompt &bull; "
            f"\\${cost_est['pricing_basis']['output_per_m']:.2f}/1M completion."
        )

        has_key = bool(api_key and api_key.strip() and "PASTE_" not in api_key and api_key not in ("your-api-key-here", "your_api_key_here"))

        # Launch Execution
        st.markdown("---")
        col_exec, col_warn = st.columns([1, 2])
        with col_exec:
            start_run = st.button(
                "🚀 Launch Pipeline Execution",
                type="primary",
                use_container_width=True,
                disabled=(num_episodes == 0 or not has_key),
            )

        with col_warn:
            if not has_key:
                st.error("An API key is required. This app does not simulate results.")

        if start_run and screenplay_files_to_run:
            st.markdown("### Execution Progress")
            with st.container():
                try:
                    diversity_promos, unrestricted_promos, evaluated_records, telemetry = execute_pipeline_ui(
                        screenplay_files=screenplay_files_to_run,
                        api_key=api_key,
                        model_id=selected_model_id,
                        max_tokens=max_tokens,
                        target_cands_per_ep=cands_per_ep,
                        output_dir=OPENAI_OUTPUT_DIR,
                        base_url=base_url,
                    )
                    st.cache_data.clear()
                    st.session_state["latest_run_promos"] = diversity_promos
                    st.session_state["latest_telemetry"] = telemetry
                    st.success(
                        f"🎉 Successfully extracted {len(diversity_promos)} ranked promotional clips across {num_episodes} episodes! "
                        "Navigate to the **Discovery Dashboard** or **Exports** tab to inspect and download."
                    )
                    failed_count = telemetry.get("evaluated_failed", 0)
                    if failed_count > 0:
                        st.warning(f"{failed_count} candidate evaluation(s) failed during this run.")
                except (MissingAPIKeyError, RuntimeError, ValueError) as e:
                    st.error(str(e))

    # -------------------------------------------------------------
    # TAB 3: Results Dashboard
    # -------------------------------------------------------------
    with tab_results:
        render_results_view(active_promos=st.session_state.get("latest_run_promos"))

    # -------------------------------------------------------------
    # TAB 4: Historical Benchmark Deep Dive
    # -------------------------------------------------------------
    with tab_benchmark:
        st.markdown("### 📜 Historical 50-Episode Benchmark (OpenRouter / Llama 3.3 70B)")
        st.caption("Permanent record of the production evaluation of 463 candidates conducted on OpenRouter.")

        st.info(
            "🔒 **Data Integrity & Provenance:** "
            "All 463 records in `raw_outputs/eval_checkpoint.jsonl` are 100% genuine live evaluations "
            "conducted with `meta-llama/llama-3.3-70b-instruct` at about $0.0785 list price. "
            "These records came from an earlier version of the code."
        )

        render_kpi_cards([
            ("Screenplays", "50 Episodes", "100% Ingested"),
            ("Shortlisted Candidates", "463 Candidates", "3-Lane Pruner"),
            ("Successful Live Evals", "463 / 463", "0% Failed / 0% Simulated"),
            ("Promos Ranked", "282 Promos", "Diversity Filter Applied"),
            ("Duration Compliance", "100.0%", "20.0s - 60.0s Bounds"),
            ("List-Price Cost", "$0.0785 USD", "Not billed usage"),
        ])

        st.markdown("#### Episode Candidate & Promo Distribution")
        st.markdown(
            """
            - **Episodes 1 to 50:** Evaluated systematically in batches of 3 candidates.
            - **Candidate Yield:** An average of 9.26 candidates per episode shortlisted from valid duration windows.
            - **Deduplication:** Overlapping windows with IoU above 0.50 were removed.
            - **Diversity Cap:** At most 1 promo per episode in the top 10 and 2 in the top 25.
            """
        )

    # -------------------------------------------------------------
    # TAB 5: Quality Audit
    # -------------------------------------------------------------
    with tab_audit:
        render_audit_view(
            api_key=api_key,
            model_id=selected_model_id,
            base_url=base_url,
        )

    # -------------------------------------------------------------
    # TAB 6: Exports & Deliverables
    # -------------------------------------------------------------
    with tab_exports:
        render_export_view()


if __name__ == "__main__":
    main()
