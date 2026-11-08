"""Interactive Results Dashboard for ranked promotional clips with filtering and deep-dive inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd

from kuku_promo_system.config import (
    RAW_OUTPUTS_DIR,
    OPENAI_OUTPUT_DIR,
    FINAL_RANKED_PROMOS_JSON,
    FINAL_RANKED_PROMOS_CSV,
    UNRESTRICTED_RANKED_PROMOS_JSON,
    UNRESTRICTED_RANKED_PROMOS_CSV,
)
from ui.components import render_kpi_cards


@st.cache_data(show_spinner=False)
def load_json_records(file_path: str) -> List[Dict[str, Any]]:
    """Cached loader for JSON records."""
    p = Path(file_path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        st.error(f"Failed to load JSON from {file_path}: {e}")
        return []


def get_available_datasets() -> Dict[str, Dict[str, Path]]:
    """Identify available output datasets in the workspace."""
    datasets = {}

    # Historical Benchmark (463 candidates OpenRouter run)
    if FINAL_RANKED_PROMOS_JSON.exists():
        datasets["Historical Benchmark (463 Candidates - OpenRouter Llama 3.3 70B)"] = {
            "diversity_json": FINAL_RANKED_PROMOS_JSON,
            "unrestricted_json": UNRESTRICTED_RANKED_PROMOS_JSON,
            "raw_evals": RAW_OUTPUTS_DIR / "raw_evaluations.json",
            "provider": "openrouter",
            "model": "meta-llama/llama-3.3-70b-instruct",
            "is_simulated": False,
        }

    # Active OpenAI Run (in raw_outputs/openai_run/)
    openai_diversity = OPENAI_OUTPUT_DIR / "ranked_promos.json"
    if openai_diversity.exists():
        datasets["Latest Run (raw_outputs/openai_run/)"] = {
            "diversity_json": openai_diversity,
            "unrestricted_json": OPENAI_OUTPUT_DIR / "ranked_promos_unrestricted.json",
            "raw_evals": OPENAI_OUTPUT_DIR / "raw_evaluations.json",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "is_simulated": False,
        }

    return datasets


def render_results_view(active_promos: Optional[List[Any]] = None):
    """Render the full promotional results inspection dashboard."""
    st.markdown("## 📊 Promotional Discovery & Ranking Dashboard")
    st.caption("Inspect, filter, and audit extracted promotional clips across all episodes.")

    datasets = get_available_datasets()

    # Session state dataset selection
    col_dset, col_rank_type = st.columns([2, 1])

    with col_dset:
        options = list(datasets.keys())
        if not options and not active_promos:
            st.info("No evaluation runs found yet. Run the pipeline from the **Pipeline Execution** tab to generate promos.")
            return

        dataset_choice = st.selectbox(
            "Select Evaluation Dataset to Inspect",
            options=options,
            index=0 if options else 0,
            help="Switch between the historical 463-candidate benchmark and recent runs.",
        )

    with col_rank_type:
        ranking_type = st.radio(
            "Ranking View",
            options=["Diversity-Adjusted", "Unrestricted Global"],
            horizontal=True,
            help="Diversity-Adjusted allows at most 1 promo per episode in the top 10 and 2 in the top 25. Unrestricted sorts by score only.",
        )

    # Load appropriate data
    selected_meta = datasets[dataset_choice]
    json_path = (
        str(selected_meta["diversity_json"])
        if ranking_type == "Diversity-Adjusted"
        else str(selected_meta["unrestricted_json"])
    )

    promos_data = load_json_records(json_path)

    if not promos_data:
        st.warning(f"No promotional records found in {json_path}.")
        return

    # Ingestion candidate texts cache for transcript viewer
    raw_evals_path = str(selected_meta["raw_evals"])
    raw_evals = load_json_records(raw_evals_path)
    cand_text_lookup = {}
    if raw_evals:
        for r in raw_evals:
            cand = r.get("candidate", {})
            cid = cand.get("candidate_id")
            if cid:
                cand_text_lookup[cid] = {
                    "window_text": cand.get("window_text", ""),
                    "lookahead_text": cand.get("lookahead_text", ""),
                    "start_sec": cand.get("start_sec", 0.0),
                    "end_sec": cand.get("end_sec", 0.0),
                }

    # Metadata & KPI Cards
    total_promos = len(promos_data)
    avg_score = round(sum(p.get("composite_score", 0.0) for p in promos_data) / max(total_promos, 1), 2)
    avg_duration = round(sum(p.get("duration_sec", 0.0) for p in promos_data) / max(total_promos, 1), 1)
    episodes_covered = len(set(p.get("episode_num") for p in promos_data))

    render_kpi_cards([
        ("Total Ranked Promos", str(total_promos), f"{ranking_type}"),
        ("Average Composite Score", f"{avg_score:.2f} / 10.0", "Quality Metric"),
        ("Average Duration", f"{avg_duration}s", "20.0s - 60.0s Window"),
        ("Episodes Covered", f"{episodes_covered} / 50", "Catalog Breadth"),
    ])

    st.markdown("---")

    # Filter Bar
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.2, 1.2, 1.5, 2.1])

    with f_col1:
        ep_options = ["All Episodes"] + sorted(list({f"Episode {p.get('episode_num')}" for p in promos_data}))
        selected_ep = st.selectbox("Episode", options=ep_options)

    with f_col2:
        hook_types = ["All Hook Types"] + sorted(list({p.get("hook", {}).get("type", "unknown") for p in promos_data}))
        selected_hook = st.selectbox("Hook Archetype", options=hook_types)

    with f_col3:
        min_score, max_score = st.slider(
            "Composite Score Range",
            min_value=0.0,
            max_value=10.0,
            value=(5.0, 10.0),
            step=0.1,
        )

    with f_col4:
        search_query = st.text_input(
            "Search keywords in rationale / transcript",
            placeholder="e.g. wedding, poison, Gerald, Juliana, betrayal...",
        ).lower().strip()

    # Apply Filters
    filtered_promos = []
    for p in promos_data:
        # Episode filter
        if selected_ep != "All Episodes":
            ep_num = int(selected_ep.replace("Episode ", ""))
            if p.get("episode_num") != ep_num:
                continue

        # Hook Type filter
        if selected_hook != "All Hook Types":
            if p.get("hook", {}).get("type") != selected_hook:
                continue

        # Score filter
        score = p.get("composite_score", 0.0)
        if not (min_score <= score <= max_score):
            continue

        # Text search
        if search_query:
            text_corpus = (
                f"{p.get('promo_id', '')} "
                f"{p.get('editorial_rationale', '')} "
                f"{p.get('hook', {}).get('description', '')} "
                f"{p.get('cliffhanger', {}).get('unresolved_question', '')} "
                f"{p.get('standalone', {}).get('conflict', '')}"
            ).lower()
            if search_query not in text_corpus:
                continue

        filtered_promos.append(p)

    st.markdown(f"**Showing {len(filtered_promos)} of {total_promos} promos matching criteria:**")

    # View layout toggle
    view_style = st.radio(
        "Display Style",
        options=["Executive Card View (Rationale & Subscores)", "Compact Table View"],
        horizontal=True,
    )

    if view_style == "Compact Table View":
        # Render clean Pandas DataFrame
        table_rows = []
        for p in filtered_promos:
            table_rows.append({
                "Rank": p.get("rank"),
                "Promo ID": p.get("promo_id"),
                "Episode": p.get("episode_num"),
                "Duration": f"{p.get('duration_sec', 0.0):.1f}s",
                "Start": p.get("start_time"),
                "End": p.get("end_time"),
                "Score": f"{p.get('composite_score', 0.0):.2f}",
                "Hook Type": p.get("hook", {}).get("type"),
                "Cliffhanger Question": p.get("cliffhanger", {}).get("unresolved_question", "")[:60] + "...",
                "Editorial Rationale": p.get("editorial_rationale", "")[:80] + "...",
            })
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    else:
        # Card View
        for p in filtered_promos[:50]:  # Limit page rendering to top 50 for smooth scrolling
            rank = p.get("rank")
            promo_id = p.get("promo_id")
            ep_num = p.get("episode_num")
            dur = p.get("duration_sec")
            score = p.get("composite_score")
            hook = p.get("hook", {})
            cliff = p.get("cliffhanger", {})
            standalone = p.get("standalone", {})
            rationale = p.get("editorial_rationale", "")
            telemetry = p.get("telemetry", {})

            with st.container():
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <div>
                                <span style="font-size: 1.15rem; font-weight: 700; color: #0F172A;">#{rank} &bull; {promo_id}</span>
                                <span style="margin-left: 8px; background: #EEF2F6; color: #334155; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">Episode {ep_num}</span>
                                <span style="margin-left: 6px; background: #F1F5F9; color: #475569; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">⏱️ {dur:.1f}s ({p.get('start_time')} - {p.get('end_time')})</span>
                            </div>
                            <div>
                                <span style="background-color: #2563EB; color: white; padding: 4px 12px; border-radius: 12px; font-weight: 700; font-size: 0.95rem;">
                                    ★ {score:.2f} / 10
                                </span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Subscore Details
                col_h, col_c, col_s = st.columns(3)
                with col_h:
                    st.markdown(f"**🪝 Hook ({hook.get('score', 0):.1f}/10)**")
                    st.caption(f"**Type:** `{hook.get('type')}`")
                    st.write(hook.get("description", "N/A"))

                with col_c:
                    st.markdown(f"**⚡ Cliffhanger ({cliff.get('score', 0):.1f}/10)**")
                    st.caption(f"**Type:** `{cliff.get('type')}` | **Resolved:** `{cliff.get('is_immediately_resolved')}`")
                    st.write(cliff.get("unresolved_question", "N/A"))

                with col_s:
                    st.markdown(f"**🎭 Standalone ({standalone.get('score', 0):.1f}/10)**")
                    st.caption(f"**Prior Lore Req:** `{standalone.get('requires_prior_lore')}`")
                    st.write(standalone.get("conflict", "N/A"))

                # Rationale Callout
                st.info(f"💡 **Editorial Rationale:** {rationale}")

                # Candidate Transcript Expander
                # Extract candidate ID from promo ID (e.g. PROMO_001_EP15_CAND_14_16 -> EP15_CAND_14_16)
                cand_key = promo_id.replace("PROMO_", "")
                if "_" in cand_key:
                    cand_key = "_".join(cand_key.split("_")[1:])  # EP15_CAND_14_16

                cand_info = cand_text_lookup.get(cand_key)

                with st.expander(f"📜 Inspect Screenplay Script Slice & Lookahead ({promo_id})"):
                    if cand_info and cand_info.get("window_text"):
                        st.markdown("**Screenplay Window Transcript:**")
                        st.code(cand_info["window_text"], language="text")
                        if cand_info.get("lookahead_text"):
                            st.markdown("**Lookahead Context (next beat after the cut):**")
                            st.code(cand_info["lookahead_text"], language="text")
                    else:
                        st.caption("Detailed script slice is available in full raw evaluations or raw screenplays.")

                    st.markdown(
                        f"""
                        <small>
                        <b>Telemetry:</b> Pruning Lane: <code>{telemetry.get('pruning_lane', 'N/A')}</code> |
                        Boundary Score: <code>{telemetry.get('boundary_score', 'N/A')}</code> |
                        Development Score: <code>{telemetry.get('development_score', 'N/A')}</code> |
                        Simulated: <code>{telemetry.get('is_simulated', 'N/A')}</code>
                        </small>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

        if len(filtered_promos) > 50:
            st.caption(f"Display capped at top 50 cards for UI performance. Switch to 'Compact Table View' or download full deliverables to inspect all {len(filtered_promos)} promos.")
