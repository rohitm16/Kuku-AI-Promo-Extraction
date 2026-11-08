"""Export and Deliverables management view providing one-click file downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st

from kuku_promo_system.config import (
    RAW_OUTPUTS_DIR,
    OPENAI_OUTPUT_DIR,
    FINAL_RANKED_PROMOS_JSON,
    FINAL_RANKED_PROMOS_CSV,
    UNRESTRICTED_RANKED_PROMOS_JSON,
    UNRESTRICTED_RANKED_PROMOS_CSV,
    FINAL_RAW_EVALUATIONS_JSON,
    DEFAULT_CHECKPOINT_FILE,
    FINAL_TELEMETRY_FILE,
)


def render_download_row(label: str, file_path: Path, mime_type: str, description: str):
    """Render a single artifact download row with file existence verification."""
    col_info, col_btn = st.columns([3, 1])

    with col_info:
        st.markdown(f"**{label}** &bull; `{file_path.name}`")
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024.0
            size_str = f"{size_kb / 1024.0:.2f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
            st.caption(f"{description} (File size: {size_str})")
        else:
            st.caption(f"{description} *(File not yet generated)*")

    with col_btn:
        if file_path.exists():
            with open(file_path, "rb") as f:
                bytes_data = f.read()
            st.download_button(
                label=f"⬇️ Download {file_path.suffix.upper()}",
                data=bytes_data,
                file_name=file_path.name,
                mime=mime_type,
                use_container_width=True,
            )
        else:
            st.button("Not Found", disabled=True, use_container_width=True, key=f"btn_disabled_{file_path.name}")

    st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)


def render_export_view():
    """Render the Export & Deliverables tab."""
    st.markdown("## 📥 Export Pipeline Deliverables")
    st.caption("Download ranked promotional exports, raw LLM evaluations, checkpoint logs, and diagnostic telemetry.")

    export_source = st.radio(
        "Select Deliverables Source",
        options=[
            "Historical Benchmark (463 Candidates - OpenRouter/Llama 3.3 70B)",
            "Latest Pipeline Run (raw_outputs/openai_run/)",
        ],
        index=0,
        horizontal=True,
    )

    if export_source.startswith("Historical"):
        base_dir = RAW_OUTPUTS_DIR
        st.info(
            "📦 **Historical Benchmark Deliverables:** Generated from the 50-episode run evaluating 463 shortlisted candidates."
        )
    else:
        base_dir = OPENAI_OUTPUT_DIR
        st.info(
            f"📦 **Latest Run Deliverables:** Routed cleanly to `{OPENAI_OUTPUT_DIR}` to preserve historical provenance."
        )

    st.markdown("### 1. Primary Ranked Promotional Deliverables")
    render_download_row(
        label="Diversity-Adjusted Ranked Promos (JSON)",
        file_path=base_dir / "ranked_promos.json",
        mime_type="application/json",
        description="Ranked promos with at most 1 per episode in the top 10 and 2 in the top 25.",
    )

    render_download_row(
        label="Diversity-Adjusted Ranked Promos (CSV)",
        file_path=base_dir / "ranked_promos.csv",
        mime_type="text/csv",
        description="Spreadsheet-ready CSV table of diversity-adjusted ranked promos with timings and subscores.",
    )

    render_download_row(
        label="Unrestricted Global Ranked Promos (JSON)",
        file_path=base_dir / "ranked_promos_unrestricted.json",
        mime_type="application/json",
        description="Greedy global ranking without per-episode caps, preserving high-density episodes.",
    )

    render_download_row(
        label="Unrestricted Global Ranked Promos (CSV)",
        file_path=base_dir / "ranked_promos_unrestricted.csv",
        mime_type="text/csv",
        description="Spreadsheet-ready CSV table of unrestricted rankings.",
    )

    st.markdown("### 2. Comprehensive Cold-Viewer LLM Evaluation Logs")
    render_download_row(
        label="Full Raw Evaluations (JSON)",
        file_path=base_dir / "raw_evaluations.json",
        mime_type="application/json",
        description="Complete candidate transcripts, lookahead contexts, LLM reasoning, subscores, and rejection flags.",
    )

    render_download_row(
        label="Evaluation Checkpoint Stream (JSONL)",
        file_path=base_dir / "eval_checkpoint.jsonl",
        mime_type="application/x-ndjson",
        description="Line-by-line streaming append-only checkpoint log for failure recovery and incremental resumption.",
    )

    st.markdown("### 3. Quality Assurance & Audit Telemetry")
    render_download_row(
        label="Pipeline Telemetry & Run Diagnostics (JSON)",
        file_path=base_dir / "phase10_final_telemetry.json" if (base_dir / "phase10_final_telemetry.json").exists() else FINAL_TELEMETRY_FILE,
        mime_type="application/json",
        description="System metrics, event/beat counts, runtime duration, candidate distributions, and model metadata.",
    )
