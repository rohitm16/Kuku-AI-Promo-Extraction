"""Shared UI components, typography, badges, and styling widgets for Streamlit."""

from __future__ import annotations

import streamlit as st
from typing import Optional


def inject_custom_css():
    """Inject custom executive styling for a clean, professional dashboard."""
    st.markdown(
        """
        <style>
        /* Modern Clean Typography & Layout */
        .main-title {
            font-size: 2.1rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.2rem;
        }
        .metric-container {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 14px 16px;
            text-align: center;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1E3A8A;
        }
        .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-top: 4px;
        }
        .badge-live {
            background-color: #DCFCE7;
            color: #166534;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.75rem;
        }
        .badge-model {
            background-color: #EFF6FF;
            color: #1E40AF;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.75rem;
            border: 1px solid #BFDBFE;
        }
        .promo-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .promo-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0F172A;
        }
        .score-pill {
            background-color: #3B82F6;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(active_provider: str = "OpenAI", active_model: str = "gpt-4o-mini"):
    """Render top-level executive header with active configuration badges."""
    inject_custom_css()
    model_badge = f'<span class="badge-model">{active_provider.upper()} &bull; {active_model}</span>'

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #E2E8F0; padding-bottom: 10px; margin-bottom: 18px;">
            <div>
                <div class="main-title">Kuku AI Promo Extraction System</div>
                <div class="sub-title">Automated Microdrama Screenplay Analysis & Promotional Clip Discovery</div>
            </div>
            <div style="text-align: right;">
                {model_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, help_text: Optional[str] = None):
    """Render a clean metric KPI box."""
    st.markdown(
        f"""
        <div class="metric-container" title="{help_text or ''}">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(metrics: list[tuple[str, str, Optional[str]]]):
    """Render a row of metric KPI boxes across equal-width columns."""
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, (label, val, help_t) in zip(cols, metrics):
        with col:
            render_metric_card(label=label, value=val, help_text=help_t)

