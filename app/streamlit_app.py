"""Streamlit demo: score a credit-card client and inspect SHAP drivers."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_default.scoring import (  # noqa: E402
    explain_client,
    get_client_by_id,
    load_bundle,
    load_client_table,
    resolve_threshold,
    score_client_frame,
)

st.set_page_config(
    page_title="Credit Default Risk Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #102a43;
        --muted: #486581;
        --panel: #f0f4f8;
        --accent: #0b6e4f;
        --warn: #9b2226;
        --ok: #1d4e89;
      }
      .block-container { padding-top: 1.4rem; max-width: 1100px; }
      h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
      .hero {
        background: linear-gradient(135deg, #102a43 0%, #243b53 55%, #0b6e4f 100%);
        color: #f0f4f8;
        padding: 1.4rem 1.6rem;
        border-radius: 0;
        margin-bottom: 1.2rem;
      }
      .hero h1 { color: #f0f4f8; margin: 0 0 0.35rem 0; font-size: 1.8rem; }
      .hero p { color: #d9e2ec; margin: 0; max-width: 46rem; }
      .metric-note { color: var(--muted); font-size: 0.9rem; }
      div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _bundle():
    return load_bundle()


@st.cache_data(show_spinner=False)
def _clients() -> pd.DataFrame:
    return load_client_table()


def _decision_label(flag: int) -> str:
    return "Likely default" if int(flag) == 1 else "Likely non-default"


def main() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Credit Default Risk Demo</h1>
          <p>
            Select a client from the UCI Taiwan credit-card portfolio, score next-month
            default risk with the calibrated model, and inspect the SHAP drivers behind
            the decision.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        bundle = _bundle()
        clients = _clients()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Controls")
        policy = st.radio(
            "Decision policy",
            options=["cost", "f1"],
            format_func=lambda x: "Cost-sensitive (business)" if x == "cost" else "F1-optimized",
            index=0,
        )
        ids = clients["ID"].tolist()
        client_id = st.selectbox("Client ID", options=ids, index=0)
        show_profile = st.checkbox("Show raw client profile", value=True)
        st.markdown("---")
        st.caption(f"Model: **{bundle.get('name', 'unknown')}**")
        if bundle.get("calibration"):
            cal = bundle["calibration"]
            st.caption(
                "Brier "
                f"{cal.get('brier_before', float('nan')):.3f} → "
                f"{cal.get('brier_after', float('nan')):.3f}"
            )
        costs = bundle.get("cost_params") or {}
        if costs:
            st.caption(
                f"Costs — FP: {costs.get('fp_cost', 1)} · FN: {costs.get('fn_cost', 5)}"
            )

    row = get_client_by_id(clients, int(client_id))
    scored = score_client_frame(bundle, row, policy=policy).iloc[0]
    threshold = float(scored["threshold"])
    proba = float(scored["default_probability"])
    decision = int(scored["predicted_default"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Client ID", int(client_id))
    c2.metric("Default probability", f"{proba:.1%}")
    c3.metric("Threshold", f"{threshold:.3f}")
    c4.metric("Decision", _decision_label(decision))

    if decision == 1:
        st.warning(
            f"Flagged under the **{policy}** policy "
            f"(probability {proba:.1%} ≥ threshold {threshold:.3f})."
        )
    else:
        st.success(
            f"Not flagged under the **{policy}** policy "
            f"(probability {proba:.1%} < threshold {threshold:.3f})."
        )

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Risk drivers (local SHAP)")
        with st.spinner("Computing explanation..."):
            local = explain_client(bundle, row, top_k=10)
        plot_df = local.sort_values("shap_value")
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        colors = ["#9b2226" if v > 0 else "#1d4e89" for v in plot_df["shap_value"]]
        ax.barh(plot_df["feature"], plot_df["shap_value"], color=colors)
        ax.axvline(0, color="#829ab1", linewidth=1)
        ax.set_xlabel("SHAP value (→ higher default risk)")
        ax.set_title(f"Why client {int(client_id)} scored {proba:.1%}")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.dataframe(local.round(4), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Operating point")
        thr_cost = resolve_threshold(bundle, "cost")
        thr_f1 = resolve_threshold(bundle, "f1")
        st.markdown(
            f"""
            - Active policy: **{policy}**
            - Cost threshold: `{thr_cost:.4f}`
            - F1 threshold: `{thr_f1:.4f}`
            - Probability is **calibrated** (isotonic) when the saved artifact includes calibration.
            """
        )
        if show_profile:
            st.subheader("Client profile")
            profile_cols = [
                c
                for c in [
                    "LIMIT_BAL",
                    "SEX",
                    "EDUCATION",
                    "MARRIAGE",
                    "AGE",
                    "PAY_1",
                    "BILL_AMT1",
                    "PAY_AMT1",
                    "default",
                ]
                if c in row.columns
            ]
            st.dataframe(row[profile_cols].T.rename(columns={row.index[0]: "value"}), use_container_width=True)

        st.markdown(
            '<p class="metric-note">Red SHAP bars push risk up; blue bars push risk down. '
            "This demo uses the saved artifact in <code>artifacts/best_model.joblib</code>.</p>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
