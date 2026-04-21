"""
dashboards/streamlit_app.py
─────────────────────────────
Interactive Streamlit dashboard for the Telecom Churn Risk platform.

Tabs:
  1. 📊 Overview       — KPI cards, churn distribution, key metrics
  2. 🗺️  Geo Risk Map   — H3 hex risk map (Folium)
  3. 🔍 Subscriber      — Single subscriber lookup & real-time scoring
  4. 📈 Model Analysis  — Feature importance, SHAP, performance curves
  5. ⚠️  Alerts          — High-risk H3 cells and revenue at risk

Usage:
    streamlit run dashboards/streamlit_app.py
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
import joblib
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load config ───────────────────────────────────────────────────────────────
@st.cache_data
def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_processed_data():
    path = Path("data/processed/features_full.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_risk_grid():
    path = Path("data/processed/risk_grid.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_resource
def load_model():
    paths = [
        "data/models/churn_model_xgboost.pkl",
        "data/models/churn_model_lightgbm.pkl",
    ]
    for p in paths:
        if Path(p).exists():
            return joblib.load(p)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/cell-tower.png", width=60)
    st.title("Churn Intelligence")
    st.markdown("---")

    cfg = load_config()
    df = load_processed_data()
    risk_grid = load_risk_grid()
    model_art = load_model()

    if df is None:
        st.error("⚠️ No processed data found.\nRun the pipeline first:\n```\npython src/features/feature_pipeline.py\n```")
        st.stop()

    st.success(f"✅ {len(df):,} subscribers loaded")

    if model_art:
        st.success("✅ Model loaded")
    else:
        st.warning("⚠️ No model found.\nRun: `python src/models/train.py`")

    st.markdown("---")
    st.caption("Portfolio Project\nGeospatial × Telecom × ML")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🗺️ Geo Risk Map", "🔍 Subscriber", "📈 Model Analysis", "⚠️ Alerts"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Churn Overview Dashboard")

    # KPI cards
    total = len(df)
    churned = df["churned"].sum()
    churn_rate = churned / total
    avg_revenue = df["monthly_charges"].mean()
    revenue_at_risk = df.loc[df["churned"] == 1, "monthly_charges"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Subscribers", f"{total:,}")
    col2.metric("Churn Rate", f"{churn_rate:.1%}", delta=f"{churn_rate - 0.18:.1%} vs benchmark")
    col3.metric("Avg Monthly Charges", f"${avg_revenue:.0f}")
    col4.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    # Churn by contract type
    with col_a:
        ct = df.groupby("contract_type")["churned"].agg(["sum", "count"])
        ct["rate"] = ct["sum"] / ct["count"]
        fig = px.bar(
            ct.reset_index(), x="contract_type", y="rate",
            title="Churn Rate by Contract Type",
            labels={"rate": "Churn Rate", "contract_type": "Contract Type"},
            color="rate", color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Churn by tenure bucket
    with col_b:
        if "tenure_bucket" in df.columns:
            tb = df.groupby("tenure_bucket")["churned"].mean().reset_index()
            tb.columns = ["Tenure", "Churn Rate"]
            fig2 = px.bar(
                tb, x="Tenure", y="Churn Rate",
                title="Churn Rate by Tenure",
                color="Churn Rate", color_continuous_scale="Oranges",
            )
            st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    # Network quality vs churn
    with col_c:
        fig3 = px.box(
            df, x="churned", y="rsrq_avg",
            title="Network Quality (RSRQ) vs Churn",
            labels={"churned": "Churned", "rsrq_avg": "RSRQ (dB)"},
            color="churned",
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Monthly charges distribution
    with col_d:
        fig4 = px.histogram(
            df, x="monthly_charges", color="churned",
            title="Monthly Charges Distribution",
            barmode="overlay", opacity=0.7,
            color_discrete_map={0: "#3498db", 1: "#e74c3c"},
            labels={"monthly_charges": "Monthly Charges ($)", "churned": "Churned"},
        )
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GEO RISK MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Geospatial Churn Risk Map")

    if risk_grid is None:
        st.warning(
            "Risk grid not found. Generate it first:\n"
            "```\npython src/models/geo_risk_map.py\n```"
        )
    else:
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        total_cells = len(risk_grid)

        tier_col = None
        if "risk_tier" not in risk_grid.columns and "avg_churn_probability" in risk_grid.columns:
            tiers = {"LOW": (0, 0.25), "MEDIUM": (0.25, 0.5), "HIGH": (0.5, 0.7), "CRITICAL": (0.7, 1.01)}
            def assign_tier(p):
                for t, (lo, hi) in tiers.items():
                    if lo <= p < hi:
                        return t
                return "CRITICAL"
            risk_grid["risk_tier"] = risk_grid["avg_churn_probability"].apply(assign_tier)

        high_cells = len(risk_grid[risk_grid["risk_tier"].isin(["HIGH", "CRITICAL"])])
        total_rev_risk = risk_grid["estimated_revenue_at_risk"].sum() if "estimated_revenue_at_risk" in risk_grid.columns else 0
        total_subs = risk_grid["subscriber_count"].sum() if "subscriber_count" in risk_grid.columns else 0

        col1.metric("Total H3 Cells", f"{total_cells:,}")
        col2.metric("High/Critical Cells", f"{high_cells:,}")
        col3.metric("Covered Subscribers", f"{total_subs:,}")
        col4.metric("Revenue at Risk", f"${total_rev_risk:,.0f}/mo")

        st.markdown("---")

        # Static choropleth using plotly (Folium map loads from saved HTML)
        map_html = Path("data/processed/churn_risk_map.html")
        if map_html.exists():
            with open(map_html) as f:
                map_content = f.read()
            from streamlit.components.v1 import html
            html(map_content, height=550, scrolling=False)
        else:
            st.info(
                "Interactive Folium map not found. "
                "Run `python src/models/geo_risk_map.py` to generate it.\n\n"
                "Showing scatter plot instead:"
            )
            if "avg_churn_probability" in risk_grid.columns:
                # Fallback scatter map
                fig_map = px.scatter_mapbox(
                    risk_grid,
                    lat="avg_churn_probability",  # placeholder
                    lon="avg_churn_probability",
                    color="risk_tier",
                    size="subscriber_count",
                    mapbox_style="carto-positron",
                    zoom=9,
                    title="Churn Risk by H3 Cell",
                )
                st.plotly_chart(fig_map, use_container_width=True)

        # Risk tier breakdown
        st.subheader("Risk Tier Breakdown")
        tier_counts = risk_grid["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Cell Count"]
        colors = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c", "CRITICAL": "#8e44ad"}
        fig_pie = px.pie(
            tier_counts, values="Cell Count", names="Risk Tier",
            color="Risk Tier", color_discrete_map=colors,
            title="H3 Cells by Churn Risk Tier",
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SUBSCRIBER LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Single Subscriber Risk Lookup")

    if model_art is None:
        st.warning("Model not loaded. Run `python src/models/train.py` first.")
    else:
        col_form, col_result = st.columns([1, 1])

        with col_form:
            st.subheader("Subscriber Attributes")

            sub_id = st.text_input("Subscriber ID", "SUB_0000001")
            tenure = st.slider("Tenure (months)", 1, 72, 18)
            monthly_charges = st.slider("Monthly Charges ($)", 20.0, 150.0, 65.0, step=1.0)
            contract = st.selectbox("Contract Type", ["month-to-month", "one-year", "two-year"])
            internet = st.selectbox("Internet Service", ["fiber_optic", "DSL", "none"])

            st.markdown("**Network Quality**")
            rsrq = st.slider("RSRQ avg (dB)", -20.0, -3.0, -11.0, step=0.5,
                             help="Good: > -10 | Poor: < -15")
            rsrp = st.slider("RSRP avg (dBm)", -130.0, -60.0, -95.0, step=1.0)
            drops = st.slider("Call drops / month", 0, 20, 2)
            outage = st.slider("Outage minutes / month", 0, 300, 30)

            if st.button("🔮 Predict Churn Risk", use_container_width=True):
                from src.api.model_loader import ModelLoader
                ldr = ModelLoader()
                ldr.model = model_art["model"]
                ldr.feature_cols = model_art["feature_cols"]
                ldr.threshold = model_art.get("threshold", 0.45)
                ldr._sub_eng.fit(pd.DataFrame())
                ldr._net_eng.fit(pd.DataFrame())

                payload = {
                    "subscriber_id": sub_id,
                    "tenure_months": tenure,
                    "monthly_charges": monthly_charges,
                    "total_charges": monthly_charges * tenure,
                    "contract_type": contract,
                    "internet_service": internet,
                    "rsrq_avg": rsrq, "rsrp_avg": rsrp,
                    "call_drops_monthly": drops,
                    "outage_minutes_monthly": outage,
                    "data_usage_gb": 10.0, "call_minutes_monthly": 300.0,
                    "sms_monthly": 50.0, "tech_support_calls": 1,
                    "dl_throughput_mbps": 20.0, "call_drop_rate_pct": drops / 300 * 100,
                    "senior_citizen": 0, "phone_service": 1, "multiple_lines": 0,
                    "international_calls": 0,
                    "payment_method": "electronic_check",
                    "latitude": -6.2088, "longitude": 106.8456,
                }
                result = ldr.predict_single(payload)
                st.session_state["pred_result"] = result

        with col_result:
            st.subheader("Prediction Result")
            if "pred_result" in st.session_state:
                r = st.session_state["pred_result"]
                prob = r["churn_probability"]
                tier = r["risk_tier"]

                tier_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "purple"}
                color = tier_color.get(tier, "gray")

                st.markdown(f"### :{color}[{tier} RISK]")

                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Churn Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 25], "color": "#d5f5e3"},
                            {"range": [25, 50], "color": "#fef9e7"},
                            {"range": [50, 70], "color": "#fdebd0"},
                            {"range": [70, 100], "color": "#f9ebea"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 2},
                                      "thickness": 0.75, "value": prob * 100},
                    },
                ))
                gauge.update_layout(height=300)
                st.plotly_chart(gauge, use_container_width=True)

                st.metric("Revenue at Risk", f"${r['monthly_revenue_at_risk']:.2f}/mo")

                if r.get("top_factors"):
                    st.markdown("**Top churn drivers:**")
                    for i, f in enumerate(r["top_factors"], 1):
                        st.write(f"  {i}. `{f}`")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Model Performance Analysis")

    img_col1, img_col2 = st.columns(2)

    fi_path = Path("data/models/feature_importance.png")
    shap_path = Path("data/models/shap_summary.png")

    with img_col1:
        if fi_path.exists():
            st.subheader("Feature Importance")
            st.image(str(fi_path), use_column_width=True)
        else:
            st.info("Feature importance plot not found. Run training first.")

    with img_col2:
        if shap_path.exists():
            st.subheader("SHAP Summary")
            st.image(str(shap_path), use_column_width=True)
        else:
            st.info("SHAP plot not found. Run training first.")

    # Churn score distribution
    if "churn_probability" in df.columns or "churn_probability_true" in df.columns:
        prob_col = "churn_probability" if "churn_probability" in df.columns else "churn_probability_true"
        st.subheader("Predicted Probability Distribution")
        fig_dist = px.histogram(
            df, x=prob_col, color="churned",
            nbins=50, barmode="overlay", opacity=0.7,
            title="Score Distribution by Actual Churn Label",
            color_discrete_map={0: "#3498db", 1: "#e74c3c"},
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ALERTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("⚠️ High-Risk Zone Alerts")

    if risk_grid is not None and "risk_tier" in risk_grid.columns:
        critical = risk_grid[risk_grid["risk_tier"].isin(["CRITICAL", "HIGH"])].sort_values(
            "estimated_revenue_at_risk" if "estimated_revenue_at_risk" in risk_grid.columns else "subscriber_count",
            ascending=False,
        )

        st.markdown(f"**{len(critical)} zones require immediate attention**")
        st.markdown(f"Total estimated revenue at risk: **${critical['estimated_revenue_at_risk'].sum():,.0f}/month**" if "estimated_revenue_at_risk" in critical.columns else "")

        display_cols = [c for c in [
            "risk_tier", "subscriber_count", "predicted_churners",
            "avg_churn_probability", "avg_rsrq",
            "estimated_revenue_at_risk", "avg_outage_minutes",
        ] if c in critical.columns]

        st.dataframe(
            critical[display_cols].head(50),
            use_container_width=True,
        )

        st.subheader("Revenue at Risk by Zone (Top 20)")
        if "estimated_revenue_at_risk" in critical.columns:
            top20 = critical.head(20).reset_index(drop=True)
            top20["zone"] = [f"Zone {i+1}" for i in range(len(top20))]
            fig_rev = px.bar(
                top20, x="zone", y="estimated_revenue_at_risk",
                color="risk_tier",
                color_discrete_map={"HIGH": "#e74c3c", "CRITICAL": "#8e44ad"},
                title="Monthly Revenue at Risk (Top 20 Zones)",
                labels={"estimated_revenue_at_risk": "Revenue at Risk ($)"},
            )
            st.plotly_chart(fig_rev, use_container_width=True)
    else:
        st.info("Generate the risk grid first:\n```\npython src/models/geo_risk_map.py\n```")
