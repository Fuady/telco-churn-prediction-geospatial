# %% [markdown]
# # 04 — Geospatial Churn Analysis
# **Project:** Telecom Churn Prediction with Geospatial Segmentation
#
# This notebook:
# - Maps subscriber churn rates onto H3 hex grid
# - Identifies geographic churn hotspots
# - Correlates network quality with spatial churn patterns
# - Computes revenue at risk by zone
# - Produces export-ready maps for marketing/ops teams

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve()))

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

try:
    import h3
    H3_AVAILABLE = True
    print("H3 available ✓")
except ImportError:
    H3_AVAILABLE = False
    print("H3 not available — install with: pip install h3")

# %%
# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet("data/processed/features_full.parquet")
print(f"Loaded {len(df):,} subscribers")

# %%
# ── Add H3 indexes if not already present ────────────────────────────────────
if H3_AVAILABLE and "h3_r8" not in df.columns:
    from src.features.geospatial_features import add_h3_indexes
    df = add_h3_indexes(df, resolutions=[7, 8])

# ── Aggregate to H3 level ─────────────────────────────────────────────────────
h3_col = "h3_r8" if "h3_r8" in df.columns else "subscriber_id"

if h3_col != "subscriber_id":
    h3_agg = df.groupby(h3_col).agg(
        n_subscribers=(h3_col, "count"),
        churn_rate=("churned", "mean"),
        avg_rsrq=("rsrq_avg", "mean"),
        avg_throughput=("dl_throughput_mbps", "mean"),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_outage=("outage_minutes_monthly", "mean"),
        total_monthly_revenue=("monthly_charges", "sum"),
    ).reset_index()

    h3_agg["revenue_at_risk"] = h3_agg["total_monthly_revenue"] * h3_agg["churn_rate"]
    h3_agg = h3_agg[h3_agg["n_subscribers"] >= 10]  # filter thin cells

    print(f"H3 cells (res=8): {len(h3_agg):,}")
    print(f"Top 10 highest churn cells:")
    print(h3_agg.nlargest(10, "churn_rate")[
        [h3_col, "n_subscribers", "churn_rate", "avg_rsrq", "revenue_at_risk"]
    ].to_string())

# %% [markdown]
# ## Churn Rate vs Network Quality by H3 Cell

# %%
if h3_col != "subscriber_id":
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Churn rate vs RSRQ
    scatter = axes[0].scatter(
        h3_agg["avg_rsrq"], h3_agg["churn_rate"],
        c=h3_agg["n_subscribers"], cmap="Blues",
        alpha=0.6, edgecolors="none", s=30
    )
    axes[0].set_xlabel("Avg RSRQ (dB) — higher is better")
    axes[0].set_ylabel("Churn Rate")
    axes[0].set_title("RSRQ vs Churn Rate per H3 Cell\n(colour = subscriber count)")
    plt.colorbar(scatter, ax=axes[0], label="Subscribers in cell")

    z = np.polyfit(h3_agg["avg_rsrq"].dropna(), h3_agg["churn_rate"].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(h3_agg["avg_rsrq"].min(), h3_agg["avg_rsrq"].max(), 100)
    axes[0].plot(x_line, p(x_line), "r--", alpha=0.7, label="Trend")
    axes[0].legend()

    # Revenue at risk distribution
    h3_agg["revenue_at_risk"].clip(0, h3_agg["revenue_at_risk"].quantile(0.95)).hist(
        bins=40, ax=axes[1], color="#e74c3c", edgecolor="white", alpha=0.8
    )
    axes[1].set_xlabel("Monthly Revenue at Risk ($)")
    axes[1].set_ylabel("Number of H3 Cells")
    axes[1].set_title("Distribution of Revenue at Risk per H3 Cell")
    total_risk = h3_agg["revenue_at_risk"].sum()
    axes[1].axvline(h3_agg["revenue_at_risk"].mean(), color="k", linestyle="--", label=f"Mean: ${h3_agg['revenue_at_risk'].mean():.0f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("docs/geo_analysis_scatter.png", bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## Top 20 High-Risk Zones (Business Output Table)

# %%
if h3_col != "subscriber_id":
    top_zones = h3_agg.nlargest(20, "revenue_at_risk")[[
        h3_col, "n_subscribers", "churn_rate",
        "avg_rsrq", "avg_outage", "revenue_at_risk"
    ]].copy()
    top_zones["churn_rate"] = (top_zones["churn_rate"] * 100).round(1).astype(str) + "%"
    top_zones["revenue_at_risk"] = "$" + top_zones["revenue_at_risk"].round(0).astype(int).astype(str)
    top_zones["avg_rsrq"] = top_zones["avg_rsrq"].round(1)
    top_zones["avg_outage"] = top_zones["avg_outage"].round(0).astype(int)
    top_zones.columns = ["H3 Cell", "Subscribers", "Churn Rate",
                         "Avg RSRQ (dB)", "Avg Outage (min)", "Monthly Rev at Risk"]
    top_zones.index = range(1, len(top_zones) + 1)
    print(top_zones.to_string())

    # Save as CSV for reporting
    top_zones.to_csv("docs/top_risk_zones.csv")
    print("\nSaved to docs/top_risk_zones.csv")

# %% [markdown]
# ## Interactive Folium Map

# %%
if H3_AVAILABLE and h3_col != "subscriber_id":
    from shapely.geometry import Polygon

    centre_lat = df["latitude"].mean()
    centre_lon = df["longitude"].mean()

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=10,
                   tiles="CartoDB positron")

    # Build a colormap
    churn_max = h3_agg["churn_rate"].quantile(0.95)
    colormap = folium.LinearColormap(
        colors=["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"],
        vmin=0, vmax=churn_max,
        caption="Churn Rate",
    )

    for _, row in h3_agg.iterrows():
        try:
            boundary = h3.h3_to_geo_boundary(row[h3_col], geo_json=True)
            poly = folium.Polygon(
                locations=[[c[1], c[0]] for c in boundary],
                color=colormap(min(row["churn_rate"], churn_max)),
                fill_color=colormap(min(row["churn_rate"], churn_max)),
                fill_opacity=0.65,
                weight=0.5,
                tooltip=(
                    f"Subscribers: {int(row['n_subscribers'])}<br>"
                    f"Churn Rate: {row['churn_rate']:.1%}<br>"
                    f"RSRQ: {row['avg_rsrq']:.1f} dB<br>"
                    f"Rev at Risk: ${row['revenue_at_risk']:.0f}/mo"
                ),
            )
            poly.add_to(m)
        except Exception:
            pass

    colormap.add_to(m)
    m.save("docs/geo_churn_analysis_map.html")
    print("Map saved to docs/geo_churn_analysis_map.html")
else:
    print("H3 not available or no H3 column — skipping interactive map")

# %% [markdown]
# ## Summary Statistics

# %%
print("\n=== GEOSPATIAL ANALYSIS SUMMARY ===")
print(f"Total subscribers analyzed: {len(df):,}")

if h3_col != "subscriber_id":
    print(f"H3 cells (res=8, ≥10 subs): {len(h3_agg):,}")
    print(f"Overall churn rate: {df['churned'].mean():.2%}")
    top_20_pct = h3_agg.nlargest(int(len(h3_agg) * 0.2), "churn_rate")
    pct_churners_in_top20 = (top_20_pct["n_subscribers"] * top_20_pct["churn_rate"]).sum() / (df["churned"].sum() + 1e-10)
    print(f"Top 20% cells account for {pct_churners_in_top20:.0%} of predicted churners")
    print(f"Total monthly revenue at risk: ${h3_agg['revenue_at_risk'].sum():,.0f}")

print("\n✅ Geospatial analysis complete.")
