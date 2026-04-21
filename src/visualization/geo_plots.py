"""
src/visualization/geo_plots.py
────────────────────────────────
Geospatial visualization utilities.
Produces Folium interactive maps and static choropleth plots.
"""

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from folium.plugins import HeatMap, MarkerCluster
from loguru import logger


# ── Risk tier colour palette ──────────────────────────────────────────────────
RISK_COLORS = {
    "LOW":      "#2ecc71",
    "MEDIUM":   "#f39c12",
    "HIGH":     "#e74c3c",
    "CRITICAL": "#8e44ad",
}

RISK_TIERS_DEFAULT = {
    "LOW":      (0.00, 0.25),
    "MEDIUM":   (0.25, 0.50),
    "HIGH":     (0.50, 0.70),
    "CRITICAL": (0.70, 1.01),
}


def assign_risk_tier(prob: float, tiers: dict = None) -> str:
    tiers = tiers or RISK_TIERS_DEFAULT
    for tier, (lo, hi) in tiers.items():
        if lo <= prob < hi:
            return tier
    return "CRITICAL"


def make_base_map(
    centre_lat: float = -6.2,
    centre_lon: float = 106.85,
    zoom: int = 10,
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """Create a base Folium map."""
    return folium.Map(location=[centre_lat, centre_lon], zoom_start=zoom, tiles=tiles)


def add_subscriber_scatter(
    m: folium.Map,
    df: pd.DataFrame,
    max_points: int = 5000,
    show_churned_only: bool = False,
) -> folium.Map:
    """
    Add subscriber scatter points to a Folium map.
    Green = retained, Red = churned.
    """
    if show_churned_only:
        sample = df[df["churned"] == 1].sample(
            min(max_points, len(df[df["churned"] == 1])), random_state=42
        )
    else:
        sample = df.sample(min(max_points, len(df)), random_state=42)

    for _, row in sample.iterrows():
        color = "#e74c3c" if row.get("churned", 0) == 1 else "#2ecc71"
        contract = row.get("contract_type", "unknown")
        prob = row.get("churn_probability", "N/A")
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=0.5,
            tooltip=f"ID: {row.get('subscriber_id', '?')}<br>Contract: {contract}<br>Churn prob: {prob}",
        ).add_to(m)

    return m


def add_churn_heatmap(
    m: folium.Map,
    df: pd.DataFrame,
    weight_col: str = "churn_probability",
) -> folium.Map:
    """Add a heatmap layer of churn probability density."""
    data_col = weight_col if weight_col in df.columns else "churned"
    heat_data = [
        [row["latitude"], row["longitude"], row[data_col]]
        for _, row in df.iterrows()
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"])
    ]
    HeatMap(
        heat_data,
        radius=12,
        blur=8,
        max_zoom=13,
        gradient={"0.0": "#2ecc71", "0.4": "#f39c12", "0.7": "#e74c3c", "1.0": "#8e44ad"},
    ).add_to(m)
    return m


def add_h3_risk_layer(
    m: folium.Map,
    risk_gdf,  # GeoDataFrame with H3 polygons
    churn_col: str = "avg_churn_probability",
    tiers: Optional[dict] = None,
) -> folium.Map:
    """Add H3 hexagonal risk polygons to a Folium map."""
    try:
        import h3
    except ImportError:
        logger.warning("H3 not installed — cannot add H3 risk layer")
        return m

    tiers = tiers or RISK_TIERS_DEFAULT

    for _, row in risk_gdf.iterrows():
        prob = row.get(churn_col, 0)
        tier = assign_risk_tier(prob, tiers)
        color = RISK_COLORS[tier]

        try:
            geojson_str = risk_gdf.loc[[_]].to_json()
            folium.GeoJson(
                geojson_str,
                style_function=lambda feat, c=color: {
                    "fillColor": c,
                    "color": "#333333",
                    "weight": 0.5,
                    "fillOpacity": 0.60,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["subscriber_count", "predicted_churners",
                            "avg_churn_probability", "risk_tier",
                            "estimated_revenue_at_risk"],
                    aliases=["Subscribers", "Churners", "Avg Prob",
                              "Risk Tier", "Rev at Risk ($)"],
                    localize=True,
                ) if all(c in risk_gdf.columns for c in [
                    "subscriber_count", "predicted_churners", "avg_churn_probability",
                    "risk_tier", "estimated_revenue_at_risk"
                ]) else None,
            ).add_to(m)
        except Exception:
            pass

    return m


def add_tower_markers(
    m: folium.Map,
    towers_df: pd.DataFrame,
    max_towers: int = 500,
    cluster: bool = True,
) -> folium.Map:
    """Add cell tower markers to the map."""
    sample = towers_df.sample(min(max_towers, len(towers_df)), random_state=42)

    radio_icons = {"LTE": "📶", "NR": "5️⃣", "UMTS": "3️⃣", "GSM": "2️⃣"}
    radio_colors = {"LTE": "blue", "NR": "purple", "UMTS": "orange", "GSM": "gray"}

    layer = MarkerCluster(name="Cell Towers") if cluster else m

    for _, row in sample.iterrows():
        radio = row.get("radio", "LTE")
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=folium.Icon(color=radio_colors.get(radio, "blue"),
                             icon="signal", prefix="fa"),
            tooltip=f"Tower: {row.get('tower_id', '?')}<br>Radio: {radio}<br>Range: {row.get('range_m', '?')}m",
        ).add_to(layer)

    if cluster:
        layer.add_to(m)
    return m


def add_map_legend(m: folium.Map, tiers: Optional[dict] = None) -> folium.Map:
    """Add a risk tier legend to the map."""
    tiers = tiers or RISK_TIERS_DEFAULT
    rows = "".join([
        f'<tr><td style="background:{RISK_COLORS[t]};width:18px;height:14px;border-radius:3px;"></td>'
        f'<td style="padding-left:8px;font-size:12px;">{t} ({lo:.0%}–{hi:.0%})</td></tr>'
        for t, (lo, hi) in tiers.items()
    ])
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:#fff;padding:10px 14px;border-radius:8px;
                border:1px solid #ccc;box-shadow:2px 2px 6px rgba(0,0,0,.15);">
      <b style="font-size:13px;">Churn Risk</b>
      <table style="margin-top:6px;border-spacing:4px;">{rows}</table>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def create_full_risk_map(
    risk_gdf,
    subscribers_df: Optional[pd.DataFrame] = None,
    towers_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    tiers: Optional[dict] = None,
) -> folium.Map:
    """
    Build a complete layered churn risk map:
    - H3 risk polygons (churn probability choropleth)
    - Optional subscriber scatter
    - Optional cell tower markers
    - Risk legend
    """
    if len(risk_gdf) > 0 and hasattr(risk_gdf, "geometry"):
        centre_lat = risk_gdf.geometry.centroid.y.mean()
        centre_lon = risk_gdf.geometry.centroid.x.mean()
    else:
        centre_lat, centre_lon = -6.20, 106.85

    m = make_base_map(centre_lat, centre_lon)

    # H3 risk hexagons
    m = add_h3_risk_layer(m, risk_gdf, tiers=tiers)

    # Optional: subscriber heatmap
    if subscribers_df is not None and len(subscribers_df) > 0:
        m = add_churn_heatmap(m, subscribers_df.sample(min(5000, len(subscribers_df))))

    # Optional: cell tower markers
    if towers_df is not None and len(towers_df) > 0:
        m = add_tower_markers(m, towers_df)

    m = add_map_legend(m, tiers)
    folium.LayerControl(collapsed=False).add_to(m)

    if output_path:
        m.save(str(output_path))
        logger.success(f"Map saved → {output_path}")

    return m


def plot_h3_risk_static(
    risk_df: pd.DataFrame,
    value_col: str = "avg_churn_probability",
    title: str = "Churn Risk by H3 Cell",
    figsize: tuple = (10, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Static matplotlib bar chart of top H3 cells by risk (fallback when Folium unavailable)."""
    top_cells = risk_df.nlargest(25, value_col)

    fig, ax = plt.subplots(figsize=figsize)
    colors = [RISK_COLORS[assign_risk_tier(p)] for p in top_cells[value_col]]
    ax.barh(range(len(top_cells)), top_cells[value_col].values,
            color=colors, edgecolor="white", alpha=0.85)

    ax.set_yticks(range(len(top_cells)))
    ax.set_yticklabels([f"Cell {i+1}" for i in range(len(top_cells))], fontsize=8)
    ax.set_xlabel(value_col)
    ax.set_title(f"{title}\n(Top 25 cells)")
    ax.axvline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.5)

    # Legend
    patches = [plt.Rectangle((0, 0), 1, 1, color=c) for c in RISK_COLORS.values()]
    ax.legend(patches, RISK_COLORS.keys(), loc="lower right", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig
