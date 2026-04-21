# %% [markdown]
# # 02 — Feature Engineering
# **Project:** Telecom Churn Prediction with Geospatial Segmentation
#
# This notebook walks through every feature engineering decision:
# - Why each feature was created
# - The business/domain intuition behind it
# - Distribution checks before/after transformation
# - Feature selection via correlation and importance
#
# Run as Jupyter notebook or script: `python notebooks/02_feature_engineering.py`

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid", palette="muted")
Path("docs").mkdir(exist_ok=True)

print("Loading raw data...")
df = pd.read_parquet("data/raw/subscribers.parquet")
print(f"Shape: {df.shape} | Churn rate: {df['churned'].mean():.2%}")

# %% [markdown]
# ## 1. Subscriber-Level Feature Engineering
#
# ### 1.1 Tenure Features
# **Business rationale:** New subscribers are the highest-risk segment in telecom.
# They haven't yet experienced enough of the service to feel "locked in".
# Annual and two-year contracts create switching costs that reduce churn probability.

# %%
from src.features.subscriber_features import SubscriberFeatureEngineer, NetworkFeatureEngineer

eng = SubscriberFeatureEngineer()
df_feat = eng.fit_transform(df.copy())

# Tenure bucket churn rates
tenure_stats = df_feat.groupby("tenure_bucket")["churned"].agg(["mean", "count"]).reset_index()
tenure_stats.columns = ["Tenure Bucket", "Churn Rate", "Count"]
tenure_stats = tenure_stats.sort_values("Churn Rate", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].bar(tenure_stats["Tenure Bucket"], tenure_stats["Churn Rate"],
            color=plt.cm.Reds(np.linspace(0.4, 0.9, len(tenure_stats))))
axes[0].set_title("Churn Rate by Tenure Bucket")
axes[0].set_ylabel("Churn Rate")
axes[0].tick_params(axis="x", rotation=15)
for i, (_, row) in enumerate(tenure_stats.iterrows()):
    axes[0].text(i, row["Churn Rate"] + 0.003, f"{row['Churn Rate']:.1%}",
                ha="center", fontsize=9)

# Contract risk score distribution
axes[1].hist(df_feat["contract_risk_score"], bins=3, edgecolor="white",
             color="#5E5BE0", alpha=0.8)
axes[1].set_title("Contract Risk Score Distribution\n(3=Month-to-Month, 1=2-Year)")
axes[1].set_xlabel("Risk Score")
axes[1].set_ylabel("Count")
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(["2-Year\n(Low Risk)", "1-Year\n(Med Risk)", "M-to-M\n(High Risk)"])

plt.tight_layout()
plt.savefig("docs/feat_tenure_contract.png", bbox_inches="tight")
plt.show()
print("Saved docs/feat_tenure_contract.png")

# %% [markdown]
# ### 1.2 Usage Ratio Features
# **Business rationale:** Raw usage numbers (e.g. 10GB data) mean nothing without context.
# A subscriber paying $100/month who uses 10GB is getting less value per dollar than
# one paying $30/month for the same usage — and thus more likely to churn.

# %%
# Distribution of data_charge_ratio by churn
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

for label, color, linestyle in [(0, "#2ecc71", "-"), (1, "#e74c3c", "--")]:
    subset = df_feat[df_feat["churned"] == label]["data_charge_ratio"]
    subset_clipped = subset.clip(0, subset.quantile(0.97))
    axes[0].hist(subset_clipped, bins=50, alpha=0.6, color=color,
                 label=f"{'Churned' if label else 'Retained'}", density=True)
axes[0].set_title("Data Usage per Dollar (data_charge_ratio)")
axes[0].set_xlabel("GB per $1 Spent")
axes[0].set_ylabel("Density")
axes[0].legend()

# Call intensity
for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
    subset = df_feat[df_feat["churned"] == label]["call_intensity"].clip(0, 30)
    axes[1].hist(subset, bins=50, alpha=0.6, color=color,
                 label=f"{'Churned' if label else 'Retained'}", density=True)
axes[1].set_title("Call Intensity (minutes/month per tenure month)")
axes[1].set_xlabel("Call Intensity")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.savefig("docs/feat_usage_ratios.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Network Quality Feature Engineering
#
# ### 2.1 Network Quality Score (Composite)
# **Business rationale:** RSRP, RSRQ, and throughput are correlated but each captures
# a different dimension of signal quality. We combine them into a single 0–100 score
# to reduce dimensionality and create a human-interpretable KPI.
#
# Formula: `0.40 * RSRQ_norm + 0.35 * RSRP_norm + 0.25 * throughput_norm`
# Weights based on 3GPP standards for LTE network quality assessment.

# %%
net_eng = NetworkFeatureEngineer()
df_feat = net_eng.fit_transform(df_feat)

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# Network quality score vs churn
sns.violinplot(data=df_feat, x="churned", y="network_quality_score",
               palette={0: "#2ecc71", 1: "#e74c3c"}, ax=axes[0, 0])
axes[0, 0].set_title("Network Quality Score (0–100) vs Churn")
axes[0, 0].set_xlabel("Churned")
axes[0, 0].set_ylabel("Network Quality Score")

# Network frustration index
sns.boxplot(data=df_feat, x="churned", y="network_frustration_index",
            palette={0: "#2ecc71", 1: "#e74c3c"}, ax=axes[0, 1])
axes[0, 1].set_title("Network Frustration Index vs Churn")
axes[0, 1].set_xlabel("Churned")

# RSRQ quality buckets
rsrq_churn = df_feat.groupby("rsrq_quality")["churned"].mean().sort_values(ascending=False)
colors = {"poor": "#e74c3c", "fair": "#f39c12", "good": "#2ecc71", "excellent": "#27ae60"}
bar_colors = [colors.get(q, "#95a5a6") for q in rsrq_churn.index]
axes[1, 0].bar(rsrq_churn.index, rsrq_churn.values, color=bar_colors, edgecolor="white")
axes[1, 0].set_title("Churn Rate by RSRQ Quality Bucket")
axes[1, 0].set_ylabel("Churn Rate")
for i, v in enumerate(rsrq_churn.values):
    axes[1, 0].text(i, v + 0.004, f"{v:.1%}", ha="center", fontsize=9)

# Outage buckets
out_churn = df_feat.groupby("outage_bucket")["churned"].mean()
axes[1, 1].bar(out_churn.index, out_churn.values,
               color=sns.color_palette("YlOrRd", len(out_churn)), edgecolor="white")
axes[1, 1].set_title("Churn Rate by Monthly Outage Duration")
axes[1, 1].set_ylabel("Churn Rate")
axes[1, 1].tick_params(axis="x", rotation=15)
for i, v in enumerate(out_churn.values):
    axes[1, 1].text(i, v + 0.004, f"{v:.1%}", ha="center", fontsize=9)

plt.suptitle("Network Quality Features vs Churn", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("docs/feat_network_quality.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Geospatial Feature Engineering
#
# ### 3.1 H3 Hexagonal Grid
# **Business rationale:** H3 (Uber's hexagonal grid system) divides the map into
# equal-area hexagons. Unlike administrative boundaries, H3 cells don't create
# artificial geographic discontinuities. Resolution 8 (~0.74 km²) captures
# neighbourhood-level effects; Resolution 7 (~5.2 km²) captures district-level.

# %%
try:
    import h3
    has_h3 = True
except ImportError:
    has_h3 = False
    print("H3 not installed. Install with: pip install h3")

if has_h3 and "h3_r8" in df_feat.columns:
    cell_stats = df_feat.groupby("h3_r8").agg(
        n=("churned", "count"),
        churn_rate=("churned", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # H3 cell size distribution
    axes[0].hist(cell_stats["n"], bins=40, color="#5E5BE0", edgecolor="white", alpha=0.8)
    axes[0].set_title("H3 Cell Size (subscribers per cell, res=8)")
    axes[0].set_xlabel("Subscribers per H3 Cell")
    axes[0].set_ylabel("Number of H3 Cells")
    axes[0].axvline(cell_stats["n"].median(), color="red", linestyle="--",
                   label=f"Median: {cell_stats['n'].median():.0f}")
    axes[0].legend()

    # Churn rate distribution across H3 cells
    axes[1].hist(cell_stats["churn_rate"], bins=40, color="#e74c3c", edgecolor="white", alpha=0.8)
    axes[1].set_title("Churn Rate Distribution Across H3 Cells (res=8)")
    axes[1].set_xlabel("Churn Rate")
    axes[1].set_ylabel("Number of H3 Cells")
    axes[1].axvline(cell_stats["churn_rate"].mean(), color="k", linestyle="--",
                   label=f"Mean: {cell_stats['churn_rate'].mean():.2%}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("docs/feat_h3_distribution.png", bbox_inches="tight")
    plt.show()

    print(f"\nH3 Statistics (res=8):")
    print(f"  Total cells: {len(cell_stats):,}")
    print(f"  Avg subscribers/cell: {cell_stats['n'].mean():.1f}")
    print(f"  Churn rate std across cells: {cell_stats['churn_rate'].std():.3f}")
    print(f"  Max cell churn rate: {cell_stats['churn_rate'].max():.2%}")

# %% [markdown]
# ### 3.2 Tower Proximity Features
# **Business rationale:** Subscribers far from cell towers get weaker signals
# (lower RSRP/RSRQ). But tower count within 1km captures network density —
# areas with many overlapping towers have better redundancy and fewer outages.

# %%
if "dist_to_nearest_tower_km" in df_feat.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    sns.boxplot(data=df_feat, x="churned", y="dist_to_nearest_tower_km",
                palette={0: "#2ecc71", 1: "#e74c3c"}, ax=axes[0])
    axes[0].set_title("Distance to Nearest Tower vs Churn")
    axes[0].set_xlabel("Churned (0=No, 1=Yes)")
    axes[0].set_ylabel("Distance to Nearest Tower (km)")

    if "towers_within_2km" in df_feat.columns:
        tower_density = df_feat.groupby("towers_within_2km")["churned"].mean()
        axes[1].plot(tower_density.index[:20], tower_density.values[:20],
                    "o-", color="#185FA5", linewidth=2, markersize=6)
        axes[1].set_title("Churn Rate vs Tower Density (within 2km)")
        axes[1].set_xlabel("Number of Towers within 2km")
        axes[1].set_ylabel("Churn Rate")
        axes[1].fill_between(tower_density.index[:20], tower_density.values[:20],
                            alpha=0.1, color="#185FA5")

    plt.tight_layout()
    plt.savefig("docs/feat_tower_proximity.png", bbox_inches="tight")
    plt.show()
else:
    print("Tower features not found. Run feature_pipeline.py first to add them.")

# %% [markdown]
# ## 4. Feature Importance (Pre-Model Screening)
#
# Before training, we screen features using mutual information and point-biserial
# correlation to drop near-zero-importance features early, reducing model complexity.

# %%
from sklearn.feature_selection import mutual_info_classif
from src.features.subscriber_features import encode_categoricals

df_encoded = encode_categoricals(df_feat.copy())

# Numeric columns only
exclude = ["churned", "subscriber_id", "snapshot_date",
           "churn_probability_true", "h3_r7", "h3_r8", "nearest_tower_radio"]
numeric_cols = [
    c for c in df_encoded.columns
    if c not in exclude and pd.api.types.is_numeric_dtype(df_encoded[c])
]

X = df_encoded[numeric_cols].fillna(-999)
y = df_encoded["churned"]

# Mutual information scores (measures nonlinear associations)
print("Computing mutual information scores (takes ~30 seconds)...")
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=numeric_cols).sort_values(ascending=False)

# Point-biserial correlation (linear association with binary target)
pb_corr = X.corrwith(y).abs().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Top 25 by mutual information
top_mi = mi_series.head(25)
axes[0].barh(top_mi.index[::-1], top_mi.values[::-1], color="#5E5BE0", alpha=0.85)
axes[0].set_title("Top 25 Features — Mutual Information Score")
axes[0].set_xlabel("MI Score")

# Top 25 by correlation
top_pb = pb_corr.head(25)
axes[1].barh(top_pb.index[::-1], top_pb.values[::-1], color="#e74c3c", alpha=0.85)
axes[1].set_title("Top 25 Features — |Point-Biserial Correlation|")
axes[1].set_xlabel("|Correlation| with Churn")

plt.tight_layout()
plt.savefig("docs/feat_importance_prescreening.png", bbox_inches="tight")
plt.show()

print(f"\nTop 10 features by mutual information:")
print(mi_series.head(10).round(4).to_string())

# %% [markdown]
# ## 5. Feature Correlation & Multicollinearity Check
#
# Tree models (XGBoost, LightGBM) handle multicollinearity better than linear models,
# but highly correlated feature pairs can make SHAP explanations harder to interpret.
# We identify pairs with |correlation| > 0.85 for potential pruning.

# %%
corr_matrix = X[mi_series.head(30).index].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 7}, linewidths=0.3)
ax.set_title("Feature Correlation Matrix (Top 30 Features by MI)", fontsize=12)
plt.tight_layout()
plt.savefig("docs/feat_correlation_matrix.png", bbox_inches="tight")
plt.show()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        val = abs(corr_matrix.iloc[i, j])
        if val > 0.80:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], round(val, 3)))

if high_corr_pairs:
    print(f"\n⚠️  High correlation pairs (|r| > 0.80):")
    for f1, f2, r in sorted(high_corr_pairs, key=lambda x: -x[2]):
        print(f"  {f1:40s} ↔ {f2:40s}  r={r}")
else:
    print("\n✅ No highly correlated feature pairs found (|r| > 0.80)")

# %% [markdown]
# ## 6. Feature Engineering Summary
#
# | Feature Group | Count | Key Features | Why Important |
# |---------------|-------|-------------|---------------|
# | Subscriber base | 11 | tenure, charges, contract | Business contract drivers |
# | Subscriber engineered | 8 | risk_score, ratios, buckets | Value perception signals |
# | Network raw | 6 | RSRQ, RSRP, throughput | QoS measurement |
# | Network engineered | 7 | quality_score, frustration_index | Composite KPIs |
# | Geospatial | 6 | h3_cell, dist_tower, tower_density | Coverage quality |
# | H3 aggregates | 6 | cell_avg_rsrq, cell_churn_rate | Neighbourhood effects |
# | Categorical encoded | ~15 | contract_type_*, payment_* | Model-consumable |
# | **Total** | **~59** | | |

print("\n✅ Feature engineering notebook complete. All charts saved to docs/")
