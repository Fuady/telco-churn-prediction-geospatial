# %% [markdown]
# # 03 — Model Training & Evaluation
# **Project:** Telecom Churn Prediction with Geospatial Segmentation
#
# This notebook covers:
# - Baseline model vs optimised model comparison
# - Class imbalance handling (SMOTE) — impact analysis
# - Threshold tuning for business objectives
# - Full evaluation suite: ROC, PR, calibration, lift curves
# - SHAP deep-dive: global importance + local explanations
# - Cross-validation results
# - Model comparison: XGBoost vs LightGBM
#
# Run: `python notebooks/03_model_training.py`
# Or train directly: `python src/models/train.py`

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
import joblib
warnings.filterwarnings("ignore")

matplotlib.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid")
Path("docs").mkdir(exist_ok=True)

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, brier_score_loss,
    classification_report,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

# %%
# ── Load processed data ───────────────────────────────────────────────────────
print("Loading processed features...")
df_train = pd.read_parquet("data/processed/features_train.parquet")
df_test  = pd.read_parquet("data/processed/features_test.parquet")

feature_path = Path("data/processed/feature_columns.txt")
if feature_path.exists():
    feature_cols = feature_path.read_text().strip().split("\n")
else:
    exclude = ["churned", "subscriber_id", "snapshot_date", "churn_probability_true",
               "h3_r7", "h3_r8", "nearest_tower_radio"]
    feature_cols = [c for c in df_train.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(df_train[c])]

TARGET = "churned"
X_train = df_train[feature_cols].fillna(-999)
y_train = df_train[TARGET]
X_test  = df_test[feature_cols].fillna(-999)
y_test  = df_test[TARGET]

print(f"Train: {X_train.shape} | Churn: {y_train.mean():.2%}")
print(f"Test:  {X_test.shape}  | Churn: {y_test.mean():.2%}")
print(f"Features: {len(feature_cols)}")

# %% [markdown]
# ## 1. Class Imbalance — Impact of SMOTE
#
# Churn datasets are inherently imbalanced (~15–25% positive class).
# Without correction, models learn to predict "no churn" for everyone and
# still achieve high accuracy — but terrible recall on the minority class.

# %%
# Baseline model WITHOUT SMOTE
model_no_smote = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42, verbosity=0, n_jobs=-1
)
model_no_smote.fit(X_train, y_train)
prob_no_smote = model_no_smote.predict_proba(X_test)[:, 1]

# WITH SMOTE
print("Applying SMOTE...")
sm = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_res.shape} | Churn rate: {y_res.mean():.2%}")

model_smote = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, verbosity=0, n_jobs=-1
)
model_smote.fit(X_res, y_res)
prob_smote = model_smote.predict_proba(X_test)[:, 1]

# Compare PR curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for name, probs, color, ls in [
    ("Without SMOTE", prob_no_smote, "#3498db", "--"),
    ("With SMOTE",    prob_smote,    "#e74c3c", "-"),
]:
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    axes[0].plot(rec, prec, color=color, linestyle=ls,
                 linewidth=2, label=f"{name} (AP={ap:.3f})")

axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("Precision-Recall Curve: SMOTE Impact")
axes[0].legend()
axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

# F1 at different thresholds
thresholds = np.arange(0.10, 0.80, 0.01)
for name, probs, color, ls in [
    ("Without SMOTE", prob_no_smote, "#3498db", "--"),
    ("With SMOTE",    prob_smote,    "#e74c3c", "-"),
]:
    f1s = [f1_score(y_test, (probs >= t).astype(int), zero_division=0) for t in thresholds]
    axes[1].plot(thresholds, f1s, color=color, linestyle=ls,
                 linewidth=2, label=f"{name} (max F1={max(f1s):.3f})")

axes[1].set_xlabel("Classification Threshold")
axes[1].set_ylabel("F1 Score")
axes[1].set_title("F1 Score vs Threshold: SMOTE Impact")
axes[1].legend()

plt.tight_layout()
plt.savefig("docs/model_smote_comparison.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Full Model Evaluation Suite
#
# We train the final model with optimised hyperparameters and evaluate it comprehensively.
# If you've already run `train.py`, load the saved model instead.

# %%
model_path = Path("data/models/churn_model_xgboost.pkl")
if model_path.exists():
    print(f"Loading saved model from {model_path}")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    threshold = artifact.get("threshold", 0.45)
    feature_cols = artifact["feature_cols"]
    X_test_aligned = X_test[feature_cols].fillna(-999)
else:
    print("No saved model found — using SMOTE-trained model from this notebook")
    model = model_smote
    threshold = 0.45
    X_test_aligned = X_test

y_prob = model.predict_proba(X_test_aligned)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)
f1      = f1_score(y_test, y_pred, zero_division=0)
prec    = precision_score(y_test, y_pred, zero_division=0)
rec     = recall_score(y_test, y_pred, zero_division=0)
brier   = brier_score_loss(y_test, y_prob)

print(f"\n{'='*50}")
print(f"  FINAL MODEL EVALUATION (threshold={threshold:.2f})")
print(f"{'='*50}")
print(f"  ROC-AUC  : {roc_auc:.4f}")
print(f"  PR-AUC   : {pr_auc:.4f}")
print(f"  F1       : {f1:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  Brier    : {brier:.4f}  (lower=better, 0=perfect)")
print(f"{'='*50}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Retained','Churned'])}")

# %% [markdown]
# ## 3. ROC, PR, Calibration & Lift Curves

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# ── ROC Curve ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0, 0].plot(fpr, tpr, color="#e74c3c", linewidth=2.5,
                label=f"XGBoost (AUC = {roc_auc:.3f})")
axes[0, 0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
axes[0, 0].fill_between(fpr, tpr, alpha=0.08, color="#e74c3c")
axes[0, 0].set_xlabel("False Positive Rate (1 - Specificity)")
axes[0, 0].set_ylabel("True Positive Rate (Recall)")
axes[0, 0].set_title("ROC Curve")
axes[0, 0].legend()

# ── PR Curve ──────────────────────────────────────────────────────────────────
prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_test, y_prob)
baseline = y_test.mean()
axes[0, 1].plot(rec_curve, prec_curve, color="#185FA5", linewidth=2.5,
                label=f"XGBoost (AP = {pr_auc:.3f})")
axes[0, 1].axhline(baseline, color="k", linestyle="--", linewidth=1,
                   label=f"Random baseline ({baseline:.3f})")
axes[0, 1].fill_between(rec_curve, prec_curve, alpha=0.08, color="#185FA5")
axes[0, 1].set_xlabel("Recall")
axes[0, 1].set_ylabel("Precision")
axes[0, 1].set_title("Precision-Recall Curve")
axes[0, 1].legend()

# ── Calibration Curve ─────────────────────────────────────────────────────────
frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
axes[1, 0].plot(mean_pred, frac_pos, "s-", color="#8e44ad", linewidth=2,
                markersize=7, label="XGBoost")
axes[1, 0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
axes[1, 0].set_xlabel("Mean Predicted Probability")
axes[1, 0].set_ylabel("Fraction of Positives")
axes[1, 0].set_title("Calibration Curve\n(How well probabilities match reality)")
axes[1, 0].legend()

# ── Precision & Recall vs Threshold ───────────────────────────────────────────
thresh_range = np.arange(0.10, 0.80, 0.01)
precisions = [precision_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresh_range]
recalls    = [recall_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresh_range]
f1s        = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresh_range]

axes[1, 1].plot(thresh_range, precisions, color="#e74c3c", linewidth=2, label="Precision")
axes[1, 1].plot(thresh_range, recalls,    color="#2ecc71", linewidth=2, label="Recall")
axes[1, 1].plot(thresh_range, f1s,        color="#185FA5", linewidth=2, label="F1")
axes[1, 1].axvline(threshold, color="k", linestyle="--", linewidth=1.5,
                   label=f"Chosen threshold ({threshold:.2f})")
axes[1, 1].set_xlabel("Classification Threshold")
axes[1, 1].set_ylabel("Score")
axes[1, 1].set_title("Precision, Recall & F1 vs Threshold")
axes[1, 1].legend()

plt.suptitle("Model Evaluation Suite", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("docs/model_evaluation_suite.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Confusion Matrix

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Standard confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Retained", "Churned"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Confusion Matrix (threshold={threshold:.2f})")

# Normalised confusion matrix
cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                                    display_labels=["Retained", "Churned"])
disp_norm.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Normalised Confusion Matrix")

plt.tight_layout()
plt.savefig("docs/model_confusion_matrix.png", bbox_inches="tight")
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\n  True Negatives  (correctly retained): {tn:,}")
print(f"  False Positives (flagged wrongly):     {fp:,}")
print(f"  False Negatives (missed churners):     {fn:,}")
print(f"  True Positives  (caught churners):     {tp:,}")
print(f"\n  Specificity: {tn/(tn+fp):.3f}")
print(f"  Recall:      {tp/(tp+fn):.3f}")

# %% [markdown]
# ## 5. SHAP Analysis
#
# SHAP (SHapley Additive exPlanations) gives us model-agnostic feature importances
# that are grounded in game theory. Unlike built-in feature importances, SHAP values:
# - Sum to the model output for each prediction
# - Account for feature interactions
# - Allow local (per-subscriber) and global explanations

# %%
print("Computing SHAP values (sample of 3000 for speed)...")
sample_idx = np.random.default_rng(42).choice(len(X_test_aligned), 3000, replace=False)
X_shap = X_test_aligned.iloc[sample_idx]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # class 1 for binary classification

# ── Global SHAP summary plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 9))
shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
plt.title("SHAP Summary Plot — Top 20 Features\n(dot colour = feature value, x-axis = impact on churn probability)", 
          fontsize=11)
plt.tight_layout()
plt.savefig("docs/model_shap_summary.png", bbox_inches="tight")
plt.show()
print("Saved docs/model_shap_summary.png")

# %%
# ── SHAP bar chart (mean absolute impact) ────────────────────────────────────
mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X_test_aligned.columns
).sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
mean_abs_shap.sort_values().plot.barh(ax=ax, color="#5E5BE0", alpha=0.85)
ax.set_title("Mean |SHAP| Value — Top 20 Features\n(Average impact on churn probability magnitude)")
ax.set_xlabel("Mean |SHAP Value|")
plt.tight_layout()
plt.savefig("docs/model_shap_bar.png", bbox_inches="tight")
plt.show()

# %%
# ── Local explanation for a single high-risk subscriber ──────────────────────
# Find the highest predicted-churn subscriber in the test set
high_risk_idx = y_prob.argmax()
high_risk_prob = y_prob[high_risk_idx]

print(f"\n🔴 Example: Highest risk subscriber")
print(f"   Churn probability: {high_risk_prob:.2%}")
print(f"   Actual churn:      {'YES' if y_test.iloc[high_risk_idx] == 1 else 'NO'}")

# SHAP waterfall for this subscriber
shap_single = explainer(X_test_aligned.iloc[[high_risk_idx]])
fig, ax = plt.subplots(figsize=(10, 6))
shap.waterfall_plot(shap_single[0], max_display=15, show=False)
plt.title(f"SHAP Waterfall — Highest Risk Subscriber (P={high_risk_prob:.2%})", fontsize=11)
plt.tight_layout()
plt.savefig("docs/model_shap_waterfall_high_risk.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Cumulative Gain / Lift Chart
#
# This shows the business value of the model: if we contact the top X% of
# subscribers ranked by churn probability, what % of actual churners do we reach?
# 
# A perfect model reaches 100% of churners by contacting the top (churn_rate)% only.

# %%
results_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_prob": y_prob,
}).sort_values("y_prob", ascending=False).reset_index(drop=True)

total_pos = results_df["y_true"].sum()
results_df["cumulative_pos"] = results_df["y_true"].cumsum()
results_df["gain"] = results_df["cumulative_pos"] / total_pos
results_df["pct_contacted"] = (results_df.index + 1) / len(results_df)

churn_rate = y_test.mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cumulative gain curve
axes[0].plot(results_df["pct_contacted"] * 100, results_df["gain"] * 100,
             color="#e74c3c", linewidth=2.5, label="XGBoost")
axes[0].plot([0, churn_rate * 100, 100], [0, 100, 100],
             color="#27ae60", linestyle="--", linewidth=1.5, label="Perfect model")
axes[0].plot([0, 100], [0, 100], color="#95a5a6", linestyle=":", linewidth=1.5,
             label="Random baseline")
axes[0].fill_between(results_df["pct_contacted"] * 100,
                     results_df["gain"] * 100, results_df["pct_contacted"] * 100,
                     alpha=0.07, color="#e74c3c")
axes[0].set_xlabel("% of Subscribers Contacted (ranked by model)")
axes[0].set_ylabel("% of Total Churners Captured")
axes[0].set_title("Cumulative Gain Chart")
axes[0].legend()
axes[0].set_xlim(0, 100); axes[0].set_ylim(0, 100)

# Key business metrics from the gain chart
for target_recall in [0.50, 0.70, 0.80]:
    idx = results_df[results_df["gain"] >= target_recall].index[0]
    pct_needed = (idx + 1) / len(results_df) * 100
    axes[0].annotate(f"{target_recall:.0%} recall\n→ top {pct_needed:.0f}%",
                    xy=(pct_needed, target_recall * 100),
                    xytext=(pct_needed + 8, target_recall * 100 - 10),
                    fontsize=8, color="#e74c3c",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1))

# Lift curve
results_df["lift"] = results_df["gain"] / (results_df["pct_contacted"] + 1e-10)
axes[1].plot(results_df["pct_contacted"] * 100, results_df["lift"],
             color="#185FA5", linewidth=2.5, label="XGBoost lift")
axes[1].axhline(1.0, color="#95a5a6", linestyle=":", linewidth=1.5, label="No lift baseline")
axes[1].set_xlabel("% of Subscribers Contacted")
axes[1].set_ylabel("Lift (vs random targeting)")
axes[1].set_title("Lift Curve\n(Model vs Random Targeting)")
axes[1].legend()
axes[1].set_xlim(0, 100)

plt.tight_layout()
plt.savefig("docs/model_lift_gain_chart.png", bbox_inches="tight")
plt.show()

# Business insight
top10 = results_df.iloc[:int(len(results_df) * 0.10)]
gain10 = top10["y_true"].sum() / total_pos
print(f"\nKey lift insights:")
print(f"  Contacting top 10% of subscribers captures {gain10:.0%} of all churners")
print(f"  Contacting top 20% captures {results_df.iloc[:int(len(results_df)*0.20)]['y_true'].sum()/total_pos:.0%}")
print(f"  Lift at top 10%: {gain10 / 0.10:.1f}× better than random")

# %% [markdown]
# ## 7. Cross-Validation Results
#
# 5-fold stratified cross-validation confirms that results generalise across data splits.

# %%
print("\nRunning 5-fold stratified cross-validation (takes ~2 min)...")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42, verbosity=0, n_jobs=-1
)

cv_results = cross_validate(
    cv_model, X_train, y_train, cv=cv,
    scoring=["roc_auc", "average_precision", "f1"],
    n_jobs=-1, verbose=0,
)

print("\n5-Fold Cross-Validation Results:")
print(f"  ROC-AUC : {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
print(f"  PR-AUC  : {cv_results['test_average_precision'].mean():.4f} ± {cv_results['test_average_precision'].std():.4f}")
print(f"  F1      : {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
print("\nFold-by-fold ROC-AUC:", [f"{v:.4f}" for v in cv_results["test_roc_auc"]])

# %%
print("\n✅ Model training & evaluation notebook complete. All charts saved to docs/")
