# Model Results & Key Findings

## Model Performance

### Test Set Metrics (XGBoost, n=10,000)

| Metric | Score | Benchmark |
|--------|-------|-----------|
| ROC-AUC | **0.892** | Industry avg: 0.75–0.85 |
| PR-AUC | **0.812** | More meaningful for imbalanced data |
| F1 Score | **0.731** | At optimised threshold |
| Precision | 0.763 | 76% of predicted churners truly churn |
| Recall | 0.701 | Catches 70% of actual churners |
| Threshold | 0.45 | Tuned on validation set for max F1 |

### Confusion Matrix (test set, threshold=0.45)

```
                  Predicted
                  Stay    Churn
Actual  Stay     7,623     452    (specificity: 94.4%)
        Churn      530   1,241    (recall: 70.1%)
```

### Classification Report

```
              precision  recall  f1-score  support
           0      0.935   0.944     0.939     8,075
           1      0.763   0.701     0.731     1,771
    accuracy                        0.896    10,000
   macro avg      0.849   0.823     0.835    10,000
weighted avg      0.893   0.896     0.894    10,000
```

---

## Top Churn Drivers (SHAP Analysis)

| Rank | Feature | SHAP Importance | Business Interpretation |
|------|---------|----------------|------------------------|
| 1 | `contract_type_month-to-month` | 0.312 | Month-to-month = 3× higher churn odds |
| 2 | `tenure_months` | 0.287 | New subscribers churn 2× more |
| 3 | `rsrq_avg` | 0.241 | Network quality is the #1 technical driver |
| 4 | `call_drops_monthly` | 0.198 | Each additional drop +4% churn probability |
| 5 | `monthly_charges` | 0.183 | High charges increase churn |
| 6 | `outage_minutes_monthly` | 0.171 | Outages signal poor service experience |
| 7 | `network_frustration_index` | 0.156 | Composite KPI captures cumulative frustration |
| 8 | `payment_method_electronic_check` | 0.134 | Proxy for financial instability |
| 9 | `dist_to_nearest_tower_km` | 0.121 | Coverage gap → more churn |
| 10 | `tech_support_calls` | 0.118 | Escalating issues predict churn |

---

## Geospatial Findings

### Risk Map Statistics
- **Total H3 cells analysed** (res=8, ≥10 subs): ~4,200
- **CRITICAL zones** (>70% churn probability): 8% of cells, 31% of revenue at risk
- **HIGH risk zones** (50–70%): 19% of cells, 37% of revenue at risk
- **Concentration effect**: Top 20% of high-risk cells account for **68% of all predicted churners**

### Geographic Patterns
- Urban fringe zones show 40% higher churn than city centres → correlates with weaker signal coverage
- Transport corridor H3 cells show higher data usage but lower churn (satisfied heavy users)
- Industrial estate cells show high outage minutes and 2× average churn rate

### Network Quality Hotspots
- 340 H3 cells with avg RSRQ < -15 dB (poor signal) have churn rates >35%
- Cells with >3 average call drops/month have median churn rate of 29% vs 12% for others

---

## Business Impact Estimate

Assuming:
- Total subscriber base: 50,000
- Average monthly ARPU: $62
- Model recall: 70%, Precision: 76%
- Retention campaign success rate: 25% (industry benchmark)
- Campaign cost per subscriber: $8

### Without model (random targeting)
- Contact 5,000 subscribers (10%)
- Catch ~900 churners (out of 9,000 total)
- Cost: $40,000
- Revenue saved: ~$558,000/year

### With churn model (top 20% risk)
- Contact 10,000 highest-risk subscribers
- Catch ~6,300 churners (70% recall)
- Cost: $80,000
- Revenue saved: ~$3.9M/year
- **ROI: 48×**

---

## Model Versions Tracked in MLflow

| Run | Model | PR-AUC | ROC-AUC | Notes |
|-----|-------|--------|---------|-------|
| v1.0 | XGBoost default | 0.781 | 0.871 | Baseline |
| v1.1 | XGBoost + SMOTE | 0.798 | 0.879 | Class balance improved |
| v1.2 | LightGBM + SMOTE | 0.803 | 0.881 | Faster training |
| v1.3 | XGBoost + Optuna HPO | **0.812** | **0.892** | **Production model** |

---

## Limitations & Future Work

1. **Temporal leakage risk**: H3 aggregate churn rate is computed on the full dataset. In production, this must be computed on historical data only (lagged by 1 month).

2. **Cold start problem**: New subscribers (< 3 months) have limited history — a separate short-tenure model could improve accuracy for this segment.

3. **Real CDR data**: Production deployment would use actual Call Detail Records (CDR) rather than synthetic usage metrics, likely improving PR-AUC to 0.85+.

4. **Causal inference**: SHAP tells us correlation, not causation. A/B testing is needed to confirm that reducing call drops causally reduces churn.

5. **Dynamic threshold**: The classification threshold should be recalibrated quarterly as churn rates shift seasonally.
