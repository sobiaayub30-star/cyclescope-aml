# 🔁 CycleScope: Graph-Based Detection of Circular Money Laundering

**Can loops in the transaction graph catch money laundering that an ordinary fraud model misses?**

Money launderers often move money **in a circle** — A → B → C → A — so that every single transfer
looks normal. This project finds such loops in the transaction graph (accounts = points,
transfers = arrows), scores each loop, and gives those scores to an XGBoost model.

👉 **Live app:** _add your `https://….streamlit.app` link here after deploying_

## Result (IBM AML benchmark, test period, same alert budget for every model)

| Model | Cycle transactions caught | Laundering cycles caught (of 32) | Recall | Precision |
|---|---|---|---|---|
| Baseline (ordinary features) | 60% | 25.0 | 53% | 50% |
| Hybrid, real-time | 62% | 26.8 | 54% | 51% |
| **Hybrid, retrospective** | **73%** | **29.8** | **55%** | **52%** |

- **Monitoring (looking back):** loop features catch clearly more laundering cycles — a
  **statistically proven** gain (paired sign test p = 0.001, gain in every one of 5 seeds).
- **Real time:** the gain is small and **not proven** — a loop only becomes visible when it closes.
- **Negative control (PaySim, no loops):** all three models identical, as expected.

## How it works

1. **Data** — a synthetic dataset with 15 planted rings (to test the method), PaySim (negative
   control) and the IBM AML benchmark HI-Small (main test, ~5 million transactions).
2. **Loop detector** — a search written for this project (Python + numba) finds money loops within
   a time window and scores them: *amount-CV* (how equal the amounts are), *time span*, *length*.
   It found 84% of IBM's test laundering cycles.
3. **No peeking** — earlier 70% of the data to learn, later 30% to judge; every feature uses only
   information available at its time (proven by tests).
4. **Models** — three XGBoost models, identical except for their features, 5 seeds each, compared at
   the same number of alerts with a paired statistical test.

## The app

An investigator's screen with four pages: **Overview**, **Alert queue** (rank by any model),
**Investigate a cycle** (the whole loop with amounts and times, each model's rank, and a SHAP
explanation of why a transfer was flagged) and **How it works**.

The app only reads a small pre-computed bundle in `app_data/` (built by Week 5 of the project
notebook), so it runs on free hosting.

### Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limits

IBM's benchmark is itself synthetic (real bank data with laundering labels is not public), and its
test period holds only 32 laundering cycles. **Future work:** a graph neural network, the larger IBM
datasets, and real-time features for loops that are still open.

## Tools

Python · pandas · NumPy · numba · XGBoost (with built-in SHAP) · scikit-learn · Plotly · Streamlit ·
Google Colab · Kaggle datasets: PaySim, IBM Transactions for Anti Money Laundering
