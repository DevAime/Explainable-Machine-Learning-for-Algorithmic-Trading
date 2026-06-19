# model.py
# DSA3900 Research Prototype — ML Pipeline
# Explainable Machine Learning for Algorithmic Trading
# Author: Aime Muganga (670232)

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.model_selection import GridSearchCV
import shap

# ─────────────────────────────────────────────────────────────
# 1. LOAD AND CLEAN DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("spy_data.csv", index_col=0, parse_dates=True)
df.index.name = "Date"
df.columns = [c.strip() for c in df.columns]
df = df.sort_index()
df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
df.dropna(inplace=True)
print(f"Data loaded: {df.shape[0]} rows | {df.index.min().date()} to {df.index.max().date()}")

# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("Engineering features...")

close = df["Close"]

# Simple and Exponential Moving Averages
df["SMA_10"] = close.rolling(10).mean()
df["SMA_20"] = close.rolling(20).mean()
df["SMA_50"] = close.rolling(50).mean()
df["EMA_10"] = close.ewm(span=10, adjust=False).mean()
df["EMA_20"] = close.ewm(span=20, adjust=False).mean()

# RSI (14)
delta = close.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# MACD (12, 26, 9)
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

# Bollinger Bands (20, 2)
sma20 = close.rolling(20).mean()
std20 = close.rolling(20).std()
df["BB_Upper"] = sma20 + 2 * std20
df["BB_Lower"] = sma20 - 2 * std20
df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
df["BB_Position"] = (close - df["BB_Lower"]) / df["BB_Width"]  # 0=at lower, 1=at upper

# Rolling Volatility (10-day std of daily returns)
daily_returns = close.pct_change()
df["Volatility_10"] = daily_returns.rolling(10).std()

# Lagged Returns
df["Return_1d"] = daily_returns
df["Return_3d"] = close.pct_change(3)
df["Return_5d"] = close.pct_change(5)

# Price relative to moving averages (momentum signals)
df["Price_SMA20_Ratio"] = close / df["SMA_20"]
df["Price_SMA50_Ratio"] = close / df["SMA_50"]

# ─────────────────────────────────────────────────────────────
# 3. TARGET VARIABLE
# ─────────────────────────────────────────────────────────────
# 1 = next day close > today close (BUY signal), 0 = SELL signal
df["Target"] = (close.shift(-1) > close).astype(int)

# Drop the last row (no next-day label) and all NaNs from indicators
df.dropna(inplace=True)
print(f"After feature engineering: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Class balance — BUY: {df['Target'].sum()} | SELL: {(df['Target']==0).sum()}")

# ─────────────────────────────────────────────────────────────
# 4. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "SMA_10", "SMA_20", "SMA_50", "EMA_10", "EMA_20",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "BB_Position",
    "Volatility_10",
    "Return_1d", "Return_3d", "Return_5d",
    "Price_SMA20_Ratio", "Price_SMA50_Ratio"
]

X = df[FEATURE_COLS]
y = df["Target"]

n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train, y_train = X.iloc[:train_end],  y.iloc[:train_end]
X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test,  y_test  = X.iloc[val_end:],    y.iloc[val_end:]

print(f"\nSplit sizes — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Scale features (needed for Logistic Regression; harmless for trees)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────────────────────
print("\nTraining models...")

# --- Logistic Regression (baseline) ---
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)

# --- Random Forest with basic hyperparameter tuning ---
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0
)
rf_grid.fit(X_train, y_train)  # trees don't need scaled data
rf = rf_grid.best_estimator_
print(f"  Best RF params: {rf_grid.best_params_}")

# --- Gradient Boosting with basic hyperparameter tuning ---
gb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1]
}
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0
)
gb_grid.fit(X_train, y_train)
gb = gb_grid.best_estimator_
print(f"  Best GB params: {gb_grid.best_params_}")

# ─────────────────────────────────────────────────────────────
# 6. EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("MODEL EVALUATION ON TEST SET")
print("="*65)

models = {
    "Logistic Regression": (lr,  X_test_sc),
    "Random Forest":       (rf,  X_test),
    "Gradient Boosting":   (gb,  X_test),
}

results = {}
for name, (model, X_t) in models.items():
    y_pred  = model.predict(X_t)
    y_proba = model.predict_proba(X_t)[:, 1]
    results[name] = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "F1":        round(f1_score(y_test, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
    }

results_df = pd.DataFrame(results).T
print(results_df.to_string())
print("="*65)

# Select best model by ROC-AUC
best_name = results_df["ROC-AUC"].idxmax()
best_model = models[best_name][0]
best_X_test = models[best_name][1]
print(f"\nBest model: {best_name} (ROC-AUC = {results_df.loc[best_name, 'ROC-AUC']})")

# ─────────────────────────────────────────────────────────────
# 7. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────
print("\nComputing SHAP values...")

# Use TreeExplainer for tree-based models, LinearExplainer for LR
if best_name == "Logistic Regression":
    explainer   = shap.LinearExplainer(best_model, X_train_sc)
    shap_values = explainer.shap_values(X_test_sc)
    shap_array  = shap_values  # already 2D for binary
else:
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    # For tree models, shap_values may be list [class0, class1] — take class 1
    if isinstance(shap_values, list):
        shap_array = shap_values[1]
    elif shap_values.ndim == 3:
        # New SHAP versions return shape (samples, features, classes) — take class 1
        shap_array = shap_values[:, :, 1]
    else:
        shap_array = shap_values

# Global feature importance plot
mean_abs_shap = np.abs(shap_array).mean(axis=0)
feat_importance = pd.Series(mean_abs_shap, index=FEATURE_COLS).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_importance.head(15).plot(kind="barh", color="steelblue")
plt.gca().invert_yaxis()
plt.title(f"Global Feature Importance (SHAP) — {best_name}", fontsize=13)
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig("shap_global.png", dpi=150)
plt.close()
print("  Saved: shap_global.png")

# Local SHAP explanation for a single prediction (first test instance)
sample_idx = 0
plt.figure(figsize=(10, 4))
shap_series = pd.Series(shap_array[sample_idx], index=FEATURE_COLS).sort_values()
colors = ["#d73027" if v < 0 else "#1a9850" for v in shap_series]
shap_series.plot(kind="barh", color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title(f"Local SHAP Explanation — Sample {val_end + sample_idx} | "
          f"Prediction: {'BUY' if best_model.predict(best_X_test[:1])[0]==1 else 'SELL'}", fontsize=12)
plt.xlabel("SHAP value (impact on model output)")
plt.tight_layout()
plt.savefig("shap_local.png", dpi=150)
plt.close()
print("  Saved: shap_local.png")

# ─────────────────────────────────────────────────────────────
# 8. EXPORT ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("\nExporting artifacts...")

# Save best model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Build test results CSV with predictions and SHAP values
test_df = X_test.copy()
test_df["Close"]      = df.loc[X_test.index, "Close"]
test_df["Date"]       = X_test.index
test_df["Actual"]     = y_test.values
test_df["Prediction"] = best_model.predict(best_X_test)
test_df["Proba_BUY"]  = best_model.predict_proba(best_X_test)[:, 1]

# Add SHAP values for each feature
shap_df = pd.DataFrame(shap_array, columns=[f"SHAP_{c}" for c in FEATURE_COLS], index=X_test.index)
test_df = pd.concat([test_df, shap_df], axis=1)
test_df.to_csv("test_results.csv", index=False)
print("  Saved: test_results.csv")

# Save top 5 features
top5 = feat_importance.head(5).index.tolist()
with open("top5_features.pkl", "wb") as f:
    pickle.dump(top5, f)
print(f"  Top 5 features: {top5}")

print("\nPipeline complete. Ready to run app.py.")