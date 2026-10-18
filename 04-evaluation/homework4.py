"""Homework 4 - Evaluation Metrics for Classification (ML Zoomcamp 2026)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "data" / "course_lead_scoring_2026.csv")

cat = df.select_dtypes(include="object").columns.tolist()
num = [c for c in df.select_dtypes(exclude="object").columns if c != "converted"]
df[cat] = df[cat].fillna("NA")
df[num] = df[num].fillna(0.0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
y_train = df_train.converted.values
y_val = df_val.converted.values

# ---- Q1
print("Q1 AUC per numerical feature (train):")
aucs = {}
for c in num:
    a = roc_auc_score(y_train, df_train[c])
    if a < 0.5:
        a = roc_auc_score(y_train, -df_train[c])
    aucs[c] = a
    print(f"  {c}: {a:.4f}")
print("  highest:", max(aucs, key=aucs.get))

# ---- Q2
features = cat + num


def train(df_tr, y_tr, C=1.0):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(df_tr[features].to_dict(orient="records"))
    m = LogisticRegression(solver="liblinear", C=C, max_iter=1000).fit(X, y_tr)
    return dv, m


def predict(dv, m, df_):
    return m.predict_proba(dv.transform(df_[features].to_dict(orient="records")))[:, 1]


dv, model = train(df_train, y_train)
y_pred = predict(dv, model, df_val)
print("\nQ2 validation AUC:", round(roc_auc_score(y_val, y_pred), 3))

# ---- Q3 / Q4
rows = []
for t in np.arange(0.0, 1.01, 0.01):
    pred = y_pred >= t
    tp = ((pred == 1) & (y_val == 1)).sum()
    fp = ((pred == 1) & (y_val == 0)).sum()
    fn = ((pred == 0) & (y_val == 1)).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    rows.append((round(t, 2), p, r, f1))
sc = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1"])

valid = sc[(sc.precision > 0) | (sc.recall > 0)].copy()
valid["diff"] = (valid.precision - valid.recall).abs()
print("Q3 precision/recall intersect at:", valid.loc[valid["diff"].idxmin(), "threshold"])
print("Q4 max F1 at threshold:", sc.loc[sc.f1.idxmax(), "threshold"],
      f"(F1={sc.f1.max():.4f})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(sc.threshold, sc.precision, label="precision")
ax[0].plot(sc.threshold, sc.recall, label="recall")
ax[0].set(xlabel="threshold", title="Precision & recall"); ax[0].legend()
ax[1].plot(sc.threshold, sc.f1, color="green")
ax[1].set(xlabel="threshold", title="F1")
fig.tight_layout(); fig.savefig(HERE / "precision_recall_f1.png", dpi=120)

# ---- Q5
kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []
for tr, va in kf.split(df_full_train):
    d_tr, d_va = df_full_train.iloc[tr], df_full_train.iloc[va]
    dv_, m_ = train(d_tr, d_tr.converted.values)
    scores.append(roc_auc_score(d_va.converted.values, predict(dv_, m_, d_va)))
print("\nQ5 fold AUCs:", np.round(scores, 4), "std:", round(np.std(scores), 3))

# ---- Q6
print("\nQ6 CV over C:")
res = {}
for C in [0.000001, 0.001, 1]:
    s = []
    for tr, va in kf.split(df_full_train):
        d_tr, d_va = df_full_train.iloc[tr], df_full_train.iloc[va]
        dv_, m_ = train(d_tr, d_tr.converted.values, C)
        s.append(roc_auc_score(d_va.converted.values, predict(dv_, m_, d_va)))
    res[C] = (round(np.mean(s), 3), round(np.std(s), 3))
    print(f"  C={C}: mean={res[C][0]} std={res[C][1]}")
print("  best C:", min(res, key=lambda c: (-res[c][0], res[c][1], c)))
