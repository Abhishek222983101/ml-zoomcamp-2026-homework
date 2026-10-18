"""Homework 3 - Machine Learning for Classification (ML Zoomcamp 2026)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / "data" / "course_lead_scoring_2026.csv"
df = pd.read_csv(DATA)

# ---- Data preparation
print("Missing values:\n", df.isnull().sum(), "\n")
cat = df.select_dtypes(include="object").columns.tolist()
num = [c for c in df.select_dtypes(exclude="object").columns if c != "converted"]
df[cat] = df[cat].fillna("NA")
df[num] = df[num].fillna(0.0)

# ---- Q1
print("Q1 mode of industry:", df["industry"].mode()[0])

# ---- Q2
corr = df[num].corr()
print("\nQ2 correlation matrix:\n", corr)
pairs = [("interaction_count", "lead_score"),
         ("number_of_courses_viewed", "lead_score"),
         ("number_of_courses_viewed", "interaction_count"),
         ("annual_income", "interaction_count")]
for a, b in pairs:
    print(f"  {a} / {b}: {corr.loc[a, b]:.4f}")

# ---- Split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
y_train = df_train.pop("converted").values
y_val = df_val.pop("converted").values
del df_test["converted"]

# ---- Q3
print("\nQ3 mutual information (train):")
mi = {c: round(mutual_info_score(df_train[c], y_train), 2) for c in cat}
print(mi, "-> biggest:", max(mi, key=mi.get))


def fit_acc(features, C=1.0):
    dv = DictVectorizer(sparse=False)
    X_tr = dv.fit_transform(df_train[features].to_dict(orient="records"))
    X_va = dv.transform(df_val[features].to_dict(orient="records"))
    m = LogisticRegression(solver="liblinear", C=C, max_iter=1000, random_state=42)
    m.fit(X_tr, y_train)
    return accuracy_score(y_val, m.predict(X_va))


features = cat + num
# ---- Q4
base = fit_acc(features)
print("\nQ4 accuracy:", round(base, 2), f"(raw {base})")

# ---- Q5
print("\nQ5 feature elimination:")
diffs = {}
for f in features:
    diffs[f] = base - fit_acc([x for x in features if x != f])
    print(f"  without {f}: diff = {diffs[f]:.6f}")
opts = ["lead_source", "number_of_courses_viewed", "interaction_count"]
print("  smallest diff among options:", min(opts, key=lambda f: diffs[f]))

# ---- Q6
print("\nQ6 regularisation:")
scores = {C: round(fit_acc(features, C), 3) for C in [0.000001, 0.00001, 0.0001, 0.001]}
print(scores)
best = max(scores.values())
print("  best C (smallest on tie):", min(C for C, s in scores.items() if s == best))
