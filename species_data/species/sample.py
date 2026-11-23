import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import time
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

train_data=np.load('species_train_clean_all.npz')
train_locs=train_data['train_locs']
train_ids=train_data['train_ids']
taxon_ids=train_data['taxon_ids']
taxon_names=train_data['taxon_names']

df=pd.DataFrame({
    'id': train_ids,
    'lat': train_locs[:,0],
    'lon': train_locs[:,1]
})


unique_pairs = df.drop_duplicates(
    subset=['lat', 'lon', 'id']
)

Y_df = unique_pairs.pivot_table(
    index=['lat', 'lon'],
    columns='id',
    values='id',
    aggfunc='count',
    fill_value=0
)

# Y_df = (Y_df > 0).astype(np.uint8)
Y_df = Y_df.reset_index()
X = Y_df[['lat', 'lon']].to_numpy()
Y = Y_df.iloc[:, 2:].to_numpy()
msss = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, val_idx = next(msss.split(X, Y))

X_train, X_val = X[train_idx], X[val_idx]
Y_train, Y_val = Y[train_idx], Y[val_idx]

baseline_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000
        ),
        n_jobs=-1
    ))
])



baseline_model.fit(X_train, Y_train)
Y_val_proba = baseline_model.predict_proba(X_val)

for t in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
    Y_pred = (Y_val_proba >= t).astype(np.uint8)
    micro_p = precision_score(Y_val, Y_pred, average='micro', zero_division=0)
    micro_r = recall_score(Y_val, Y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(Y_val, Y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(Y_val, Y_pred, average='macro', zero_division=0)
    print(f"t={t:<6} predicted positives={Y_pred.sum():<7} "
          f"micro P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f} | macro F1={macro_f1:.3f}")


# print("Probability matrix:", Y_val_proba.shape)
# print("Minimum probability:", Y_val_proba.min())
# print("Maximum probability:", Y_val_proba.max())

Y_val_pred = (Y_val_proba >= 0.5).astype(np.uint8)

micro_precision = precision_score(Y_val, Y_val_pred, average='micro', zero_division=0)
micro_recall = recall_score(Y_val, Y_val_pred, average='micro', zero_division=0)
micro_f1 = f1_score(Y_val, Y_val_pred, average='micro', zero_division=0)

macro_precision = precision_score(Y_val, Y_val_pred, average='macro', zero_division=0)
macro_recall = recall_score(Y_val, Y_val_pred, average='macro', zero_division=0)
macro_f1 = f1_score(Y_val, Y_val_pred, average='macro', zero_division=0)

print("Micro Precision:", micro_precision)
print("Micro Recall:", micro_recall)
print("Micro F1:", micro_f1)

print("\nMacro Precision:", macro_precision)
print("Macro Recall:", macro_recall)
print("Macro F1:", macro_f1)






