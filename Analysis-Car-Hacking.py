import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing     import LabelEncoder
from sklearn.ensemble          import RandomForestClassifier
from sklearn.feature_selection import f_classif, VarianceThreshold
from sklearn.inspection        import permutation_importance

from collections import defaultdict
from lime.lime_tabular import LimeTabularExplainer
from tqdm              import tqdm
import shap
os.environ['CUDA_VISIBLE_DEVICES'] = ''

csv_path = r'C:\\Users\\whf80\\Desktop\\enhanced-Car-Hacking.csv'
df = pd.read_csv(csv_path, encoding='utf-8')

ext_cols = [
    'frequency', 'data_mean', 'data_std', 'data_max', 'data_min',
    'entropy', 'hamming_weight',
    'id_mean_period', 'id_std_period', 'rolling_dt_mean', 'rolling_dt_std',
    'rolling_id_entropy',
    'byte_0_mean', 'byte_0_std', 'byte_1_mean', 'byte_1_std',
    'byte_2_mean', 'byte_2_std', 'byte_3_mean', 'byte_3_std',
    'byte_4_mean', 'byte_4_std', 'byte_5_mean', 'byte_5_std',
    'byte_6_mean', 'byte_6_std', 'byte_7_mean', 'byte_7_std'
]

X = (df[ext_cols]
       .replace([np.inf, -np.inf], np.nan)
       .fillna(0)
       .astype(np.float32)
       .values)

le = LabelEncoder()
y = le.fit_transform(df['Class'].astype(str))
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf.feature_names_in_ = np.array(ext_cols)

# ANOVA F-test 
F_values, p_values = f_classif(X_train, y_train)

indices = np.argsort(F_values)

s = pd.Series(F_values, index=ext_cols)

s_sorted = s.sort_values(ascending=False, na_position='last')

plt.figure(figsize=(10, 8))
plt.barh(s_sorted.index, s_sorted.values, align='center')
plt.gca().invert_yaxis()
plt.xlabel('F-value')
plt.tight_layout()
plt.show()

# Permutation Importance 
result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring='accuracy'
)

# sort features by importance
importances = result.importances_mean
stds        = result.importances_std
indices     = np.argsort(importances)[::-1]

# print top 10
print("Top 10 features by permutation importance:")
for i in indices[:10]:
    print(f"{ext_cols[i]:<20}  {importances[i]:.4f} ± {stds[i]:.4f}")

# plot all features
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(
    y=np.arange(len(ext_cols)),
    width=importances[indices],
    xerr=stds[indices],
    align='center'
)
ax.set_yticks(np.arange(len(ext_cols)))
ax.set_yticklabels([ext_cols[i] for i in indices])
ax.invert_yaxis()
ax.set_xlabel("Decrease in accuracy")

ax.set_xlim(left=0)

plt.tight_layout()
plt.show()

lime_explainer = LimeTabularExplainer(
    training_data        = X_train,
    feature_names        = ext_cols,
    class_names          = class_names,
    mode                 = 'classification',
    discretize_continuous= True
)

# LIME 
n_total = len(X_test)
n_sample = max(1, int(0.05 * n_total))         
rng = np.random.RandomState(42)                
sample_indices = rng.choice(n_total, size=n_sample, replace=False)

agg_weights = defaultdict(list)

for idx in tqdm(sample_indices, desc='LIME Progress', unit='sample'):
    exp = lime_explainer.explain_instance(
        data_row    = X_test[idx],
        predict_fn  = rf.predict_proba,
        num_features= len(ext_cols)
    )
    class_map = exp.as_map()[1]
    for feat_idx, weight in class_map:
        feat_name = ext_cols[feat_idx]
        agg_weights[feat_name].append(abs(weight))

mean_abs_weights = {feat: np.mean(ws) for feat, ws in agg_weights.items()}

# Visualize 
sorted_feats = sorted(mean_abs_weights, key=lambda f: mean_abs_weights[f], reverse=True)
sorted_vals  = [mean_abs_weights[f] for f in sorted_feats]

plt.figure(figsize=(10, 8))
y_pos = np.arange(len(sorted_feats))
plt.barh(y_pos, sorted_vals, align='center')
plt.yticks(y_pos, sorted_feats, fontsize=10)
plt.gca().invert_yaxis()
plt.xlabel('LIME weight')
plt.tight_layout()
plt.show()


# SHAP 
explainer = shap.TreeExplainer(rf)

rng = np.random.RandomState(42)
n_samples = max(1, int(0.001 * X_test.shape[0]))
indices = rng.choice(X_test.shape[0], size=n_samples, replace=False)
X_sub = X_test[indices]

shap_values_sub = np.zeros_like(X_sub)
batch_size = 100
for start in tqdm(range(0, X_sub.shape[0], batch_size), desc="Computing SHAP values", unit="batch"):
    end = min(start + batch_size, X_sub.shape[0])
    batch = X_sub[start:end]
    shap_batch = explainer.shap_values(batch)[1]  # class=1 SHAP value 
    shap_values_sub[start:end] = shap_batch

assert shap_values_sub.shape == X_sub.shape, f"SHAP {shap_values_sub.shape} vs data {X_sub.shape}"

shap.summary_plot(
    shap_values_sub,
    X_sub,
    feature_names=ext_cols,
    plot_type="bar",
    show=True
)
