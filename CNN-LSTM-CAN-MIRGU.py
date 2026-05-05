
'''
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import re
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def parse_log(file_path: str, label: int, max_bytes: int = 8) -> pd.DataFrame:
    pattern = re.compile(
        r'\('
            r'(?P<timestamp>[\d\.]+)'
        r'\)\s+'
        r'(?P<interface>\w+)\s+'
        r'(?P<arbitration>[0-9A-Fa-f]+)#'
        r'(?P<data>[0-9A-Fa-f]+)\s+'
        r'(?P<flag>\d+)'
    )
    records = []
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            m = pattern.match(line.strip())
            if not m:
                continue
            gd = m.groupdict()
            hexstr = gd['data']
            byte_strs = [hexstr[i:i+2] for i in range(0, len(hexstr), 2)]
            byte_vals = [int(b, 16) for b in byte_strs]
            byte_vals += [0] * (max_bytes - len(byte_vals))
            rec = {f"Data{i}": byte_vals[i] for i in range(max_bytes)}
            rec['label'] = label
            records.append(rec)
    return pd.DataFrame(records)

base = r"E:\B\1\data\Car-Dataset\CAN-MIRGU-main\\CAN-MIRGU.csv"

df_benign  = parse_log(f"{base}\\Benign_day1_file1.log",  label=0)
df_masq    = parse_log(f"{base}\\masquerade_attack.log", label=1)
df_real    = parse_log(f"{base}\\real_attacks.log",     label=1)
df_suspend = parse_log(f"{base}\\suspension_attack.log", label=1)

df = pd.concat([df_benign, df_masq, df_real, df_suspend], ignore_index=True)

# Features 
feature_cols = [f"Data{i}" for i in range(8)]
X = df[feature_cols].values.astype(np.float32)
y = df["label"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train = X_train.reshape((-1, X_train.shape[1], 1))
X_test  = X_test.reshape((-1, X_test.shape[1], 1))

num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)

# CNN-LSTM model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    LSTM(100),
    Dropout(0.5),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

# Train
history = model.fit(
    X_train, y_train_cat,
    epochs=1,
    batch_size=64,
    validation_split=0.1
)

# Evaluation
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
ticks = np.arange(num_classes)
plt.xticks(ticks, [f"class{i}" for i in range(num_classes)], rotation=45)
plt.yticks(ticks, [f"class{i}" for i in range(num_classes)])
thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curves 
plt.figure(figsize=(8,6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_cat[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import itertools

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load data
csv_path = r'E:\B\1\data\Car-Dataset\CAN-MIRGU-main\\enhancement-CAN-MIRGU.csv'
df = pd.read_csv(csv_path, encoding='utf-8')

df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
for col in ['Interface', 'Flag', 'DATA']:
    df[col] = df[col].fillna('').astype('category').cat.codes

def parse_id(x):
    if isinstance(x, str):
        try:    return int(x, 16)
        except:
            try: return float(x)
            except: return 0
    return x

df['Arbitration_ID'] = df['Arbitration_ID'].apply(parse_id)

# ANOVA F-test
ext_cols = [
    'delta_time','frequency','data_mean','data_std','data_max','data_min',
    'entropy','is_all_zero','hamming_weight',
    'id_mean_period','id_std_period','rolling_dt_mean','rolling_dt_std',
    'rolling_id_entropy'
] + [f'byte_{i}_mean' for i in range(8)] + [f'byte_{i}_std' for i in range(8)]

X_all = (
    df[ext_cols]
      .replace([np.inf, -np.inf], np.nan)
      .fillna(0)
      .astype(np.float32)
      .values
)
y_all = LabelEncoder().fit_transform(df['Class'].astype(str))
class_names = LabelEncoder().fit(df['Class'].astype(str)).classes_

X_train_all, X_unused, y_train_all, _ = train_test_split(
    X_all, y_all, test_size=0.7, stratify=y_all, random_state=42
)

F_values, p_values = f_classif(X_train_all, y_train_all)
anova_df = pd.DataFrame({
    'feature': ext_cols,
    'F_value': F_values,
    'p_value': p_values
}).sort_values('F_value', ascending=False)

# top_k
top_k = 16
top_feats = anova_df['feature'].iloc[:top_k].tolist()

print("Selected top features:", top_feats)

X = df[top_feats].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).values
y = y_all

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

X_train_seq = X_train.reshape((X_train.shape[0], top_k, 1))
X_test_seq  = X_test.reshape((X_test.shape[0],  top_k, 1))

# CNN-LSTM
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu',
           input_shape=(top_k,1), padding='same'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Train
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_seq, y_train,
    validation_split=0.1,
    epochs=1,
    batch_size=32,
    callbacks=[es],
    verbose=2
)

# Evaluation
loss, acc = model.evaluate(X_test_seq, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}\n")

# Confusion Matrix
y_pred = np.argmax(model.predict(X_test_seq), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45)
plt.yticks(ticks, class_names)
thresh = cm.max()/2
for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i,cm[i,j],ha='center',va='center',
             color='white' if cm[i,j]>thresh else 'black')
plt.tight_layout()
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ROC Curve
y_true_bin  = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))
y_score_bin = model.predict(X_test_seq)
if y_true_bin.shape[1] > 1:
    plt.figure(figsize=(8,6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_score_bin[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],linestyle='--',color='gray')
    plt.title('ROC Curves')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
else:
    print("ROC skipped (only one class).")
