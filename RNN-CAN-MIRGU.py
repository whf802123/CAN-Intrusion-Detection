
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing    import LabelEncoder
from sklearn.utils            import class_weight
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve
)

from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import SimpleRNN, Dense
from tensorflow.keras.callbacks  import EarlyStopping

csv_path = r'C:\\Users\\whf80\\Desktop\\Car-Dataset\\CAN-MIRGU-main\\enhancement-CAN-MIRGU.csv'
df = pd.read_csv(csv_path, encoding='utf-8')

def parse_id(x):
    if isinstance(x, str):
        try:   return int(x, 16)
        except:
            try: return float(x)
            except: return 0
    return x
df['Arbitration_ID'] = df['Arbitration_ID'].apply(parse_id)

# Features list
ext_cols = [
    'frequency','data_mean','data_std','data_max','data_min',
    'entropy','is_all_zero','hamming_weight',
    'id_mean_period','id_std_period','rolling_dt_mean','rolling_dt_std',
    'rolling_id_entropy'
] + [f'byte_{i}_mean' for i in range(8)] + [f'byte_{i}_std' for i in range(8)]
feature_cols = ext_cols

X = ( df[feature_cols]
      .replace([np.inf, -np.inf], np.nan)
      .fillna(0)
      .astype(np.float32)
      .values )
y = LabelEncoder().fit_transform(df['Class'].astype(str))
class_names = LabelEncoder().fit(df['Class'].astype(str)).classes_

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

from sklearn.feature_selection import f_classif
F, _ = f_classif(X_train, y_train)
top_k = min(40, X_train.shape[1])
top_feats = [feature_cols[i] for i in np.argsort(F)[::-1][:top_k]]

Xtr = pd.DataFrame(X_train, columns=feature_cols)[top_feats].values
Xte = pd.DataFrame(X_test,  columns=feature_cols)[top_feats].values

# RNN reshape：samples × timesteps × features
timesteps = Xtr.shape[1]
Xtr_rnn = Xtr.reshape((-1, timesteps, 1))
Xte_rnn = Xte.reshape((-1, timesteps, 1))

# class_weight='balanced'
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {cls: float(w) for cls, w in zip(np.unique(y_train), weights)}

# Train RNN
model = Sequential([
    SimpleRNN(64, activation='tanh', input_shape=(timesteps, 1)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
model.fit(
    Xtr_rnn, y_train,
    validation_split=0.1,
    epochs=1, batch_size=256,
    class_weight=class_weight_dict,
    callbacks=[es],
    verbose=2
)

y_prob = model.predict(Xte_rnn).ravel()
precision, recall, thresh = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best = np.nanargmax(f1_scores)
best_thresh = thresh[best]
print(f"Best threshold by weighted F1: {best_thresh:.3f}, F1={f1_scores[best]:.3f}")

y_pred = (y_prob >= best_thresh).astype(int)

acc   = accuracy_score(y_test, y_pred)
precw = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recw  = recall_score(y_test, y_pred,    average='weighted', zero_division=0)
f1w   = f1_score(y_test, y_pred,        average='weighted', zero_division=0)

print(f"\nWeighted metrics on test set:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precw:.4f}")
print(f"Recall:    {recw:.4f}")
print(f"F1-score:  {f1w:.4f}\n")

print("Classification Report:")
print(classification_report(
    y_test, y_pred, target_names=class_names, zero_division=0, digits=4
))

# Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('True')
for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = 'white' if cm[i,j]>cm.max()/2 else 'black'
    plt.text(j, i, cm[i,j], ha='center', va='center', color=color)
plt.xticks([0,1], class_names, rotation=45)
plt.yticks([0,1], class_names)
plt.tight_layout()
plt.show()

'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import itertools
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.layers import SimpleRNN, LSTM
from tensorflow.keras.models import Sequential

# Load data 
csv_path = r"C:\\Users\\whf80\\Desktop\\Car-Dataset\\CAN-MIRGU-main\\CAN-MIRGU.csv"
df = pd.read_csv(csv_path)

# Analyze Arbitration_ID 
def parse_id(x):
    if isinstance(x, str):
        try:
            return int(x, 16)
        except ValueError:
            try:
                return float(x)
            except:
                return 0
    return x

df['Arbitration_ID'] = df['Arbitration_ID'].apply(parse_id)

for i in range(8):
    df[f'Data{i}'] = (
        df['DATA'].fillna('').astype(str).str.split().str[i]
        .apply(lambda x: int(x, 16) if isinstance(x, str) and x else 0)
    )

feature_cols = ['Arbitration_ID']
if 'DLC' in df.columns:
    feature_cols.append('DLC')
feature_cols += [f'Data{i}' for i in range(8)]

X = df[feature_cols].astype(np.float32).values

le = LabelEncoder()
y = le.fit_transform(df['Class'].astype(str))
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,
    stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

timesteps = X_train.shape[1]   # = 特征数，比如 9 或 10
X_train_rnn = X_train.reshape(-1, timesteps, 1)
X_test_rnn  = X_test.reshape(-1, timesteps, 1)

# RNN 
model = Sequential([
    SimpleRNN(64, activation='tanh', input_shape=(timesteps, 1)),
    Dense(64, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Train 
model.fit(
    X_train_rnn, y_train,
    epochs=10, batch_size=256,
    validation_split=0.1
)

y_prob = model.predict(X_test_rnn)
y_pred = np.argmax(y_prob, axis=1)

# Evaluation 
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred) 
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
thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ROC Curve 
y_true_bin  = label_binarize(y_test,  classes=range(len(class_names)))
y_score_bin = label_binarize(y_pred, classes=range(len(class_names)))
if y_true_bin.shape[1] > 1:
    plt.figure(figsize=(8,6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.title('ROC Curves')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
else:
    print("ROC skipped: only one class present.")
'''
