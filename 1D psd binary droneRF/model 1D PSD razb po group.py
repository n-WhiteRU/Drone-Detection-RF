import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
 
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
 
# ==================== SPLIT ====================
def stratified_group_split_advanced(X, y, groups, test_size=0.2, max_iter=200):
 
    unique_classes = set(np.unique(y))
    unique_groups = np.unique(groups)
 
    group_to_classes = {}
    for g in unique_groups:
        group_to_classes[g] = set(y[groups == g])
 
    best_split = None
    best_score = -1
 
    rng = np.random.RandomState(42)
 
    for i in range(max_iter):
 
        rng.shuffle(unique_groups)
 
        test_groups = []
        train_groups = []
 
        test_classes = set()
 
        for g in unique_groups:
            if len(test_groups) / len(unique_groups) < test_size:
                test_groups.append(g)
                test_classes |= group_to_classes[g]
            else:
                train_groups.append(g)
 
        train_classes = set()
        for g in train_groups:
            train_classes |= group_to_classes[g]
 
        train_ok = unique_classes.issubset(train_classes)
        test_ok  = unique_classes.issubset(test_classes)
 
        score = len(test_classes)
 
        if train_ok and test_ok:
            print(f"✔ Идеальный split найден на итерации {i}")
            break
 
        if train_ok and score > best_score:
            best_score = score
            best_split = (train_groups.copy(), test_groups.copy())
 
    if not (train_ok and test_ok):
        print("⚠ Используем лучший найденный split")
        train_groups, test_groups = best_split
 
    train_idx = np.where(np.isin(groups, train_groups))[0]
    test_idx  = np.where(np.isin(groups, test_groups))[0]
 
    return train_idx, test_idx
 
# ==================== ЗАГРУЗКА ====================
data = np.load(r'C:\kolya\дрони\data\data DroneRF\dataset.npy', allow_pickle=True).item()
 
X = data["X"]
y_binary = data["y_binary"]
y_full = data["y_full"]
groups = data["groups"]
 
# ==================== НОРМАЛИЗАЦИЯ ====================
X = X - np.mean(X, axis=1, keepdims=True)
X = X / (np.std(X, axis=1, keepdims=True) + 1e-8)
X = X[..., np.newaxis]
 
# ==================== SPLIT ====================
train_idx, test_idx = stratified_group_split_advanced(X, y_full, groups)
 
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_binary[train_idx], y_binary[test_idx]
y_full_test = y_full[test_idx]
 
# ==================== ВЕСА КЛАССОВ ====================
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: weights[0]*1.5, 1: weights[1]}
 
# ==================== МОДЕЛЬ ====================
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=X.shape[1:]),
    BatchNormalization(),
    MaxPooling1D(2),
 
    Conv1D(64, 5, activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling1D(),
 
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    Dropout(0.5),
 
    Dense(1, activation='sigmoid')
])
 
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
 
# ==================== ОБУЧЕНИЕ ====================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=64,
    class_weight=class_weights,
    verbose=1
)
 
# ==================== СОХРАНЕНИЕ HISTORY ====================
pd.DataFrame(history.history).to_csv("history.csv", index=False)
 
# ==================== ПРЕДСКАЗАНИЯ ====================
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)
 
# ==================== GLOBAL ====================
global_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "auc": roc_auc_score(y_test, y_prob)
}
 
print("\n=== GLOBAL METRICS ===")
for k, v in global_metrics.items():
    print(f"{k}: {v:.4f}")
 
pd.DataFrame([global_metrics]).to_csv("global_metrics.csv", index=False)
 
# ==================== PER CLASS ====================
per_class = []
 
num_classes = 13
 
for cls in range(num_classes):
 
    y_true_cls = (y_full_test == cls).astype(int)
 
    # если класс вообще отсутствует в тесте
    if np.sum(y_true_cls) == 0:
        metrics = {
            "class": cls,
            "samples": 0,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "auc": np.nan
        }
        per_class.append(metrics)
        continue
 
    # бинарное предсказание уже есть (дрон/фон)
    # но для каждого класса считаем отдельно
    try:
        auc = roc_auc_score(y_true_cls, y_prob)
    except:
        auc = np.nan
 
    metrics = {
        "class": cls,
        "samples": int(np.sum(y_true_cls)),
        "accuracy": accuracy_score(y_true_cls, y_pred),
        "precision": precision_score(y_true_cls, y_pred, zero_division=0),
        "recall": recall_score(y_true_cls, y_pred),
        "f1": f1_score(y_true_cls, y_pred),
        "auc": auc
    }
 
    per_class.append(metrics)
 
df_per_class = pd.DataFrame(per_class)
df_per_class.to_csv("per_class_metrics.csv", index=False)
 
print("\n=== PER CLASS METRICS ===")
print(df_per_class)
 
# ==================== CONFUSION MATRIX ====================
cm = confusion_matrix(y_test, y_pred)
 
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Фон", "БПЛА"],
            yticklabels=["Фон", "БПЛА"])
 
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
 
# ==================== ГРАФИК ====================
plt.figure()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("Training history")
plt.savefig("training_plot.png", dpi=300)
plt.show()
 
# ==================== СОХРАНЕНИЕ ====================
model.save("cnn_binary.keras")