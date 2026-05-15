import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
 
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
 
# ==================== ЗАГРУЗКА ====================
data = np.load(r'C:\kolya\дрони\data\data DroneRF\dataset.npy', allow_pickle=True).item()
 
X = data["X"]
y = data["y_type"]
groups = data["groups"]
 
# ==================== НОРМАЛИЗАЦИЯ ====================
X = X - np.mean(X, axis=1, keepdims=True)
X = X / (np.std(X, axis=1, keepdims=True) + 1e-8)
 
X = X[..., np.newaxis]
 
# ==================== МУЛЬТИКЛАСС ====================
num_classes = len(np.unique(y))
y = tf.keras.utils.to_categorical(y, num_classes)
 
# ==================== SPLIT ====================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, np.argmax(y, axis=1), groups))
 
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
 
# ==================== ВЕСА КЛАССОВ ====================
y_train_labels = np.argmax(y_train, axis=1)
 
classes = np.unique(y_train_labels)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_labels)
 
class_weights = dict(enumerate(weights))
print("Class weights:", class_weights)
 
# ==================== МОДЕЛЬ ====================
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=X.shape[1:]),
    BatchNormalization(),
    MaxPooling1D(2),
 
    Conv1D(64, 5, activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling1D(),
 
    Dense(64, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    Dropout(0.5),
 
    Dense(num_classes, activation='softmax')
])
 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
 
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
 
model.summary()
 
# ==================== ОБУЧЕНИЕ ====================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=64,
    class_weight=class_weights,
    verbose=1
)
 
# ==================== ПРЕДСКАЗАНИЯ ====================
y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
 
# ==================== ОЦЕНКА ====================
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
 
# ==================== ГРАФИК ОБУЧЕНИЯ ====================
plt.figure()
 
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
 
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
 
plt.legend()
plt.title("История обучения")
 
plt.savefig("график тип.png", dpi=300)
plt.show()
 
# ==================== МАТРИЦА ОШИБОК ====================
labels = [f"Класс {i}" for i in range(num_classes)]
 
plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_true, y_pred),
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)
 
plt.xlabel("Предсказанный")
plt.ylabel("Настоящий")
plt.title("Матрица ошибок")
 
plt.savefig("матрица тип.png", dpi=300)
plt.show()
 
# ==================== СОХРАНЕНИЕ ====================
model.save("droni_multiclass.keras")