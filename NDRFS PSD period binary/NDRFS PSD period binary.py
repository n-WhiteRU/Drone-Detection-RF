import numpy as np
import torch
import scipy.signal
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================
DATASET_PATH = Path(r'C:\kolya\дрони\dataset NDRFS\dataset.pt')
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Файл {DATASET_PATH} не найден. Укажите правильный путь.")

data = torch.load(DATASET_PATH)
x_iq = data['x_iq'].numpy()          # (n_samples, 2, seq_len)
y_original = data['y'].numpy()       # мультиклассовые метки (0..6 и т.д.)

print(f"Загружено: X_iq {x_iq.shape}, y {y_original.shape}")
print(f"Исходные классы: {np.unique(y_original)}")

# ============================
# 2. БИНАРНЫЕ МЕТКИ (фон = класс 4 -> 0, дроны = всё остальное -> 1)
# ============================
BACKGROUND_CLASS = 4
y_binary = np.where(y_original == BACKGROUND_CLASS, 0, 1)

print(f"Бинарное распределение: фон (0): {np.sum(y_binary==0)}, дроны (1): {np.sum(y_binary==1)}")

# ============================
# 3. IQ → PSD методом периодиограмм
# ============================
def iq_to_psd(iq_batch, fs=1.0, nperseg=256, noverlap=None, nfft=None):
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg
    n_samples = iq_batch.shape[0]
    psd_list = []
    for i in range(n_samples):
        complex_sig = iq_batch[i, 0, :] + 1j * iq_batch[i, 1, :]
        f, Pxx = scipy.signal.periodogram(complex_sig, fs=fs, window='hann', nfft=nfft, scaling='density')
        psd_list.append(Pxx)
    return np.array(psd_list), f

NPERSEG = 256
NFFT = 256
NOVERLAP = NPERSEG // 2

print("Вычисление PSD...")
X_psd, freqs = iq_to_psd(x_iq, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT)
print(f"PSD форма: {X_psd.shape}")

# ============================
# 4. НОРМАЛИЗАЦИЯ (центрирование + масштабирование каждого образца)
# ============================
X_psd = X_psd - np.mean(X_psd, axis=1, keepdims=True)
X_psd = X_psd / (np.std(X_psd, axis=1, keepdims=True) + 1e-8)
X_psd = X_psd[..., np.newaxis]   # (samples, freq_bins, 1)

# ============================
# 5. СТРАТИФИЦИРОВАННОЕ РАЗДЕЛЕНИЕ (по бинарным меткам)
# ============================
num_classes = 2
y_cat = tf.keras.utils.to_categorical(y_binary, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X_psd, y_cat, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print("Train distribution (фон/дрон):", np.bincount(np.argmax(y_train, axis=1)))
print("Test distribution:", np.bincount(np.argmax(y_test, axis=1)))

# ============================
# 6. ВЕСА КЛАССОВ (балансировка)
# ============================
y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(class_weight='balanced', classes = np.array([0, 1]), y=y_train_labels)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Веса классов (фон/дрон):", class_weight_dict)

# ============================
# 7. МОДЕЛЬ 1D CNN
# ============================
input_shape = X_train.shape[1:]

model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(64, 5, activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling1D(),

    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    Dropout(0.5),

    Dense(num_classes, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ============================
# 8. ОБУЧЕНИЕ
# ============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    class_weight=class_weight_dict,
    verbose=1
)

# ============================
# 9. ОЦЕНКА
# ============================
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nМатрица ошибок (0=фон, 1=дрон):")
cm = confusion_matrix(y_true, y_pred)
print(cm)
print("\nКлассификационный отчёт:")
report = classification_report(y_true, y_pred, target_names=['Фон', 'Дрон'])
print(report)

# Сохранение отчёта в TXT
with open('classification_report_binary.txt', 'w', encoding='utf-8') as f:
    f.write("=== Бинарная классификация: фон (класс 4) vs дроны (все остальные) ===\n")
    f.write(report)
    f.write("\n\nConfusion matrix:\n")
    f.write(np.array2string(cm))
print("Отчёт сохранён в 'classification_report_binary.txt'")

# ============================
# 10. ГРАФИКИ ОБУЧЕНИЯ
# ============================
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(), plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(), plt.title('Loss')
plt.suptitle('Обучение 1D CNN на PSD (фон vs дрон)')
plt.savefig('training_binary.png', dpi=300)
plt.show()

# ============================
# 11. МАТРИЦА ОШИБОК (визуализация)
# ============================
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Фон', 'Дрон'],
            yticklabels=['Фон', 'Дрон'])
plt.xlabel('Предсказанный')
plt.ylabel('Настоящий')
plt.title('Матрица ошибок (фон vs дрон)')
plt.tight_layout()
plt.savefig('confusion_binary.png', dpi=300)
plt.show()

# ============================
# 12. СОХРАНЕНИЕ МОДЕЛИ
# ============================
model.save('psd_binary_classifier.keras')
print("Модель сохранена как 'psd_binary_classifier.keras'")