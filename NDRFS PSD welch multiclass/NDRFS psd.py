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
# 1. ЗАГРУЗКА ДАННЫХ (NDRFSC)
# ============================
DATASET_PATH = Path(r'C:\kolya\дрони\dataset NDRFS\dataset.pt')
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Файл {DATASET_PATH} не найден. Укажите правильный путь.")

data = torch.load(DATASET_PATH)
x_iq = data['x_iq'].numpy()          # (n_samples, 2, seq_len)
y = data['y'].numpy()                # (n_samples,)

print(f"Загружено: X_iq {x_iq.shape}, y {y.shape}")
print(f"Классы: {np.unique(y)}")

# ============================
# 2. ПРЕОБРАЗОВАНИЕ IQ → PSD (УЭЛЧ)
# ============================
def iq_to_psd(iq_batch, fs=1.0, nperseg=256, noverlap=None, nfft=None):
    """
    Пакетное преобразование IQ-сигналов в PSD.
    iq_batch: (n_samples, 2, seq_len)
    Возвращает: (n_samples, n_freq) - линейная PSD
    """
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg
    
    n_samples = iq_batch.shape[0]
    psd_list = []
    for i in range(n_samples):
        complex_sig = iq_batch[i, 0, :] + 1j * iq_batch[i, 1, :]
        f, Pxx = scipy.signal.welch(complex_sig, fs=fs, nperseg=nperseg,
                                    noverlap=noverlap, nfft=nfft,
                                    window='hann', scaling='density')
        psd_list.append(Pxx)
    return np.array(psd_list), f

# Параметры PSD (можно менять)
NPERSEG = 256   # длина сегмента
NFFT = 256
NOVERLAP = NPERSEG // 2

print("Вычисление PSD...")
X_psd, freqs = iq_to_psd(x_iq, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT)
print(f"PSD форма: {X_psd.shape}")   # (n_samples, n_freq)

# ============================
# 3. НОРМАЛИЗАЦИЯ PSD
# ============================
# Центрирование и масштабирование для каждого образца (как в исходном примере)
X_psd = X_psd - np.mean(X_psd, axis=1, keepdims=True)
X_psd = X_psd / (np.std(X_psd, axis=1, keepdims=True) + 1e-8)

# Добавляем ось канала для Conv1D: (samples, freq_bins, 1)
X_psd = X_psd[..., np.newaxis]
print("Форма после нормализации:", X_psd.shape)

# ============================
# 4. СТРАТИФИЦИРОВАННОЕ РАЗДЕЛЕНИЕ (каждый класс представлен везде)
# ============================
num_classes = len(np.unique(y))
y_cat = tf.keras.utils.to_categorical(y, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X_psd, y_cat, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])
print("Распределение классов в train:", np.bincount(np.argmax(y_train, axis=1)))
print("Распределение классов в test:", np.bincount(np.argmax(y_test, axis=1)))

# ============================
# 5. ВЕСА КЛАССОВ (для балансировки)
# ============================
y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weight_dict = dict(enumerate(class_weights))
print("Веса классов:", class_weight_dict)

# ============================
# 6. ПОСТРОЕНИЕ МОДЕЛИ (1D CNN)
# ============================
input_shape = X_train.shape[1:]   # (freq_bins, 1)

model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(64, 5, activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling1D(),

    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
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

# ============================
# 7. ОБУЧЕНИЕ
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
# 8. ОЦЕНКА
# ============================
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nМатрица ошибок:")
print(confusion_matrix(y_true, y_pred))
print("\nКлассификационный отчёт:")
print(classification_report(y_true, y_pred))

# ============================
# 9. ГРАФИКИ ОБУЧЕНИЯ
# ============================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.suptitle('Обучение 1D CNN на PSD признаках')
plt.savefig('training_history.png', dpi=300)
plt.show()

# ============================
# 10. МАТРИЦА ОШИБОК
# ============================
# Попробуем загрузить имена классов (если есть class_stats.csv)
try:
    import pandas as pd
    stats_path = DATASET_PATH.parent / 'class_stats.csv'
    class_stats = pd.read_csv(stats_path, index_col=0)
    class_names = class_stats['class'].values
except:
    class_names = [f'Класс {i}' for i in range(num_classes)]

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, y_pred),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Предсказанный')
plt.ylabel('Настоящий')
plt.title('Матрица ошибок (классификация по PSD)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# ============================
# 11. СОХРАНЕНИЕ МОДЕЛИ
# ============================
model.save('psd_drone_classifier.keras')
print("Модель сохранена как 'psd_drone_classifier.keras'")