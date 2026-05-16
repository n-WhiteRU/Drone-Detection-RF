import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# ============================
# 1. ЗАГРУЗКА ИСХОДНЫХ ДАННЫХ
# ============================
DATASET_FOLDER = Path('dataset NDRFS/')
dataset_path = DATASET_FOLDER / 'dataset.pt'
data = torch.load(dataset_path)

x_spec = data['x_spec']          # (n_samples, 2, H, W)
y = data['y']                    # (n_samples,)
print(f"Спектрограммы: {x_spec.shape}, метки: {y.shape}")

# ============================
# 2. БИНАРНЫЕ МЕТКИ (фон = класс 4 -> 0, дроны = всё остальное -> 1)
# ============================
BACKGROUND_CLASS = 4
y_binary = (y != BACKGROUND_CLASS).long()   # 0 – фон, 1 – дрон

num_bg = (y_binary == 0).sum().item()
num_drone = (y_binary == 1).sum().item()
print(f"Фон: {num_bg}, Дроны: {num_drone}")

# ============================
# 3. ФУНКЦИЯ ПРЕОБРАЗОВАНИЯ (2 канала -> 3 канала)
# ============================
def preprocess_spectrogram(spec_2ch):
    """
    spec_2ch: тензор (2, H, W) или (batch, 2, H, W)
    Возвращает (3, H, W) или (batch, 3, H, W)
    """
    if spec_2ch.dim() == 3:
        re = spec_2ch[0:1, :, :]
        im = spec_2ch[1:2, :, :]
        return torch.cat([re, im, re], dim=0)
    else:
        re = spec_2ch[:, 0:1, :, :]
        im = spec_2ch[:, 1:2, :, :]
        return torch.cat([re, im, re], dim=1)

# ============================
# 4. ПРЕДОБРАБОТКА ВСЕХ СПЕКТРОГРАММ С ПРОГРЕСС-БАРОМ
# ============================
n_samples = x_spec.shape[0]
print("Предобработка спектрограмм...")
preprocessed_list = []
for i in tqdm(range(n_samples), desc="Преобразование в 3 канала"):
    spec_3ch = preprocess_spectrogram(x_spec[i])   # (3, H, W)
    preprocessed_list.append(spec_3ch)

# Склеиваем в один тензор
x_spec_3ch = torch.stack(preprocessed_list, dim=0)   # (n_samples, 3, H, W)
print(f"Готовый тензор: {x_spec_3ch.shape}")

# ============================
# 5. СОХРАНЕНИЕ ПРЕДОБРАБОТАННЫХ ДАННЫХ
# ============================
output_path = DATASET_FOLDER / 'preprocessed_spec_3ch.pt'
torch.save({
    'x_spec_3ch': x_spec_3ch,
    'y_binary': y_binary
}, output_path)
print(f"Предобработанные данные сохранены в {output_path}")

# ============================
# 6. ДАТАСЕТ ДЛЯ ЗАГРУЗКИ ГОТОВЫХ ДАННЫХ (БЕЗ ПОВТОРНОЙ ОБРАБОТКИ)
# ============================
class SpectrogramDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, transform=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.x[idx]          # (3, H, W)
        label = self.y[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ============================
# 7. ПРИМЕР ЗАГРУЗКИ И СОЗДАНИЯ DATALOADER
# ============================
# Загружаем предобработанные данные
loaded = torch.load(output_path)
x_ready = loaded['x_spec_3ch']
y_ready = loaded['y_binary']