import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# ============================
# 0. ПУТИ И ПАРАМЕТРЫ
# ============================
DATASET_FOLDER = Path('dataset NDRFS/')
PREPROCESSED_FILE = DATASET_FOLDER / 'preprocessed_spec_3ch.pt'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используем устройство:", DEVICE)

# ============================
# 1. ЗАГРУЗКА ПРЕДОБРАБОТАННЫХ ДАННЫХ
# ============================
if not PREPROCESSED_FILE.exists():
    raise FileNotFoundError(f"Файл {PREPROCESSED_FILE} не найден. Сначала выполните предобработку.")

data = torch.load(PREPROCESSED_FILE)
x_spec_3ch = data['x_spec_3ch']   # (n_samples, 3, 224, 224)
y_binary = data['y_binary']       # (n_samples,)

print(f"Загружены предобработанные данные: {x_spec_3ch.shape}, метки: {y_binary.shape}")
print(f"Фон (0): {(y_binary == 0).sum().item()}, Дроны (1): {(y_binary == 1).sum().item()}")

# ============================
# 2. ДАТАСЕТ (уже преобразованные данные, никаких дополнительных трансформаций)
# ============================
class ReadySpectrogramDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Создаём датасет
full_dataset = ReadySpectrogramDataset(x_spec_3ch, y_binary)

# ============================
# 3. СТРАТИФИЦИРОВАННОЕ РАЗДЕЛЕНИЕ
# ============================
indices = np.arange(len(full_dataset))
y_np = y_binary.numpy()

train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y_np
)

train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
print("Train distribution:", np.bincount(y_np[train_idx]))
print("Test distribution:", np.bincount(y_np[test_idx]))

# ============================
# 4. DATA LOADERS
# ============================
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================
# 5. ВЕСА КЛАССОВ ДЛЯ БАЛАНСИРОВКИ
# ============================
from sklearn.utils.class_weight import compute_class_weight
classes = np.array([0, 1])
y_train = y_np[train_idx]
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print("Веса классов (фон, дрон):", class_weight_dict)

# Для BCEWithLogitsLoss используем pos_weight = вес_дрона / вес_фона
pos_weight = torch.tensor([class_weight_dict[1] / class_weight_dict[0]]).to(DEVICE)

# ============================
# 6. МОДЕЛЬ VGG-19
# ============================
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model = models.vgg19(weights='IMAGENET1K_V1')
# Заменяем классификатор на бинарный
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 1)
model = model.to(DEVICE)

# ============================
# 7. ФУНКЦИИ ОБУЧЕНИЯ И ОЦЕНКИ С ПРОГРЕСС-БАРАМИ
# ============================
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

def train_epoch(model, loader, criterion, optimizer, device, epoch_num):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch_num} [Train]", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item() * inputs.size(0)
        total_loss += batch_loss
        preds = (torch.sigmoid(outputs) >= 0.5).long().view(-1)
        batch_correct = (preds == targets.long().view(-1)).sum().item()
        correct += batch_correct
        total += targets.size(0)
        
        # Обновляем прогресс-бар
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{batch_correct/inputs.size(0):.4f}'
        })
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion, device, desc="Valid"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).long().view(-1)
            correct += (preds == targets.long().view(-1)).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.long().view(-1).cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{(preds == targets.long().view(-1)).float().mean().item():.4f}'})
    return total_loss / total, correct / total, all_preds, all_targets

# ============================
# 8. ЦИКЛ ОБУЧЕНИЯ С ПРОГРЕСС-БАРОМ ПО ЭПОХАМ
# ============================
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("Начало обучения...")
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
    val_loss, val_acc, _, _ = eval_epoch(model, test_loader, criterion, DEVICE, desc=f"Epoch {epoch} [Valid]")
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    scheduler.step(val_loss)
    
    # Вывод основной информации (без tqdm)
    print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

# Финальная оценка на тесте
_, _, y_pred, y_true = eval_epoch(model, test_loader, criterion, DEVICE, desc="Final Test")

# ============================
# 9. ОТЧЁТ И ГРАФИКИ
# ============================
print("\nМатрица ошибок (0=фон, 1=дрон):")
cm = confusion_matrix(y_true, y_pred)
print(cm)
print("\nКлассификационный отчёт:")
report = classification_report(y_true, y_pred, target_names=['Фон', 'Дрон'])
print(report)

# Сохранение отчёта
with open('classification_report_vgg19.txt', 'w', encoding='utf-8') as f:
    f.write("=== Бинарная классификация VGG-19 (спектрограммы) ===\n")
    f.write(report)
    f.write("\n\nConfusion matrix:\n")
    f.write(np.array2string(cm))

# Графики обучения
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.legend()
plt.title('Accuracy')
plt.suptitle('Обучение VGG-19 на спектрограммах (фон vs дрон)')
plt.savefig('vgg19_training.png', dpi=300)
plt.show()

# Матрица ошибок (визуализация)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Фон', 'Дрон'],
            yticklabels=['Фон', 'Дрон'])
plt.xlabel('Предсказанный')
plt.ylabel('Настоящий')
plt.title('Матрица ошибок VGG-19')
plt.savefig('vgg19_confusion_matrix.png', dpi=300)
plt.show()

# Сохранение модели
torch.save(model.state_dict(), 'vgg19_binary_classifier.pth')
print("Модель сохранена как 'vgg19_binary_classifier.pth'")