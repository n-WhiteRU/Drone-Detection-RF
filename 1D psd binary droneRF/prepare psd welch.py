import numpy as np
import os
from scipy.signal import welch
import tqdm
 
# ==================== ПУТИ ====================
LOAD_DIR = r'C:\Users\debil\OneDrive\Рабочий стол\дроны\data\DroneRF'
SAVE_PATH = r'C:\Users\debil\OneDrive\Рабочий стол\дроны\data\dataset uni.npy'
 
# ==================== КЛАССЫ ====================
BUI = [
    ['00000'],                            # фон
    ['10000', '10001', '10010', '10011'], # Bebop
    ['10100', '10101', '10110', '10111'], # AR.Drone
    ['11000']                             # Phantom
]
 
# ==================== ПАРАМЕТРЫ ====================
L = int(1e5)
NFFT = 1024
Q = 10
 
# ==================== PSD ====================
def compute_psd_linear(signal):
    signal = signal - np.mean(signal)
    _, Pxx = welch(signal, nperseg=1024, noverlap=512, nfft=NFFT)
    return Pxx
 
# ==================== МАППИНГ ====================
mode_map = {
    '00': 0,
    '01': 1,
    '10': 2,
    '11': 3
}
 
# ==================== ХРАНИЛИЩА ====================
X = []
 
y_binary = []
y_type = []
y_full = []
 
groups = []
 
powers = []
vars_ = []
 
group_id = 0
 
# ==================== СБОР ====================
for type_id, group in enumerate(BUI):
 
    for code in group:
 
        # -------- режим --------
        mode_code = code[-2:]
        mode_id = mode_map.get(mode_code, 0)
 
        # -------- количество файлов --------
        if code == '00000':
            N = 40
        elif code == '10111':
            N = 17
        else:
            N = 20
 
        for n in tqdm.tqdm(range(N), desc=f"{code}"):
 
            h_path = os.path.join(LOAD_DIR, f'{code}H_{n}.csv')
            l_path = os.path.join(LOAD_DIR, f'{code}L_{n}.csv')
 
            if not os.path.exists(h_path) or not os.path.exists(l_path):
                continue
 
            x_sig = np.loadtxt(h_path, delimiter=',', dtype=np.float32)
            y_sig = np.loadtxt(l_path, delimiter=',', dtype=np.float32)
 
            for k in range(len(x_sig)//L):
 
                seg_L = x_sig[k*L:(k+1)*L]
                seg_H = y_sig[k*L:(k+1)*L]
 
                # ===== PSD =====
                psd_L = compute_psd_linear(seg_L)
                psd_H = compute_psd_linear(seg_H)
 
                # ===== СШИВКА =====
                mean_L_tail = np.mean(psd_L[-Q:])
                mean_H_head = np.mean(psd_H[:Q])
 
                scale = mean_L_tail / (mean_H_head + 1e-12)
                psd_H_scaled = psd_H * scale
 
                full_linear = np.concatenate((psd_L, psd_H_scaled))
 
                # ===== В dB =====
                full_db = 10 * np.log10(full_linear + 1e-12)
 
                # ==================== МЕТКИ ====================
 
                # --- бинарная ---
                binary_label = 0 if type_id == 0 else 1
 
                # --- тип дрона ---
                type_label = type_id  # 0..3
 
                # --- полный класс (13 классов) ---
                if type_id == 0:
                    full_label = 0
                else:
                    full_label = (type_id - 1) * 4 + mode_id + 1
                    # диапазон 1..12
 
                # ==================== СОХРАНЕНИЕ ====================
                X.append(full_db)
 
                y_binary.append(binary_label)
                y_type.append(type_label)
                y_full.append(full_label)
 
                groups.append(group_id)
 
                powers.append(np.mean(full_db))
                vars_.append(np.var(full_db))
 
            group_id += 1
 
# ==================== В МАССИВЫ ====================
X = np.array(X)
y_binary = np.array(y_binary)
y_type = np.array(y_type)
y_full = np.array(y_full)
groups = np.array(groups)
 
powers = np.array(powers)
vars_ = np.array(vars_)
 
print("Before filtering:", X.shape)
print("Binary:", np.bincount(y_binary))
print("Type:", np.bincount(y_type))
print("Full:", np.bincount(y_full))
 
# ==================== ФИЛЬТР ====================
drone_mask = (y_binary == 1)
 
power_thr = np.percentile(powers[drone_mask], 5)
var_thr = np.percentile(vars_[drone_mask], 5)
 
mask = np.ones(len(X), dtype=bool)
mask[drone_mask] = (
    (powers[drone_mask] > power_thr) &
    (vars_[drone_mask] > var_thr)
)
 
# ==================== ПРИМЕНЕНИЕ ====================
X = X[mask]
y_binary = y_binary[mask]
y_type = y_type[mask]
y_full = y_full[mask]
groups = groups[mask]
 
print("After filtering:", X.shape)
 
# ==================== СОХРАНЕНИЕ ====================
np.save(SAVE_PATH, {
    "X": X,
    "y_binary": y_binary,
    "y_type": y_type,
    "y_full": y_full,
    "groups": groups
})
 
print("Saved to:", SAVE_PATH)