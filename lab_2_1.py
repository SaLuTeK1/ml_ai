# l2_part1_variant1.py
import os
import argparse
import random
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ----------------- 1) Еталони (біполярні {-1,+1}) -----------------
# Сенсори: [TempHigh, OilPressureLow, VibrationHigh, RPM_Unstable]
# Пояснення класів:
#  - 'Норма'       : усі сенсори в нормі -> [-1, -1, -1, -1]
#  - 'Перегрів'    : висока t°           -> [ +1, -1, -1, -1]
#  - 'Низький тиск': падіння тиску часто тягне вібрації -> [-1, +1, +1, -1]
#  - 'Детонація'   : висока t°, вібрації і нестабільні оберти -> [ +1, -1, +1, +1]
CLASSES = ["Норма", "Перегрів", "Низький тиск", "Детонація"]
PROTOTYPES = np.array([
    [-1, -1, -1, -1],   # Норма
    [ 1, -1, -1, -1],   # Перегрів
    [-1,  1,  1, -1],   # Низький тиск масла
    [ 1, -1,  1,  1],   # Детонація
], dtype=np.int8)  # shape=(4,4)

# ----------------- 2) Генерація зашумлених прикладів -----------------
def add_noise(vec: np.ndarray, flip_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Інвертує кожен біт з імовірністю flip_prob."""
    flips = rng.random(vec.shape) < flip_prob
    noisy = vec.copy()
    noisy[flips] *= -1
    return noisy

def make_dataset(samples_per_class=300, noise=0.2, seed=42):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for cls_idx, proto in enumerate(PROTOTYPES):
        for _ in range(samples_per_class):
            X_list.append(add_noise(proto, noise, rng))
            y_list.append(cls_idx)
    X = np.stack(X_list).astype(np.float32)  # (N,4)
    y = np.array(y_list, dtype=np.int64)     # (N,)
    return X, y

# ----------------- 3) Мережа Хеммінга + MAXNET (NumPy) --------------
@dataclass
class HammingMaxnetConfig:
    epsilon: float = 0.1   # гальмівний коеф. для MAXNET; має бути < 1/n_neurons
    max_iter: int = 50

class HammingMaxnet:
    def __init__(self, config=HammingMaxnetConfig()):
        self.cfg = config
        self.prototypes_ = None        # shape=(n_classes, n_features)
        self.classes_ = None           # shape=(n_classes,)
        self.last_activations_traj_ = None  # динаміка MAXNET (для останнього predict_one)

    # "Навчання" — просто запам’ятовуємо прототипи (можна брати центроїди, але тут еталони фіксовані)
    def fit(self, prototypes: np.ndarray):
        self.prototypes_ = prototypes.astype(np.float32)
        # нормалізуємо до довжини 1 — це перетворює скалярний добуток на косинусну схожість
        norms = np.linalg.norm(self.prototypes_, axis=1, keepdims=True)
        self.prototypes_ = self.prototypes_ / np.clip(norms, 1e-8, None)
        self.classes_ = np.arange(self.prototypes_.shape[0])
        return self

    def hamming_layer(self, x: np.ndarray) -> np.ndarray:
        """Шар Хеммінга: оцінки схожості як скалярний добуток W @ x."""
        # нормалізуємо вхід
        x = x.astype(np.float32)
        x = x / max(np.linalg.norm(x), 1e-8)
        return self.prototypes_ @ x  # (n_classes,)

    def maxnet(self, scores: np.ndarray, record_traj=False) -> int:
        """Ітеративний WTA. Порогова активація relu(). Повертає індекс-переможець."""
        a = scores.astype(np.float32).copy()
        n = a.size
        eps = min(self.cfg.epsilon, 0.9 / n)  # підстрахуємось: умова збіжності eps < 1/n
        traj = [a.copy()] if record_traj else None

        for _ in range(self.cfg.max_iter):
            if (a > 0).sum() <= 1:
                break
            s = a.sum()
            # a_j <- a_j - eps * (sum_except_j)
            a = a - eps * (s - a)
            a = np.maximum(a, 0)  # поріг
            if record_traj: traj.append(a.copy())

        if record_traj: self.last_activations_traj_ = np.stack(traj)  # (t, n)
        return int(np.argmax(a))

    def predict_one(self, x: np.ndarray, record_traj=False) -> int:
        scores = self.hamming_layer(x)
        winner = self.maxnet(scores, record_traj=record_traj)
        return winner

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = [self.predict_one(x, record_traj=False) for x in X]
        return np.array(preds, dtype=int)

# ----------------- 4) Візуалізації -----------------
def plot_maxnet_dynamics(net: HammingMaxnet, x: np.ndarray, class_names, out_path):
    _ = net.predict_one(x, record_traj=True)
    traj = net.last_activations_traj_  # shape=(t, n_classes)
    t = np.arange(traj.shape[0])
    plt.figure(figsize=(7,4))
    for j in range(traj.shape[1]):
        plt.plot(t, traj[:, j], label=class_names[j])
    plt.xlabel("Ітерація")
    plt.ylabel("Активація")
    plt.title("Динаміка MAXNET (WTA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_confusion(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(values_format='d', cmap='Blues')
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ----------------- 5) main -----------------
def main():
    parser = argparse.ArgumentParser(description="ЛР2 ч.1: Хеммінг+MAXNET для біполярних векторів")
    parser.add_argument("--samples-per-class", type=int, default=300, help="скільки зразків з шумом на кожен клас")
    parser.add_argument("--noise", type=float, default=0.2, help="ймовірність інверсії біта (0..1)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="eps для MAXNET (має бути < 1/n_classes)")
    parser.add_argument("--max-iter", type=int, default=50, help="макс. к-ть ітерацій MAXNET")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plots", action="store_true", help="зберегти графіки (динаміка MAXNET + матриця похибок)")
    args = parser.parse_args()

    os.makedirs("runs_l2", exist_ok=True)

    # Дані
    X, y = make_dataset(samples_per_class=args.samples_per_class, noise=args.noise, seed=args.seed)

    # Модель
    net = HammingMaxnet(HammingMaxnetConfig(epsilon=args.epsilon, max_iter=args.max_iter))
    net.fit(PROTOTYPES)

    # Прогноз
    y_pred = net.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc*100:.2f}%  (noise={args.noise}, samples={len(y)})")

    # Графіки
    if args.plots:
        # 1) Динаміка MAXNET на одному випадковому прикладі
        idx = random.randrange(len(X))
        plot_maxnet_dynamics(net, X[idx], CLASSES, out_path=os.path.join("runs_l2", "maxnet_dynamics.png"))
        # 2) Матриця похибок
        plot_confusion(y, y_pred, CLASSES, out_path=os.path.join("runs_l2", "confusion.png"))
        print("Збережено графіки в runs_l2/")

    # Маленький звіт у консоль по кожному класу
    for i, name in enumerate(CLASSES):
        cls_mask = (y == i)
        acc_i = accuracy_score(y[cls_mask], y_pred[cls_mask])
        print(f"  {name:12s}: {acc_i*100:5.2f}%  (N={cls_mask.sum()})")

if __name__ == "__main__":
    main()
