#!/usr/bin/env python3
# main.py — ЛР1: MNIST, порівняння MLP vs CNN + вплив активацій
# Варіант 1: batch=32, epochs=5, optimizer=Adam

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # тихіші логи TF

import time
import json
import random
import argparse
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------- Логування ----------------------
def setup_logger(log_level=logging.INFO):
    os.makedirs("runs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join("runs", f"train_{ts}.log")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler()],
    )
    logging.info(f"Лог-файл: {log_path}")
    return ts

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

# ---------------------- Дані ----------------------
def load_mnist(subset=None):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32")/255.0)[..., None]
    x_test  = (x_test.astype("float32")/255.0)[..., None]
    if subset is not None:
        x_train, y_train = x_train[:subset], y_train[:subset]
    return (x_train, y_train), (x_test, y_test)

# ---------------------- Моделі ----------------------
def build_mlp(activation="relu"):
    inp = layers.Input(shape=(28,28,1))
    x = layers.Flatten()(inp)
    if activation == "leaky_relu":
        x = layers.Dense(128)(x); x = layers.LeakyReLU(0.01)(x)
    else:
        x = layers.Dense(128, activation=activation)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inp, out, name=f"mlp_{activation}")

def build_cnn(activation="relu"):
    inp = layers.Input(shape=(28,28,1))
    # block 1
    if activation == "leaky_relu":
        x = layers.Conv2D(24, 3)(inp); x = layers.LeakyReLU(0.01)(x)
    else:
        x = layers.Conv2D(24, 3, activation=activation)(inp)
    x = layers.MaxPool2D(2)(x)
    # block 2
    if activation == "leaky_relu":
        x = layers.Conv2D(36, 3)(x); x = layers.LeakyReLU(0.01)(x)
    else:
        x = layers.Conv2D(36, 3, activation=activation)(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Flatten()(x)
    # dense
    if activation == "leaky_relu":
        x = layers.Dense(128)(x); x = layers.LeakyReLU(0.01)(x)
    else:
        x = layers.Dense(128, activation=activation)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inp, out, name=f"cnn_{activation}")

# ---------------------- Колбек часу епох ----------------------
class EpochTimeLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        logging.info("Початок навчання...")

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()
        logging.info(f"Епоха {epoch+1} почалась")

    def on_epoch_end(self, epoch, logs=None):
        dt = time.time() - self._t0
        logging.info(f"Епоха {epoch+1} завершена | "
                     f"loss={logs.get('loss'):.4f} acc={logs.get('accuracy'):.4f} "
                     f"val_loss={logs.get('val_loss'):.4f} val_acc={logs.get('val_accuracy'):.4f} "
                     f"({dt:.1f} с)")

# ---------------------- Збереження моделей ----------------------
def save_model(model, run_tag, out_dir="models", fmt="keras"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{model.name}_{run_tag}")
    if fmt == "keras":
        path = base + ".keras"
        model.save(path)
    elif fmt == "savedmodel":
        path = base  # папка
        model.save(path)
    else:  # weights
        path = base + ".weights.h5"
        model.save_weights(path)
    logging.info(f"Модель збережено: {path}")

# ---------------------- Тренування ----------------------
def train_and_eval(model, x_train, y_train, x_test, y_test,
                   batch=32, epochs=5, lr=1e-3,
                   use_es=True, run_tag="", save=False, save_dir="models", save_fmt="keras"):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  jit_compile=True)

    csv_name = os.path.join("runs", f"history_{model.name}_{run_tag}.csv")
    callbacks = [keras.callbacks.CSVLogger(csv_name), EpochTimeLogger()]
    if use_es:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=1, restore_best_weights=True))

    logging.info(f"=== Старт тренування: {model.name} | batch={batch}, epochs={epochs}, lr={lr} ===")
    history = model.fit(x_train, y_train,
                        batch_size=batch, epochs=epochs,
                        validation_split=0.1, verbose=2,
                        callbacks=callbacks)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logging.info(f"Тест: loss={test_loss:.4f} acc={test_acc:.4f}")

    if save:
        save_model(model, run_tag, out_dir=save_dir, fmt=save_fmt)

    return float(test_acc), history

# ---------------------- Візуалізації ----------------------
def plot_accuracy_bar(results, out_png):
    import pandas as pd
    import matplotlib.pyplot as plt
    df = (pd.DataFrame(results) * 100).round(2)
    ax = df.plot(kind="bar", rot=0, figsize=(8,4), title="Accuracy by activation, %")
    ax.set_ylabel("Accuracy, %")
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
    logging.info(f"Графік точностей збережено: {out_png}")
    return df

def confusion_and_examples(model, x_test, y_test, out_cm_png, out_examples_png):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(values_format='d', cmap='Blues')
    plt.title("Confusion matrix (best CNN)"); plt.tight_layout()
    plt.savefig(out_cm_png, dpi=160); plt.close()
    logging.info(f"Матриця похибок збережена: {out_cm_png}")

    wrong = np.where(y_pred != y_test)[0][:12]
    if len(wrong) > 0:
        plt.figure(figsize=(8,6))
        for i, idx in enumerate(wrong):
            plt.subplot(3,4,i+1)
            plt.imshow(x_test[idx].squeeze(), cmap='gray')
            plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}")
            plt.axis('off')
        plt.suptitle("Misclassified examples"); plt.tight_layout()
        plt.savefig(out_examples_png, dpi=160); plt.close()
        logging.info(f"Приклади помилок збережено: {out_examples_png}")

# ---------------------- main ----------------------
def main():
    print("TensorFlow:", tf.__version__, flush=True)
    print("GPU:", tf.config.list_physical_devices("GPU"), flush=True)

    ap = argparse.ArgumentParser(description="ЛР1: MNIST — MLP vs CNN + активації")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--subset", type=int, default=None, help="швидкий прогін на N зразках")
    ap.add_argument("--no-es", action="store_true", help="без EarlyStopping")
    ap.add_argument("--plots", action="store_true", help="зберегти графіки")
    ap.add_argument("--save-models", action="store_true", help="зберегти всі натреновані моделі")
    ap.add_argument("--models-dir", type=str, default="models")
    ap.add_argument("--save-format", choices=["keras","savedmodel","weights"], default="keras")
    args = ap.parse_args()

    run_tag = setup_logger(); set_seed(42)

    (x_train, y_train), (x_test, y_test) = load_mnist(subset=args.subset)
    logging.info(f"x_train: {x_train.shape}, x_test: {x_test.shape}, subset={args.subset}")

    activations = ["relu", "leaky_relu", "tanh", "sigmoid"]
    results = {"MLP": {}, "CNN": {}}

    # --- MLP ---
    logging.info("\n================ MLP experiments ================\n")
    for act in activations:
        model = build_mlp(act)
        acc, _ = train_and_eval(model, x_train, y_train, x_test, y_test,
                                batch=args.batch, epochs=args.epochs, lr=args.lr,
                                use_es=not args.no_es, run_tag=run_tag,
                                save=args.save_models, save_dir=args.models_dir, save_fmt=args.save_format)
        results["MLP"][act] = acc

    # --- CNN ---
    logging.info("\n================ CNN experiments ================\n")
    best_cnn_model, best_acc = None, -1.0
    for act in activations:
        model = build_cnn(act)
        acc, _ = train_and_eval(model, x_train, y_train, x_test, y_test,
                                batch=args.batch, epochs=args.epochs, lr=args.lr,
                                use_es=not args.no_es, run_tag=run_tag,
                                save=args.save_models, save_dir=args.models_dir, save_fmt=args.save_format)
        results["CNN"][act] = acc
        if acc > best_acc: best_acc, best_cnn_model = acc, model

    # --- Підсумок + CSV ---
    try:
        import pandas as pd
        df = (pd.DataFrame(results)*100).round(2)
        print("\nAccuracy by activation, %:\n"); print(df.to_string(), flush=True)
        csv_out = os.path.join("runs", f"results_{run_tag}.csv")
        df.to_csv(csv_out, index=True); logging.info(f"Результати збережені у {csv_out}")
        print(f"\nBest MLP: {df['MLP'].idxmax()} = {df['MLP'].max():.2f}%")
        print(f"Best CNN: {df['CNN'].idxmax()} = {df['CNN'].max():.2f}%")
        print(f"\nAverage MLP acc: {df['MLP'].mean():.2f}%")
        print(f"Average CNN acc: {df['CNN'].mean():.2f}%")
    except Exception as e:
        logging.warning(f"pandas недоступний ({e}) — друкую JSON")
        print(json.dumps(results, indent=2), flush=True)

    # --- Графіки для звіту ---
    if args.plots:
        acc_png = os.path.join("runs", f"acc_by_activation_{run_tag}.png")
        plot_accuracy_bar(results, acc_png)
        cm_png = os.path.join("runs", f"confusion_best_cnn_{run_tag}.png")
        ex_png = os.path.join("runs", f"errors_best_cnn_{run_tag}.png")
        confusion_and_examples(best_cnn_model, x_test, y_test, cm_png, ex_png)

if __name__ == "__main__":
    main()
