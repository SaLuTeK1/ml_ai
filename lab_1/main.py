# python main.py --save-models --save-format keras --models-dir models --plots
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

# ---------------------- Конфіг логування ----------------------
def setup_logger(log_level=logging.INFO):
    os.makedirs("runs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join("runs", f"train_{ts}.log")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )
    logging.info(f"Лог-файл: {log_path}")
    return ts

# ---------------------- Seed ----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ---------------------- Дані ----------------------
def load_mnist(subset=None):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32")/255.0)[..., None]  # (N,28,28,1)
    x_test  = (x_test.astype("float32")/255.0)[..., None]
    if subset is not None:
        x_train, y_train = x_train[:subset], y_train[:subset]
    return (x_train, y_train), (x_test, y_test)

# ---------------------- Моделі ----------------------
def build_mlp(activation="relu", num_classes=10):
    inp = layers.Input(shape=(28,28,1))
    x = layers.Flatten()(inp)
    if activation == "leaky_relu":
        x = layers.Dense(128)(x); x = layers.LeakyReLU(alpha=0.01)(x)
    else:
        x = layers.Dense(128, activation=activation)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name=f"mlp_{activation}")

def build_cnn(activation="relu", num_classes=10):
    inp = layers.Input(shape=(28,28,1))
    if activation == "leaky_relu":
        x = layers.Conv2D(24, 3)(inp); x = layers.LeakyReLU(alpha=0.01)(x)
    else:
        x = layers.Conv2D(24, 3, activation=activation)(inp)
    x = layers.MaxPool2D(2)(x)
    if activation == "leaky_relu":
        x = layers.Conv2D(36, 3)(x); x = layers.LeakyReLU(alpha=0.01)(x)
    else:
        x = layers.Conv2D(36, 3, activation=activation)(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Flatten()(x)
    if activation == "leaky_relu":
        x = layers.Dense(128)(x); x = layers.LeakyReLU(alpha=0.01)(x)
    else:
        x = layers.Dense(128, activation=activation)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name=f"cnn_{activation}")

# ---------------------- Callback для часу епох ----------------------
class EpochTimeLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self._epoch_start = None
        logging.info("Початок навчання...")

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()
        logging.info(f"Епоха {epoch+1} почалась")

    def on_epoch_end(self, epoch, logs=None):
        dur = time.time() - self._epoch_start
        self.epoch_times.append(dur)
        msg = (f"Епоха {epoch+1} завершена | "
               f"loss={logs.get('loss'):.4f} "
               f"acc={logs.get('accuracy'):.4f} "
               f"val_loss={logs.get('val_loss'):.4f} "
               f"val_acc={logs.get('val_accuracy'):.4f} "
               f"({dur:.1f} с)")
        logging.info(msg)

# ---------------------- Тренування ----------------------
def train_and_eval(model, x_train, y_train, x_test, y_test,
                   batch=32, epochs=5, lr=1e-3,
                   use_es=True, run_tag=""):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True
    )

    csv_name = os.path.join("runs", f"history_{model.name}_{run_tag}.csv")
    csv_cb = keras.callbacks.CSVLogger(csv_name)
    time_cb = EpochTimeLogger()
    callbacks = [csv_cb, time_cb]

    if use_es:
        es = keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=1, restore_best_weights=True
        )
        callbacks.append(es)

    logging.info(f"=== Старт тренування: {model.name} | "f"batch={batch}, epochs={epochs}, lr={lr} ===")

    history = model.fit(
        x_train, y_train,
        batch_size=batch,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logging.info(f"Тест: loss={test_loss:.4f} acc={test_acc:.4f}")
    return float(test_acc), history

# ---------------------- Побудова графіків ----------------------
def plot_accuracy_bar(results, out_png):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = (pd.DataFrame(results) * 100).round(2)
    ax = df.plot(kind="bar", rot=0, figsize=(8,4), title="Accuracy by activation, %")
    ax.set_ylabel("Accuracy, %")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    logging.info(f"Графік точностей збережено: {out_png}")
    return df

def save_model(model, run_tag, out_dir, fmt="keras"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{model.name}_{run_tag}")
    if fmt == "keras":
        path = base + ".keras"
        model.save(path)
    elif fmt == "savedmodel":
        path = base
        model.save(path)
    else:
        path = base + ".weights.h5"
        model.save_weights(path)
    logging.info(f"Модель збережено: {path}")


def confusion_and_examples(best_model, x_test, y_test, out_cm_png, out_examples_png, sklearn=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_pred = np.argmax(best_model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(values_format='d', cmap='Blues')
    plt.title("Confusion matrix (best model)")
    plt.tight_layout()
    plt.savefig(out_cm_png, dpi=160)
    plt.close()
    logging.info(f"Матриця похибок збережена: {out_cm_png}")

    wrong_idx = np.where(y_pred != y_test)[0][:12]
    if len(wrong_idx) > 0:
        plt.figure(figsize=(8,6))
        for i, idx in enumerate(wrong_idx):
            plt.subplot(3,4,i+1)
            plt.imshow(x_test[idx].squeeze(), cmap='gray')
            plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}")
            plt.axis('off')
        plt.suptitle("Misclassified examples")
        plt.tight_layout()
        plt.savefig(out_examples_png, dpi=160)
        plt.close()
        logging.info(f"Приклади помилок збережено: {out_examples_png}")
    else:
        logging.info("Немає помилок для прикладів (дуже крутий результат!)")

# ---------------------- Головна ----------------------
def main():
    # базова діагностика одразу в консоль
    print("TensorFlow:", tf.__version__, flush=True)
    print("GPU:", tf.config.list_physical_devices("GPU"), flush=True)

    parser = argparse.ArgumentParser(description="Lab1 AI: MNIST MLP vs CNN + activations")
    parser.add_argument("--epochs", type=int, default=5, help="епохи (варіант=5)")
    parser.add_argument("--batch", type=int, default=32, help="batch size (варіант=32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate для Adam")
    parser.add_argument("--subset", type=int, default=None, help="використати лише N зразків для швидкого прогону")
    parser.add_argument("--no-es", action="store_true", help="вимкнути EarlyStopping")
    parser.add_argument("--save-models", action="store_true", help="зберігати всі натреновані моделі")
    parser.add_argument("--models-dir", type=str, default="models", help="каталог для збереження моделей")
    parser.add_argument("--save-format", choices=["keras", "savedmodel", "weights"], default="keras",
                        help="формат збереження: .keras, SavedModel або лише ваги (.h5)")
    parser.add_argument("--plots", action="store_true", help="будувати графіки та матрицю похибок")
    args = parser.parse_args()

    run_tag = setup_logger()
    set_seed(42)

    # Дані
    (x_train, y_train), (x_test, y_test) = load_mnist(subset=args.subset)
    logging.info(f"x_train: {x_train.shape}, x_test: {x_test.shape}, subset={args.subset}")

    activations = ["relu", "leaky_relu", "tanh", "sigmoid"]
    results = {"MLP": {}, "CNN": {}}

    # --- MLP ---
    logging.info("\n================ MLP experiments ================\n")
    for act in activations:
        model = build_mlp(act)
        acc, _ = train_and_eval(
            model, x_train, y_train, x_test, y_test,
            batch=args.batch, epochs=args.epochs, lr=args.lr,
            use_es=not args.no_es, run_tag=run_tag
        )
        if args.save_models:
            save_model(model, run_tag, args.models_dir, args.save_format)

        results["MLP"][act] = acc

    # --- CNN ---
    logging.info("\n================ CNN experiments ================\n")
    best_cnn_model = None
    best_cnn_act = None
    best_cnn_acc = -1.0

    for act in activations:
        model = build_cnn(act)
        acc, _ = train_and_eval(
            model, x_train, y_train, x_test, y_test,
            batch=args.batch, epochs=args.epochs, lr=args.lr,
            use_es=not args.no_es, run_tag=run_tag
        )
        if args.save_models:
            save_model(model, run_tag, args.models_dir, args.save_format)

        results["CNN"][act] = acc
        if acc > best_cnn_acc:
            best_cnn_acc = acc
            best_cnn_act = act
            best_cnn_model = model

    best_path_dir = os.path.join(args.models_dir, "best")
    os.makedirs(best_path_dir, exist_ok=True)
    save_model(best_cnn_model, run_tag, best_path_dir, args.save_format)

    try:
        import pandas as pd
        df = (pd.DataFrame(results) * 100).round(2)
        print("\nAccuracy by activation, %:\n")
        print(df.to_string(), flush=True)
        csv_out = os.path.join("runs", f"results_{run_tag}.csv")
        df.to_csv(csv_out, index=True)
        logging.info(f"Результати збережені у {csv_out}")
    except Exception as e:
        logging.warning(f"pandas недоступний ({e}), вивід JSON:")
        print(json.dumps(results, indent=2), flush=True)

    # Кращі
    mlp_best_act = max(results["MLP"], key=results["MLP"].get)
    cnn_best_act = max(results["CNN"], key=results["CNN"].get)
    logging.info(f"Best MLP: {mlp_best_act} = {results['MLP'][mlp_best_act]*100:.2f}%")
    logging.info(f"Best CNN: {cnn_best_act} = {results['CNN'][cnn_best_act]*100:.2f}%")
    if args.plots:
        acc_png = os.path.join("runs", f"acc_by_activation_{run_tag}.png")
        df = plot_accuracy_bar(results, acc_png)

        # Матриця похибок та приклади — для найкращої CNN
        cm_png = os.path.join("runs", f"confusion_{cnn_best_act}_{run_tag}.png")
        ex_png = os.path.join("runs", f"errors_{cnn_best_act}_{run_tag}.png")
        confusion_and_examples(best_cnn_model, x_test, y_test, cm_png, ex_png)

if __name__ == "__main__":
    main()
