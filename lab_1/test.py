import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras
import numpy as np
from PIL import Image

model = keras.models.load_model("models/cnn_relu_20251021-211539.keras")

def load_img_28x28(path):
    img = Image.open(path).convert("L").resize((28,28))
    x = np.array(img).astype("float32")/255.0
    x = x[None, ..., None]
    return x

x = load_img_28x28("img_1.png")
probs = model.predict(x, verbose=0)[0]
pred = probs.argmax()
print("pred:", pred, "prob:", float(probs[pred]))
