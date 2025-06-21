from pathlib import Path

from matplotlib import pyplot as plt
import tensorflow as tf  # type: ignore
import numpy as np
from gan import GAN

DATASET_BASE_DIR = Path("../MNIST-ObjectDetection/data/mnist_detection/")

dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_BASE_DIR / "train" / "classification",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=(28, 28),
    shuffle=True,
    seed=42,
)
dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_BASE_DIR / "test" / "classification",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=(28, 28),
    shuffle=True,
    seed=42,
)
dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_BASE_DIR / "validation" / "classification",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=(28, 28),
    shuffle=True,
    seed=42,
)

dataset_combined = dataset_train.concatenate(dataset_test).concatenate(
    dataset_validation
)

images_list = []
for batch in dataset_combined:
    x, y = batch
    for img in x:
        images_list.append(img)
images = np.array(images_list)
images = images.reshape((-1, 28 * 28, 1))

# normalize from -1 to 1
images = (images / 255.0) * 2 - 1

model = GAN(Path("gan/"))
model.compile()

history = model.fit(images, callbacks=model.callbacks, epochs=1000)
