"""Common utils/help functions/consts for the project."""

from typing import Any
from pathlib import Path

from matplotlib import pyplot as plt
import tensorflow as tf  # type: ignore
import numpy as np


# Consts
LEAKY_RELU_ALPHA = 0.2
DROPOUT_RATE = 0.35
BATCH_NORM_MOMENTUM = 0.5
LEARNING_RATE = 0.0005
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999

WRITE_MODELS_AFTER_N_EPOCH = 50

DATASET_BASE_DIR = Path("../MNIST-ObjectDetection/data/mnist_detection/")


# Functions
def draw_images(images: np.ndarray, grid: tuple[int, int] = (5, 5)) -> Any:
    """
    Draws the images.

    :param images: numpy array of images with shape [N, h, w, c],
                    where:
                    N - number of images
                    h - height
                    w - width
                    c - number of channels
    :returns: plt canvas
    """

    grayscale = images.shape[-1] == 1
    fig = plt.figure(figsize=(10, 10))
    canvas = fig.canvas
    for i in range(grid[0]):
        for j in range(grid[1]):
            idx = i * grid[0] + j
            plt.subplot(grid[0], grid[1], idx + 1)
            plt.axis(False)
            if grayscale:
                plt.imshow(images[idx], cmap="gray")
            else:
                plt.imshow(images[idx])
    plt.tight_layout()
    fig.canvas
    canvas.draw()
    return canvas


def get_dataset() -> np.ndarray:
    """
    Returns the loaded Dataset from DATASET_BASE_DIR.
    Uses keras API to load the dataset from directory
    and then filters out the labels.
    """
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

    # normalize from -1 to 1
    images = (images / 255.0) * 2 - 1
    return images


# Classes
class BasicGAN(tf.keras.Model):
    """
    Abstraction for GAN classes.
    """

    def __init__(self, logdir: Path, latent_dim: int):
        super().__init__()

        # Defines basic callbacks for training

        self.logdir = logdir
        self.latent_dim = latent_dim
        self.file_writer = tf.summary.create_file_writer(str(logdir / "images"))
        self.write_images_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: self._visualize_callback(epoch)
        )
        self.write_models_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_begin=lambda epoch, logs: self._save_models(epoch)
        )

        self.callbacks = [
            tf.keras.callbacks.CSVLogger(logdir / "gan.csv"),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.TensorBoard(logdir),
            self.write_images_callback,
            self.write_models_callback,
        ]

    def _visualize_callback(self, epoch: int):
        """
        Visualizes the images and saves the output to tensorboard summary.

        :param epoch: epoch number
        """

        images = self.generate_random(25)
        canvas = draw_images(images)
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
        if epoch % WRITE_MODELS_AFTER_N_EPOCH == 0:
            plt.savefig(self.logdir / f"{epoch}.png")
        plt.close()
        with self.file_writer.as_default():
            tf.summary.image(
                "Generation Results", image.reshape((1,) + image.shape), epoch
            )

    def _save_models(self, epoch: int):
        """
        Saves both models (generator and discriminator) to the logdir

        :param epoch: epoch number
        """

        if epoch % WRITE_MODELS_AFTER_N_EPOCH == 0:
            self.gen.save(self.logdir / f"gen-{epoch}.keras")
            self.disc.save(self.logdir / f"disc-{epoch}.keras")

    def generate_random(self, n: int) -> tf.Tensor:
        """
        Generates random subset from generator.

        :param n: number of samples to generate (aka batch_size)
        :returns: np.ndarray output of the generator
        """
        raise NotImplementedError("This method must be implemented in every subclass")
