import os
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser

import numpy as np
import seaborn as sb
import tensorflow as tf
from matplotlib import pyplot as plt


DATA_PATH = Path("data/")


def show(images, labels=None):
    """
    Plots and displays the images and labels.
    """

    for i in range(5):
        for j in range(5):

            idx = i * 5 + j
            plt.subplot(5, 5, idx + 1)
            plt.axis(False)
            plt.imshow(images[idx], cmap="gray")

            if labels is not None:
                plt.title(labels[idx])
    plt.tight_layout()
    plt.show()


def generate_label(gen, label, initial_confidence, n):
    """
    Generates n random digits with label from gen.
    The process starts with the initial_confidence level.
    :return: np.ndarray of images
    """

    samples = []

    curr_confidence = initial_confidence

    while len(samples) < n:

        # generate random noise
        z = tf.random.normal(
            (
                n * 10,
                latent_dim,
            )
        )

        # generate images
        generated = gen(z)
        generated = (generated.numpy().reshape((-1, 28, 28)) * 255).astype("uint8")

        # classify images
        labels = classifier(generated)
        labels = labels.numpy()

        # select only samples with confidence level > current confidence
        confidences = labels.max(-1)
        mask = confidences > curr_confidence

        # transform one-hot encoded vectors in labels
        labels = labels.argmax(-1)

        # select the images and labels
        generated = generated[mask]
        labels = labels[mask]

        # sb.displot(labels, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # plt.show()

        # show(generated, labels)

        # sb.displot(labels, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # select only images with label label
        label_mask = labels == label
        generated = generated[label_mask]

        for img in generated:
            samples.append(img)

        if len(generated) == 0:
            curr_confidence *= 0.9

    return np.array(samples)


if __name__ == "__main__":
    argparser = ArgumentParser("generate_images.py")
    argparser.add_argument(
        "genpath",
        help="The path to the generator.",
        type=Path,
    )
    argparser.add_argument(
        "--latent-dim", help="The size of latent dimension.", type=int, default=100
    )
    argparser.add_argument(
        "--images-per-class",
        help="The number of images per class, that will be generated.",
        type=int,
        default=100,
    )
    argparser.add_argument(
        "--confidence",
        help="The level of confidences of the models.",
        type=float,
        default=0.8,
    )

    argv = argparser.parse_args()
    genpath = argv.genpath
    latent_dim = argv.latent_dim
    ipc = argv.images_per_class
    confidence_level = argv.confidence

    gen = tf.keras.models.load_model(genpath)

    classifier = tf.keras.models.load_model("../mnist-classification/classifier.keras")

    plt.figure(figsize=(10, 4))
    plt.title("Images")
    plt.axis(False)

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.axis(False)

        gen_subset = generate_label(gen, i, confidence_level, ipc)

        if len(gen_subset) != 0:
            plt.imshow(gen_subset[0], cmap="gray")
            plt.title(i)

        for j in range(ipc):
            classdir = DATA_PATH / str(i)
            filename = f"{j}.png"
            os.makedirs(classdir, exist_ok=True)

            img = Image.fromarray(gen_subset[j])
            img.save(classdir / filename)

    plt.show()
