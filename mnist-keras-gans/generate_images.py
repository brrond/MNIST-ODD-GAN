import os
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser

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

    argv = argparser.parse_args()
    genpath = argv.genpath
    latent_dim = argv.latent_dim
    ipc = argv.images_per_class
    n = ipc * 10

    gen = tf.keras.models.load_model(genpath)

    z = tf.random.normal(
        (
            n,
            latent_dim,
        )
    )
    generated = gen(z)
    generated = generated.numpy().reshape((n, 28, 28))

    classifier = tf.keras.models.load_model("../mnist-classification/classifier.keras")

    labels = classifier(generated)
    labels = labels.numpy()

    confidences = labels.max(-1)
    labels = labels.argmax(-1)

    sb.displot(labels, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.show()

    show(generated, labels)

    sb.displot(labels, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    plt.figure(figsize=(10, 4))
    plt.title("Images")
    plt.axis(False)

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.axis(False)

        idx = labels == i
        gen_subset = (generated[idx] * 255).astype("uint8")

        lab_subset = labels[idx]
        if len(gen_subset) != 0:
            plt.imshow(gen_subset[0], cmap="gray")
            plt.title(lab_subset[0])

        if len(gen_subset) < ipc:
            print(f"There are not enough samples for the {i} class.")

        for j in range(min(ipc, len(gen_subset))):
            classdir = DATA_PATH / str(i)
            filename = f"{j}.png"
            os.makedirs(classdir, exist_ok=True)

            img = Image.fromarray(gen_subset[j])
            img.save(classdir / filename)

    plt.show()
