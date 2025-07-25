from pathlib import Path

import tensorflow as tf  # type: ignore

from common import (
    LEAKY_RELU_ALPHA,
    BATCH_NORM_MOMENTUM,
    LEARNING_RATE,
    DROPOUT_RATE,
    BasicGAN,
    ADAM_BETA_1,
    ADAM_BETA_2,
)


class GAN(BasicGAN):
    """
    The most basic GAN possible.
    Original paper: https://arxiv.org/abs/1406.2661
    """

    @staticmethod
    def create_discriminator(input_shape=(28 * 28 * 1,)) -> tf.keras.Sequential:
        """
        Creates a simple GAN discriminator.

        :returns: tf.keras.Sequential model of the discriminator
        """

        assert len(input_shape) == 1

        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Dense(512),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.Dense(256),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ],
            name="MNIST_GAN_Discriminator",
        )

    @staticmethod
    def create_generator(
        latent_dim=(100,), output_shape=(28 * 28 * 1,)
    ) -> tf.keras.Sequential:
        """
        Creates a simple GAN generator.

        :returns: tf.keras.Sequential model of the generator
        """

        assert len(latent_dim) == 1
        assert len(output_shape) == 1

        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=latent_dim),
                tf.keras.layers.Dense(256),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.Dense(512),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM),
                tf.keras.layers.Dense(1024),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM),
                tf.keras.layers.Dense(output_shape[0], activation="tanh"),
            ],
            name="MNIST_GAN_Generator",
        )

    def __init__(self, logdir: Path, latent_dim=100):
        super().__init__(logdir, latent_dim)

        self.latent_dim = latent_dim
        self.latent_dim_shape = (latent_dim,)
        self.disc = GAN.create_discriminator()
        self.gen = GAN.create_generator((self.latent_dim,))

        self.bc = tf.keras.losses.BinaryCrossentropy()

        self.gen_optimizer = tf.keras.optimizers.Adam(
            LEARNING_RATE, ADAM_BETA_1, ADAM_BETA_2
        )
        self.disc_optimizer = tf.keras.optimizers.Adam(
            LEARNING_RATE, ADAM_BETA_1, ADAM_BETA_2
        )

    def discriminator_loss(self, real, fake):
        real_loss = self.bc(tf.ones_like(real), real)
        fake_loss = self.bc(tf.zeros_like(fake), fake)
        return (real_loss + fake_loss) / 2.0

    def generator_loss(self, fake):
        return self.bc(tf.ones_like(fake), fake)

    @tf.function
    def train_step(self, real_images):
        shape = tf.shape(real_images)
        rand = tf.random.normal((shape[0], self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.gen(rand, training=True)

            out = self.disc(real_images, training=True)
            fake_out = self.disc(fake_images, training=True)

            g_loss = self.generator_loss(fake_out)
            d_loss = self.discriminator_loss(out, fake_out)

        gen_grads = gen_tape.gradient(g_loss, self.gen.trainable_weights)
        disc_grads = disc_tape.gradient(d_loss, self.disc.trainable_weights)
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.disc.trainable_weights)
        )
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.gen.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}

    def generate_random(self, n: int) -> tf.Tensor:
        rand = tf.random.normal((n, self.latent_dim, 1))
        generated = self.gen(rand, training=False)
        shape = tf.shape(generated)
        return tf.reshape(generated, (shape[0], 28, 28, 1))
