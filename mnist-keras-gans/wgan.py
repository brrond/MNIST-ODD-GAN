from pathlib import Path

import numpy as np
import tensorflow as tf  # type: ignore

from common import (
    LEAKY_RELU_ALPHA,
    BATCH_NORM_MOMENTUM,
    LEARNING_RATE,
    DROPOUT_RATE,
    BasicGAN,
)


class WGAN(BasicGAN):
    """
    Wasserstein GAN.
    Original paper: https://arxiv.org/abs/1701.07875
    """

    @staticmethod
    def create_critic(input_shape=(28, 28, 1)) -> tf.keras.Sequential:
        """
        Creates a critic.

        :returns: tf.keras.Sequential model of the critic
        """

        assert len(input_shape) == 3

        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Conv2D(32, 3, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.Conv2D(64, 3, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.Conv2D(128, 3, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.Conv2D(256, 3, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1),
            ],
            name="MNIST_WGAN_Critic",
        )

    @staticmethod
    def create_generator(
        latent_dim=(128,), output_shape=(28, 28, 1)
    ) -> tf.keras.Sequential:
        """
        Creates a generator.

        :returns: tf.keras.Sequential model of the generator
        """

        assert len(latent_dim) == 1
        assert len(output_shape) == 3

        gen = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=latent_dim),
                tf.keras.layers.Dense(7 * 7 * 256, use_bias=False),
                tf.keras.layers.Reshape((7, 7, 256)),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.Conv2DTranspose(
                    1024, 5, strides=1, padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.Conv2DTranspose(
                    512, 5, strides=2, padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.Conv2DTranspose(
                    256, 5, strides=2, padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.Conv2DTranspose(
                    128, 5, strides=1, padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.Conv2DTranspose(
                    64, 5, strides=1, padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM),
                tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA),
                tf.keras.layers.Conv2DTranspose(
                    1, 5, strides=1, padding="same", activation="tanh", use_bias=False
                ),
            ],
            name="MNIST_WGAN_Generator",
        )

        assert gen.output_shape == (None, 28, 28, 1), gen.output_shape
        return gen

    def __init__(self, logdir: Path, latent_dim=100):
        super().__init__(logdir, latent_dim)

        self.latent_dim = latent_dim
        self.critic = WGAN.create_critic()
        self.disc = self.critic  # necessary for BasicGAN to save critic model
        self.gen = WGAN.create_generator((self.latent_dim,))

        # params from the paper
        self.n_critic = 5
        self.clip_value = 0.1
        self.gen_optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)
        self.critic_optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)

        # secondary params
        self.prev_g_loss = 0.0
        self.train_step_counter = 0

    def critic_loss(self, real, fake):
        real_loss = self.wasserstein_loss(tf.ones_like(real), real)
        fake_loss = self.wasserstein_loss(-tf.ones_like(fake), fake)
        return (real_loss + fake_loss) * 0.5

    def generator_loss(self, fake):
        return self.wasserstein_loss(tf.ones_like(fake), fake)

    def wasserstein_loss(self, y_true, y_pred):
        return tf.keras.backend.mean(y_true * y_pred)

    @tf.function
    def train_step(self, real_images):
        self.train_step_counter += 1

        shape = tf.shape(real_images)
        rand = tf.random.normal((shape[0], self.latent_dim))

        with tf.GradientTape() as critic_tape:
            fake_images = self.gen(rand, training=False)

            out = self.critic(real_images, training=True)
            fake_out = self.critic(fake_images, training=True)

            c_loss = self.critic_loss(out, fake_out)

        critic_grads = critic_tape.gradient(c_loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_weights)
        )

        for var in self.critic.trainable_variables:
            var.assign(tf.clip_by_value(var, -self.clip_value, self.clip_value))

        if self.train_step_counter % self.n_critic:
            with tf.GradientTape() as gen_tape:
                fake_images = self.gen(rand, training=True)

                fake_out = self.critic(fake_images, training=False)

                self.prev_g_loss = self.generator_loss(fake_out)

            gen_grads = gen_tape.gradient(self.prev_g_loss, self.gen.trainable_weights)
            self.gen_optimizer.apply_gradients(
                zip(gen_grads, self.gen.trainable_weights)
            )

        return {"c_loss": c_loss, "g_loss": self.prev_g_loss}

    def generate_random(self, n: int) -> tf.Tensor:
        rand = tf.random.normal((n, self.latent_dim))
        return self.gen(rand, training=False)
