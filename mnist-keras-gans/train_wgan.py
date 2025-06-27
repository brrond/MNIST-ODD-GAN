from pathlib import Path
import sys

import tensorflow as tf

from common import get_dataset
from wgan import WGAN

images = get_dataset()

model = WGAN(Path("wgan/"))
model.compile()

if len(sys.argv) == 3:
    args = sys.argv
    genpath = args[1]
    discpath = args[2]
    model.gen = tf.keras.models.load_model(genpath)
    model.disc = tf.keras.models.load_model(discpath)

# user different (smaller) batch size here to ensure, that the critic has some time to become slightly better than gen
# is necessary to make the training process a bit longer to implement n_critic logic from the paper
history = model.fit(images, callbacks=model.callbacks, epochs=3001, batch_size=8)
