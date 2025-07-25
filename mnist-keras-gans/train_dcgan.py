from pathlib import Path
import sys

import tensorflow as tf

from common import get_dataset
from dcgan import DCGAN

images = get_dataset()

model = DCGAN(Path("dcgan/"))
model.compile()

if len(sys.argv) == 3:
    args = sys.argv
    genpath = args[1]
    discpath = args[2]
    model.gen = tf.keras.models.load_model(genpath)
    model.disc = tf.keras.models.load_model(discpath)

history = model.fit(images, callbacks=model.callbacks, epochs=1001)
