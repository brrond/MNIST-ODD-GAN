from pathlib import Path

from common import get_dataset
from gan import GAN

images = get_dataset()
images = images.reshape((-1, 28 * 28, 1))

model = GAN(Path("gan/"))
model.compile()

history = model.fit(images, callbacks=model.callbacks, epochs=1000)
