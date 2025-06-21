from pathlib import Path

from common import get_dataset
from dcgan import DCGAN

images = get_dataset()

model = DCGAN(Path("dcgan/"))
model.compile()

history = model.fit(images, callbacks=model.callbacks, epochs=1000)
