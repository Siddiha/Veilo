from __future__ import annotations

import io
from PIL import Image


class ImagePreprocessor:
    def preprocess_bytes(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = image.resize((512, 512))
        return image


