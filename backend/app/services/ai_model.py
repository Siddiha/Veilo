from __future__ import annotations

from typing import List, Tuple


class LungCancerModel:
    def __init__(self) -> None:
        self.initialized = True

    def predict(self, preprocessed_image) -> Tuple[float, List[dict]]:
        probability = 0.42
        annotations = [
            {"x": 120, "y": 150, "width": 64, "height": 64, "label": "nodule"}
        ]
        return probability, annotations


