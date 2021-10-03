import uuid

import numpy as np

from visualsearch.domain import image as i


def test_image_model_init():
    code = uuid.uuid4()
    unit_features = np.arange(2048)
    magnitude = 100
    path = "/example/path"
    image = i.Image(code=code, unit_features=unit_features, magnitude=magnitude, path=path)

    assert not (False in (image.unit_features == unit_features))  # all values should match
    assert image.code == code
    assert image.path == path
    assert image.magnitude == magnitude
