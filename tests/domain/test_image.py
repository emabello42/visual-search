from visualsearch.domain import image as i
import numpy as np
import uuid

def test_image_model_init():
    id = uuid.uuid4()
    unit_features = np.arange(2048)
    magnitude = 100
    path = "/example/path"
    image = i.Image(id=id, unit_features=unit_features, magnitude=magnitude, path=path)

    assert not(False in (image.unit_features == unit_features)) # all values should match
    assert image.id == id
    assert image.path == path
    assert image.magnitude == magnitude
