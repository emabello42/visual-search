from visualsearch.domain import image as i
import numpy as np
import uuid

def test_image_model_init():
    id = uuid.uuid4()
    unit_features = np.arange(2048)
    magnitude = 100
    path = "/example/path"
    score = 0.9
    cat_id = uuid.uuid4()
    cat_description = "sample description"
    label = 90
    category = i.Category(id=cat_id, label=label, description=cat_description)
    image = i.Image(id=id, unit_features=unit_features, magnitude=magnitude,
            path=path, score=score, category=category)

    assert not(False in (image.unit_features == unit_features)) # all values should match
    assert image.id == id
    assert image.score == score
    assert image.path == path
    assert image.magnitude == magnitude
    assert image.category == category
    assert category.id == cat_id
    assert category.label == label
    assert category.description == cat_description
