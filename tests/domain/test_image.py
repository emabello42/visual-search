from visualsearch.domain import image as i
import numpy as np
import uuid

def test_image_model_init():
    img_id = uuid.uuid4()
    features = np.arange(2048)
    path = "/example/path"
    score = 0.9
    cat_id = uuid.uuid4()
    cat_description = "sample description"
    label = 90
    category = i.Category(id=cat_id, label=label, description=cat_description)
    image = i.Image(id=img_id, features=features, path=path, score=score, category=category)

    assert image.id == img_id
    assert not(False in (image.features == features)) # all values should match
    assert image.score == score
    assert image.path == path
    assert image.category == category
    assert category.id == cat_id
    assert category.label == label
    assert category.description == cat_description
