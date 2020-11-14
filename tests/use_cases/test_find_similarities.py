import os
import pytest
from unittest import mock
import numpy as np
from visualsearch.domain import image as i
from visualsearch.use_cases import find_similarities as uc
from visualsearch.request_objects import find_similarities as req
import uuid

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../testdata")

@pytest.fixture
def domain_images():
    category = i.Category(id=uuid.uuid4(), label = 40, description = "sample description")
    img1 = i.Image(id=uuid.uuid4(), path = os.path.join(FIXTURE_DIR, "img1.jpg"),
            features = np.arange(2048),
            category = category,
            score = 0.9)
    img2 = i.Image(id=uuid.uuid4(), path = os.path.join(FIXTURE_DIR, "img2.jpg"),
            features = np.arange(2048),
            category = category,
            score = 0.9)

    return [img1, img2]


@pytest.mark.datafiles(
        os.path.join(FIXTURE_DIR, 'img1.jpg'),
        os.path.join(FIXTURE_DIR, 'img2.jpg'),
        os.path.join(FIXTURE_DIR, 'img3.jpg')
        )
def test_find_similarities_with_path(datafiles, domain_images):
    # create model and repo mocks
    path = str(datafiles)  # Convert from py.path object to path (str)
    model = mock.Mock()
    fake_features = {
            'features': np.arange(2048),
            'label': 40,
            'score': 0.8
            }
    model.compute_features.return_value = fake_features
    repo = mock.Mock()
    repo.find_similars.return_value = domain_images
    
    input_image = os.path.join(path, "img1.jpg")
    request_object = req.FindSimilaritiesRequestObject.from_dict({'params':{'path': input_image}})
    fs_use_case = uc.FindSimilarities(repository = repo, model = model)
    response = fs_use_case.execute(request_object)
    assert bool(response) is True
    repo.find_similars.assert_called_with(features = fake_features)
    model.compute_features.assert_called_with(path = input_image)
    assert response.value == domain_images
