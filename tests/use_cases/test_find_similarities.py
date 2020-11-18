import os
import logging
import pytest
from unittest import mock
from visualsearch.domain import image as i
from visualsearch.use_cases import find_similarities as uc
import visualsearch.request_objects as req
import uuid
import numpy as np
from visualsearch.models.feature_extractor import ImageFeatures

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../testdata")

@pytest.fixture
def domain_images():
    img1 = i.Image(id=uuid.uuid4(), path = os.path.join(FIXTURE_DIR, "img1.jpg"),
            unit_features = np.arange(2048),
            magnitude = 100.1)
    img2 = i.Image(id=uuid.uuid4(), path = os.path.join(FIXTURE_DIR, "img2.jpg"),
            unit_features = np.arange(2048),
            magnitude = 90.8)

    return [img1, img2]


@pytest.fixture
def img_features_list():
    img_feat1 = ImageFeatures(
                            unit_features = np.arange(2048),
                            magnitude = 101.1,
                            path = os.path.join(FIXTURE_DIR, 'img1.jpg')
                            )
    img_feat2 = ImageFeatures(
                            unit_features = np.arange(2048),
                            magnitude = 102.1,
                            path = os.path.join(FIXTURE_DIR, 'img2.jpg')
                            )

    img_feat3 = ImageFeatures(
                            unit_features = np.arange(2048),
                            magnitude = 103.1,
                            path = os.path.join(FIXTURE_DIR, 'img3.jpg')
                            )

    return [img_feat1, img_feat2, img_feat3]


def test_find_similarities_with_path(img_features_list, domain_images):
    # create feature_extractor and repo mocks
    repo = mock.Mock()
    feature_extractor = mock.Mock()
    repo.find_similars.return_value = domain_images
    feature_extractor.process_image.return_value = img_features_list[2]
    file_path = img_features_list[2].path
    request_object = req.FindSimilaritiesRequestObject.from_dict({'params':{'path': file_path}})
    fs_use_case = uc.FindSimilarities(repo, feature_extractor)
    response = fs_use_case.execute(request_object)
    logging.debug(response.value)
    assert bool(response) is True
    feature_extractor.process_image.assert_called_with(file_path)
    repo.find_similars.assert_called()
    assert response.value == domain_images
