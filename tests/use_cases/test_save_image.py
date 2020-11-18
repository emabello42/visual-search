import pytest
from visualsearch.use_cases import save_image as uc
import visualsearch.request_objects as req
import os
from unittest import mock
import numpy as np
from visualsearch.models.feature_extractor import ImageFeatures
from visualsearch.domain.image import Image
import uuid
import logging

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../testdata")



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

def test_save_image(img_features_list):
    repo = mock.Mock()
    feature_extractor = mock.Mock()

    file_path = img_features_list[0].path

    # fake image feature values
    unit_features = np.arange(2048)
    magnitude = 100

    # expected return values for mocked functions
    repo.save_image.return_value = 1
    feature_extractor.process_image.return_value = img_features_list[0] 

    use_case_save_image = uc.SaveImage(repo, feature_extractor)
    request_object = req.ImageRequestObject.from_dict({'params':{'path': file_path}})
    response = use_case_save_image.execute(request_object)
    logging.debug(response.value)
    assert bool(response) is True
    feature_extractor.process_image.assert_called_with(file_path)
    repo.save_image.assert_called()
    assert response.value == 1

def test_save_image_all(img_features_list):
    repo = mock.Mock()
    feature_extractor = mock.Mock()

    # expected return values
    repo.save_image.return_value = 1
    feature_extractor.process_batch.return_value = img_features_list 

    request_object = req.ImageRequestObject.from_dict({'params':{'path': FIXTURE_DIR}})
    use_case_save_image = uc.SaveImage(repo, feature_extractor)
    response = use_case_save_image.execute(request_object)
    logging.debug(response.value)
    assert bool(response) is True
    assert repo.save_image.call_count == 3
    feature_extractor.process_batch.assert_called_with(FIXTURE_DIR)
    assert response.value == 3
