import pytest
from visualsearch.use_cases import save_image as uc
import visualsearch.request_objects as req
import os
from unittest import mock
import numpy as np
from visualsearch.models.feature_extractor import ImageFeatures
from visualsearch.domain.image import Category, Image
import uuid
import logging

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../testdata")

@pytest.mark.datafiles(
        os.path.join(FIXTURE_DIR, 'img1.jpg'),
        os.path.join(FIXTURE_DIR, 'img2.jpg'),
        os.path.join(FIXTURE_DIR, 'img3.jpg'),
        )

@pytest.fixture
def categories():
    expected_category = Category(id=uuid.uuid4(), label=10, description="test")
    return [expected_category]

@pytest.fixture
def img_features_list(datafiles):
    img_feat1 = ImageFeatures(
                            unit_features = np.arange(2048),
                            magnitude = 101.1,
                            score = 0.8,
                            label = 10,
                            path = os.path.join(FIXTURE_DIR, 'img1.jpg')
                            )
    img_feat2 = ImageFeatures(
                            unit_features = np.arange(2048),
                            magnitude = 102.1,
                            score = 0.7,
                            label = 10,
                            path = os.path.join(FIXTURE_DIR, 'img2.jpg')
                            )

    img_feat3 = ImageFeatures(
                            unit_features = np.arange(2048),
                            magnitude = 103.1,
                            score = 0.4,
                            label = 10,
                            path = os.path.join(FIXTURE_DIR, 'img3.jpg')
                            )

    return [img_feat1, img_feat2, img_feat3]

def test_save_image(categories, img_features_list):
    repo = mock.Mock()
    feature_extractor = mock.Mock()

    file_path = img_features_list[0].path

    # fake image feature values
    unit_features = np.arange(2048)
    magnitude = 100
    score = 0.9
    label = 10

    # expected return values for mocked functions
    repo.find_categories.return_value = categories
    repo.save_image.return_value = 1
    feature_extractor.process_image.return_value = img_features_list[0] 

    use_case_save_image = uc.SaveImage(repo, feature_extractor)
    request_object = req.ImageRequestObject.from_dict({'params':{'path': file_path}})
    response = use_case_save_image.execute(request_object)
    logging.debug(response.value)
    assert bool(response) is True
    feature_extractor.process_image.assert_called_with(file_path)
    repo.find_categories.assert_called_with(label = label)
    repo.save_image.assert_called()
    assert response.value == 1

def test_save_image_all(categories, img_features_list):
    repo = mock.Mock()
    feature_extractor = mock.Mock()

    # expected return values
    repo.find_categories.return_value = categories
    repo.save_image.return_value = 1
    feature_extractor.process_batch.return_value = img_features_list 

    request_object = req.ImageRequestObject.from_dict({'params':{'path': FIXTURE_DIR}})
    use_case_save_image = uc.SaveImage(repo, feature_extractor)
    response = use_case_save_image.execute(request_object)
    logging.debug(response.value)
    assert bool(response) is True
    repo.find_categories.assert_called()
    assert repo.save_image.call_count == 3
    feature_extractor.process_batch.assert_called_with(FIXTURE_DIR)
    assert response.value == 3
