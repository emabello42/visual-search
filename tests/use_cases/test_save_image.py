import pytest
from visualsearch.use_cases import save_image as uc
import visualsearch.request_objects as req
import os
from unittest import mock
import numpy as np
from visualsearch.domain.image import Image
import uuid
import logging

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../testdata")


@pytest.fixture
def domain_images():
    img1 = Image(code=uuid.uuid4(), path=os.path.join(FIXTURE_DIR, "img1.jpg"),
                 unit_features=np.arange(2048),
                 magnitude=100.1)
    img2 = Image(code=uuid.uuid4(), path=os.path.join(FIXTURE_DIR, "img2.jpg"),
                 unit_features=np.arange(2048),
                 magnitude=90.8)

    img3 = Image(code=uuid.uuid4(), path=os.path.join(FIXTURE_DIR, "img3.jpg"),
                 unit_features=np.arange(2048),
                 magnitude=91.8)

    return [img1, img2, img3]

#
# def test_save_image(domain_images):
#     repo = mock.Mock()
#     feature_extractor = mock.Mock()
#
#     file_path = domain_images[0].path
#
#     # expected return values for mocked functions
#     repo.save_image.return_value = 1
#     feature_extractor.process_image.return_value = domain_images[0]
#
#     use_case_save_image = uc.SaveImage(repo, feature_extractor)
#     request_object = req.ImageRequestObject.from_dict({'params': {'path': file_path}})
#     response = use_case_save_image.execute(request_object)
#     logging.debug(response.value)
#     assert bool(response) is True
#     feature_extractor.process_image.assert_called_with(file_path)
#     repo.save_image.assert_called_with(domain_images[0])
#     assert response.value == 1


def test_save_image_all(domain_images):
    repo = mock.Mock()
    feature_extractor = mock.Mock()

    # expected return values
    repo.save_image.return_value = 1
    repo.close_save_batch_process.return_value = 3
    feature_extractor.process_batch.return_value = domain_images
    image_data = [1,2,3]
    feature_extractor.get_image_data.return_value = image_data
    request_object = req.ImageRequestObject.from_dict({'params': {'path': FIXTURE_DIR}})
    use_case_save_image = uc.SaveImage(repo, feature_extractor)
    response = use_case_save_image.execute(request_object)
    logging.debug(response.value)
    assert bool(response) is True
    feature_extractor.process_batch.assert_called_with(image_data)
    assert response.value == 3
