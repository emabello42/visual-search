import pytest
from visualsearch.use_cases import save_image as uc
import visualsearch.request_objects as req
import os
from unittest import mock

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../testdata")

@pytest.mark.datafiles(
        os.path.join(FIXTURE_DIR, 'img1.jpg'),
        os.path.join(FIXTURE_DIR, 'img2.jpg'),
        os.path.join(FIXTURE_DIR, 'img3.jpg'),
        )

def test_save_image(datafiles):
    path = str(datafiles)
    repo = mock.Mock()
    feature_extractor = mock.Mock()

    fake_features = {
            'example_feat1': "test",
            'example_feat2': 40,
            'example_feat3': 0.8
            }
    input_image = os.path.join(path, "img1.jpg")
    feature_extractor.process_image.return_value = fake_features
    repo.save_image.return_value = 1
    save_image_uc = uc.SaveImage(repository = repo, feature_extractor = feature_extractor)
    request_object = req.ImageRequestObject.from_dict({'params':{'path': input_image}})
    response = save_image_uc.execute(request_object)
    assert bool(response) is True
    repo.save_image.assert_called_with(image_data = {'path': input_image, **fake_features})
    feature_extractor.process_image.assert_called_with(path = input_image)
    assert response.value == 1
