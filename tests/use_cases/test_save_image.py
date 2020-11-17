import pytest
from visualsearch.use_cases import save_image as uc
import visualsearch.request_objects as req
import os
from unittest import mock
import numpy as np

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../testdata")

@pytest.mark.datafiles(
        os.path.join(FIXTURE_DIR, 'img1.jpg'),
        os.path.join(FIXTURE_DIR, 'img2.jpg'),
        os.path.join(FIXTURE_DIR, 'img3.jpg'),
        )

def test_save_image(datafiles):
    path = str(datafiles)
    repo = mock.Mock()
    input_image = os.path.join(path, "img1.jpg")
    repo.save_image.return_value = 1
    save_image_uc = uc.SaveImage(repository = repo)
    request_object = req.ImageRequestObject.from_dict({'params':{'path': input_image}})
    response = save_image_uc.execute(request_object)
    assert bool(response) is True
    repo.save_image.assert_called_with(file_path = input_image) # the use case could make some optimizations
    assert response.value == 1

def test_save_image_all(datafiles):
    path = str(datafiles)
    repo = mock.Mock()

    repo.save_image_all.return_value = 3
    save_image_uc = uc.SaveImage(repository = repo)
    request_object = req.ImageRequestObject.from_dict({'params':{'path': path}})
    response = save_image_uc.execute(request_object)
    assert bool(response) is True
    repo.save_image_all.called_with(dir_path = path)
    assert response.value == 3
