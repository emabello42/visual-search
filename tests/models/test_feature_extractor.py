from visualsearch.models import feature_extractor as fe
import os
import pytest
import numpy as np

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../testdata")

def test_process_image():
    feature_extractor = fe.FeatureExtractor()
    file_img = os.path.join(FIXTURE_DIR, "img1.jpg")
    features = feature_extractor.process_image(path = file_img)
    assert np.linalg.norm(features.unit_features) == pytest.approx(1.0)
    assert features.magnitude >= 0
    assert features.path == file_img

def test_process_image_batch():
    feature_extractor = fe.FeatureExtractor()
    file_list = [
                os.path.join(FIXTURE_DIR, "img1.jpg"),
                os.path.join(FIXTURE_DIR, "img2.jpg"),
                os.path.join(FIXTURE_DIR, "img3.jpg")
                ]
    for features in feature_extractor.process_batch(path = FIXTURE_DIR):
        assert np.linalg.norm(features.unit_features) == pytest.approx(1.0)
        assert features.magnitude >= 0
        assert features.path in file_list
