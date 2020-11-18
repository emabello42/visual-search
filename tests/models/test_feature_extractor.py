from visualsearch.models import feature_extractor as fe
import os
import pytest
import numpy as np

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../testdata")

@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, 'img1.jpg'))
def test_process_image(datafiles):
    path = str(datafiles)
    feature_extractor = fe.FeatureExtractor()
    input_image = os.path.join(path, "img1.jpg")
    features = feature_extractor.process_image(path = input_image)
    assert np.linalg.norm(features.unit_features) == pytest.approx(1.0)
    assert features.magnitude >= 0
    assert features.label >= 0
    assert 0.0 <= features.score <= 1.0
