from visualsearch.models import feature_extractor as fe
import os
import pytest
import torch
from visualsearch.domain import image as i
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../testdata")


def test_process_image():
    feature_extractor = fe.FeatureExtractor()
    file_img = os.path.join(FIXTURE_DIR, "img1.jpg")
    features = feature_extractor.process_image(path=file_img)
    assert torch.norm(features.unit_features) == pytest.approx(1.0)
    assert features.magnitude >= 0
    assert features.path == file_img


def test_process_image_batch():
    feature_extractor = fe.FeatureExtractor()
    file_list = [
        os.path.join(FIXTURE_DIR, "img1.jpg"),
        os.path.join(FIXTURE_DIR, "img2.jpg"),
        os.path.join(FIXTURE_DIR, "img3.jpg")
    ]
    for output_batch, paths in feature_extractor.process_batch(path=FIXTURE_DIR):
        for idx, (unit_features, magnitude) in enumerate(zip(output_batch.unit_features, output_batch.magnitudes)):
            assert torch.norm(unit_features) == pytest.approx(1.0)
            assert magnitude.item() >= 0
            assert paths[idx] in file_list
