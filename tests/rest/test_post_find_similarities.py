import json
from unittest import mock
from visualsearch.domain.similarity import Similarity
import visualsearch.response_objects as res
from visualsearch.utils import from_bytes_to_image
import os
import logging

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../testdata")

similarities_dict_list = [
    {'path': "/example/path1.jpg", 'score': 0.98},
    {'path': "/example/path2.jpg", 'score': 0.78},
    {'path': "/example/path3.jpg", 'score': 0.68},
    {'path': "/example/path4.jpg", 'score': 0.48},
    {'path': "/example/path5.jpg", 'score': 0.18}
]

similarity_list = [Similarity.from_dict(s) for s in similarities_dict_list]


@mock.patch('visualsearch.use_cases.find_similarities.FindSimilarities')
def test_post(mock_use_case, client):
    mock_use_case().execute.return_value = res.ResponseSuccess(similarity_list)
    # prepare headers for http request
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    img = open(os.path.join(FIXTURE_DIR, "img1.jpg"), 'rb').read()
    http_response = client.post('/find_similarities', data=img, headers=headers)
    assert json.loads(http_response.data.decode('UTF-8')) == similarities_dict_list
    mock_use_case().execute.assert_called()
    args, kwargs = mock_use_case().execute.call_args
    assert args[0].image == from_bytes_to_image(img)

    assert http_response.status_code == 200
    assert http_response.mimetype == "application/json"
