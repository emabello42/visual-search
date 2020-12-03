import json
from flask import Blueprint, request, Response
import logging
from visualsearch.repository import postgresrepo as pg_repo
from visualsearch.use_cases import find_similarities as uc
from visualsearch.models import feature_extractor as fe
from visualsearch.serializers import SimilarityJsonEncoder
import visualsearch.response_objects as res
import visualsearch.request_objects as req
import numpy as np
from PIL import Image as PILImage

blueprint = Blueprint('find_similarities', __name__)

STATUS_CODES = {
    res.ResponseSuccess.SUCCESS: 200,
    res.ResponseFailure.SYSTEM_ERROR: 500
}

connection_data = {
    'dbname': "visualsearchdb",
    'user': "postgres",
    'password': "",
    'host': "localhost"
}

feat_file = "features.npy"


@blueprint.route('/find_similarities', methods=['POST'])
def similarities():
    req_obj = req.FindSimilaritiesRequestObject.from_bytes(request.data)
    repo = pg_repo.PostgresRepo(connection_data, feat_file)
    feat_extractor = fe.FeatureExtractor()
    use_case = uc.FindSimilarities(repo, feat_extractor)
    response = use_case.execute(req_obj)
    return Response(json.dumps(response.value, cls=SimilarityJsonEncoder),
                    mimetype='application/json',
                    status=STATUS_CODES[response.type])
