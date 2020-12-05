import visualsearch.response_objects as res
import logging


class FindSimilarities():
    def __init__(self, repository, feature_extractor):
        self.repo = repository
        self.feature_extractor = feature_extractor

    def execute(self, request_obj):
        if not request_obj:
            return res.ResponseFailure.build_from_invalid_request_object(request_obj)

        try:
            features = self.feature_extractor.process_image(request_obj.image)
            similars = self.repo.find_similars(features)
            return res.ResponseSuccess(similars)
        except Exception as exc:
            return res.ResponseFailure.build_system_error(
                    "{}: {}".format(exc.__class__.__name__, "{}".format(exc)))
