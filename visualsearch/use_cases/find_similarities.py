from visualsearch.response_objects import response_objects as res

class FindSimilarities():
    def __init__(self, repository, model):
        self.repo = repository
        self.model = model

    def execute(self, request_obj):
        if not request_obj:
            return res.ResponseFailure.build_from_invalid_request_object(request_obj)

        try:
            features = self.model.compute_features(path = request_obj.params['path'])
            images = self.repo.find_similars(features = features)
            return res.ResponseSuccess(images)
        except Exception as exc:
            return res.ResponseFailure.build_system_error(
                    "{}: {}".format(exc.__class__.__name__, "{}".format(exc)))
