import visualsearch.response_objects as res

class FindSimilarities():
    def __init__(self, repository):
        self.repo = repository

    def execute(self, request_obj):
        if not request_obj:
            return res.ResponseFailure.build_from_invalid_request_object(request_obj)

        try:
            images = self.repo.find_similars(file_path = request_obj.params['path'])
            return res.ResponseSuccess(images)
        except Exception as exc:
            return res.ResponseFailure.build_system_error(
                    "{}: {}".format(exc.__class__.__name__, "{}".format(exc)))
