import visualsearch.response_objects as res
import os


class FindSimilarities():
    def __init__(self, repository, feature_extractor):
        self.repo = repository
        self.feature_extractor = feature_extractor

    def execute(self, request_obj):
        if not request_obj:
            return res.ResponseFailure.build_from_invalid_request_object(request_obj)

        try:
            file_path = request_obj.params['path']
            if os.path.isfile(file_path):
                query_image = self.feature_extractor.process_image(file_path)
                similars = self.repo.find_similars(query_image)
                return res.ResponseSuccess(similars)
            else:
                raise Exception("File or directory {} does not exist".format(file_path))
        except Exception as exc:
            return res.ResponseFailure.build_system_error(
                    "{}: {}".format(exc.__class__.__name__, "{}".format(exc)))
