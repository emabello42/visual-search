import visualsearch.response_objects as res

class SaveImage():
    def __init__(self, repository, feature_extractor):
        self.repo = repository
        self.feature_extractor = feature_extractor

    def execute(self, request_obj):
        if not request_obj:
            return res.ResponseFailure.build_from_invalid_request_object(request_obj)

        try:
            features = self.feature_extractor.process_image(path = request_obj.params['path'])
            saved_images_count = self.repo.save_image(image_data = {'path': request_obj.params['path'], **features})
            return res.ResponseSuccess(saved_images_count)
        except Exception as exc:
            return res.ResponseFailure.build_system_error(
                    "{}: {}".format(exc.__class__.__name__, "{}".format(exc)))
