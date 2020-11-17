import visualsearch.response_objects as res
import os
import threading

class SaveImage():
    def __init__(self, repository):
        self.repo = repository

    def execute(self, request_obj):
        if not request_obj:
            return res.ResponseFailure.build_from_invalid_request_object(request_obj)

        try:
            path = request_obj.params['path']
            if os.path.isfile(path):
                saved_images_count = self.repo.save_image(file_path = path)
                return res.ResponseSuccess(saved_images_count)
            elif os.path.isdir(path):
                saved_images_count = self.repo.save_image_all(dir_path = path)
                return res.ResponseSuccess(saved_images_count)
            else:
                raise Exception("File or directory {} does not exist".format(path))
        except Exception as exc:
            return res.ResponseFailure.build_system_error(
                    "{}: {}".format(exc.__class__.__name__, "{}".format(exc)))
