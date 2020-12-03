import visualsearch.response_objects as res
import os
import logging


class SaveImage:
    def __init__(self, repository, feature_extractor):
        self.repo = repository
        self.feature_extractor = feature_extractor

    def execute(self, request_obj):
        if not request_obj:
            return res.ResponseFailure.build_from_invalid_request_object(request_obj)

        try:
            path = request_obj.params['path']
            if os.path.isdir(path):
                saved_images_count = self.__save_image_all(dir_path=path)
                return res.ResponseSuccess(saved_images_count)
            else:
                raise Exception("Directory {} does not exist".format(path))
        except Exception as exc:
            return res.ResponseFailure.build_system_error(
                "{}: {}".format(exc.__class__.__name__, "{}".format(exc)))

    def __save_image_all(self, dir_path):
        self.repo.start_save_batch_process()
        for data in self.feature_extractor.process_batch(dir_path):
            self.repo.save_queue.put(data)
        cnt_saved_images = self.repo.close_save_batch_process()
        return cnt_saved_images
