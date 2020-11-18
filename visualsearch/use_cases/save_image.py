from visualsearch.domain.image import Image
import visualsearch.response_objects as res
import os
import threading
import queue
import uuid

class SaveImage():
    def __init__(self, repository, feature_extractor):
        self.repo = repository
        self.feature_extractor = feature_extractor

    def execute(self, request_obj):
        if not request_obj:
            return res.ResponseFailure.build_from_invalid_request_object(request_obj)

        try:
            path = request_obj.params['path']
            if os.path.isfile(path):
                saved_images_count = self.__save_image(file_path = path)
                return res.ResponseSuccess(saved_images_count)
            elif os.path.isdir(path):
                saved_images_count = self.__save_image_all(dir_path = path)
                return res.ResponseSuccess(saved_images_count)
            else:
                raise Exception("File or directory {} does not exist".format(path))
        except Exception as exc:
            return res.ResponseFailure.build_system_error(
                    "{}: {}".format(exc.__class__.__name__, "{}".format(exc)))


    def __save_image(self, file_path):
        data = self.feature_extractor.process_image(file_path)
        new_image = Image(
                        id = uuid.uuid4(),
                        path = file_path,
                        unit_features = data.unit_features,
                        magnitude = data.magnitude
                        )
        saved_images_count = self.repo.save_image(new_image)
        return saved_images_count

    def __async_save_image(self):
        while True:
            data = self.queueImgFeatures.get()
            new_image = Image(
                            id = uuid.uuid4(),
                            path = data.path,
                            unit_features = data.unit_features,
                            magnitude = data.magnitude
                            )
            self.saved_images_counter += self.repo.save_image(new_image)
            self.queueImgFeatures.task_done()

    def __save_image_all(self, dir_path):
        self.queueImgFeatures = queue.Queue(10000)
        self.saved_images_counter = 0

        # turn-on the worker thread
        threading.Thread(target=self.__async_save_image, daemon=True).start()
        for img_features in self.feature_extractor.process_batch(dir_path):
            self.queueImgFeatures.put(img_features)

        # block until all tasks are done
        self.queueImgFeatures.join()
        return self.saved_images_counter
