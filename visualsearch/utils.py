import io
import numpy as np
import time
import numpy as np
from PIL import Image as PILImage


class ProcessingStats:
    def __init__(self):
        self.stats = {}

    def start(self, field):
        if not(field in self.stats):
            self.stats[field] = []
        self.stats[field].append(time.time())

    def end(self, field):
        if not(field in self.stats):
            raise Exception(str(field) + " stat is not initialized")

        self.stats[field][-1] = time.time() - self.stats[field][-1]

    def __str__(self):
        result = "Processing stats summary:\n"
        for field, measurements in self.stats.items():
            result += f"{field}: Avg = {np.mean(measurements)} s ; Sum = {np.sum(measurements)} s; Iterations = {len(measurements)}\n"
        return result


def adapt_array(arr):
    """ Adapts numpy array to be stored in binary format in the database """

    output = io.BytesIO()
    # in general large arrays are expected
    np.savez_compressed(output, arr=arr)

    return output.getvalue()


def convert_array(byte_data):
    """ Convert binary array stored in the database to numpy array representation """
    data = np.load(io.BytesIO(byte_data))
    return data['arr']


def from_bytes_to_image(byte_data):
    npimg = np.frombuffer(byte_data, np.uint8)
    return PILImage.fromarray(npimg).convert('RGB')
