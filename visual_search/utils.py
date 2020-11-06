import io
import numpy as np

def adapt_array(arr):
    """ Adapts numpy array to be stored in string format in the database """

    output = io.BytesIO()
    np.savez_compressed(output, arr=arr) # in general large arrays are expected

    return output.getvalue()

def convert_array(string_data):
    """ Convert string stored in the database to numpy array representation """
    data = np.load(io.BytesIO(string_data))
    return data['arr']