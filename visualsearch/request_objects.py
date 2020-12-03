import collections
from visualsearch.utils import from_bytes_to_image


class InvalidRequestObject:

    def __init__(self):
        self.errors = []

    def add_error(self, parameter, message):
        self.errors.append({'parameter': parameter, 'message': message})

    def has_errors(self):
        return len(self.errors) > 0

    def __bool__(self):
        return False


class ValidRequestObject:

    @classmethod
    def from_dict(cls, adict):
        raise NotImplementedError

    def __bool__(self):
        return True


class ImageRequestObject(ValidRequestObject):
    accepted_params = set({'path'})

    def __init__(self, params=None):
        self.params = params

    @classmethod
    def from_dict(cls, adict):
        invalid_req = InvalidRequestObject()

        if 'params' in adict:
            if not isinstance(adict['params'], collections.abc.Mapping):
                invalid_req.add_error('params', "Is not iterable")
                return invalid_req

            for key, value in adict['params'].items():
                if key not in cls.accepted_params:
                    invalid_req.add_error('params', "Key {} cannot be used".format(key))

        if invalid_req.has_errors():
            return invalid_req

        return cls(params=adict.get('params', None))


class FindSimilaritiesRequestObject:

    def __init__(self, image):
        self.image = image

    # @classmethod
    # def from_dict(cls, adict):
    #     invalid_req = InvalidRequestObject()
    #
    #     if 'params' in adict:
    #         if not isinstance(adict['params'], collections.abc.Mapping):
    #             invalid_req.add_error('params', "Is not iterable")
    #             return invalid_req
    #
    #         for key, value in adict['params'].items():
    #             if key not in cls.accepted_params:
    #                 invalid_req.add_error('params', "Key {} cannot be used".format(key))
    #
    #         if 'image' not in adict['params']:
    #             invalid_req.add_error('params', "An image must be provided")
    #
    #     if invalid_req.has_errors():
    #         return invalid_req
    #
    #     return cls(params=adict.get('params'))

    @classmethod
    def from_bytes(cls, byte_data):
        try:
            return cls(image=from_bytes_to_image(byte_data))
        except Exception as exp:
            invalid_req = InvalidRequestObject()
            invalid_req.add_error('image', "An image must be provided")
            return invalid_req


