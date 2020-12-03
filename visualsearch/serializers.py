import json


class SimilarityJsonEncoder(json.JSONEncoder):

    def default(self, o):
        try:
            to_serialize = {
                'path': o.path,
                'score': o.score
            }

            return to_serialize
        except AttributeError:
            return super().default(o)
