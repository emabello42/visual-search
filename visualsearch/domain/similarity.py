class Similarity:
    def __init__(self, path, score):
        self.path = path
        self.score = score

    @classmethod
    def from_dict(cls, adict):
        return cls(
                path=adict['path'],
                score=adict['score'],
        )

    def to_dict(self):
        return {
            'path': self.path,
            'score': self.score
        }

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()
