class Image:
    def __init__(self, code, path, unit_features, magnitude):
        self.code = code
        self.path = path
        self.unit_features = unit_features
        self.magnitude = magnitude

    @classmethod
    def from_dict(cls, adict):
        return cls(
                code=adict['code'],
                path=adict['path'],
                unit_features=adict['unit_features'],
                magnitude=adict['magnitude'],
        )

    def to_dict(self):
        return {
            'code': self.code,
            'path': self.path,
            'unit_features': self.unit_features,
            'magnitude': self.magnitude,
        }

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()
