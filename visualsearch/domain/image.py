
class Category:
    def __init__(self, id, label, description):
        self.id = id
        self.label = label
        self.description = description

    @classmethod
    def from_dict(cls, adict):
        return cls(
                id=adict['id'],
                label=adict['label'],
                description=adict['description']
        )

    def to_dict(self):
        return {
            'id': self.id,
            'label': self.label,
            'description': self.description
        }

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()


class Image:
    def __init__(self, id, path, unit_features, magnitude, category, score):
        self.id = id
        self.path = path
        self.unit_features = unit_features
        self.magnitude = magnitude
        self.category = category
        self.score = score

    @classmethod
    def from_dict(cls, adict):
        return cls(
                id=adict['id'],
                path=adict['path'],
                unit_features=adict['unit_features'],
                magnitude=adict['magnitude'],
                category=Category.from_dict(adict['category']),
                score=adict['score']
        )

    def to_dict(self):
        return {
            'id': self.id,
            'path': self.path,
            'unit_features': self.unit_features,
            'magnitude': self.magnitude,
            'category': self.category.to_dict(),
            'score': self.score
        }

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()
