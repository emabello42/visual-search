
class Category:
    def __init__(self, id, label, description):
        self.id = id
        self.label = label
        self.description = description


class Image:
    def __init__(self, id, path, features, category, score):
        self.id = id
        self.path = path
        self.features = features
        self.category = category
        self.score = score
