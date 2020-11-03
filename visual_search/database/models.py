from ..application import db

class Category(db.Model):
    __tablename__ = 'categories'
    
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String)
    images = db.relationship("Image", back_populates="category")
    
class Image(db.Model):
    __tablename__ = 'images'
    
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String)
    unit_features = db.Column(db.ARRAY(db.Float))
    magnitude = db.Column(db.Float)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id'))
    category = db.relationship("Category", back_populates="images")