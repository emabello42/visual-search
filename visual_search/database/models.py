from application import db
from sqlalchemy.dialects.postgresql import UUID
import uuid


class Category(db.Model):
    __tablename__ = 'categories'

    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String)
    images = db.relationship("Image", back_populates="category")


class Image(db.Model):
    __tablename__ = 'images'

    id = db.Column(UUID(as_uuid=True), primary_key=True,
                   default=uuid.uuid4, unique=True, nullable=False)
    path = db.Column(db.String)
    unit_features_data = db.Column(db.String)
    magnitude = db.Column(db.Float)
    score = db.Column(db.Float)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id'))
    category = db.relationship("Category", back_populates="images")