from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    code = Column(String(36), nullable=False)
    path = Column(String)
    features_idx = Column(Integer)
    magnitude = Column(Float)
