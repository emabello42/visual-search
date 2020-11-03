from ..application import db, DBConfig
from sqlalchemy_utils import create_database
import ast
from models import Category

def init_database():
    db.create_all()

    # create image categories
    with open("../imagenet.txt", "r") as f:
        dict_categories = ast.literal_eval(f.read())
        categories = []
        for k,v in dict_categories.items():
            categories.append(Category(id= k, description=v))
        
        db.session.bulk_save_objects(categories)
        db.session.commit()


if __name__ == "__main__":
    # create_database(DBConfig.get_uri())
    init_database()