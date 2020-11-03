from application import db, application
import api.images
import sys

if __name__ == '__main__':
    db.create_all()
    db.session.commit()
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)