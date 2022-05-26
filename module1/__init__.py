from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt

db = SQLAlchemy()
bcrypt = Bcrypt()


def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    app.config['SECRET_KEY'] = 'thisisasecretkey'

    db.init_app(app)
    bcrypt.init_app(app)

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    from .models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    from .auth import bp_auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    from .summary import bp_summary as summary_blueprint
    app.register_blueprint(summary_blueprint)

    from .annotation import bp_annotation as annotation_blueprint
    app.register_blueprint(annotation_blueprint)

    from .configuration import bp_configuration as configuration_blueprint
    app.register_blueprint(configuration_blueprint)

    from .policies import bp_policies as policies_blueprint
    app.register_blueprint(policies_blueprint)

    return app
