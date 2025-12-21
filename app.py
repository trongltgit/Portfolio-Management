# app.py
from flask import Flask
from config import Config
from db import init_db, seed_admin

# IMPORT AUTH MODULE (PART 2)
from auth import auth_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # -----------------------------
    # INIT DATABASE
    # -----------------------------
    with app.app_context():
        init_db()
        seed_admin()

    # -----------------------------
    # REGISTER BLUEPRINTS
    # -----------------------------
    app.register_blueprint(auth_bp)

    # -----------------------------
    # PLACEHOLDER ROUTES
    # (PART 3+ sẽ thay thế)
    # -----------------------------
    @app.route("/dashboard")
    def dashboard():
        return "USER Dashboard – PART 4"

    @app.route("/admin")
    def admin():
        return "ADMIN Dashboard – PART 3"

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
