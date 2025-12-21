# app.py
from flask import Flask
from config import Config
from db import init_db, seed_admin

# -----------------------------
# IMPORT BLUEPRINTS
# -----------------------------
from auth import auth_bp
from user import user_bp
from market import market_bp

def create_app():
    # CREATE FLASK APP INSTANCE
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
    app.register_blueprint(auth_bp)   # /login, /logout, /change-password
    app.register_blueprint(user_bp)   # /user/*
    app.register_blueprint(market_bp) # /market/*

    # -----------------------------
    # PLACEHOLDER ROUTES (can replace with real dashboards later)
    # -----------------------------
    @app.route("/dashboard")
    def dashboard():
        return "USER Dashboard – PART 4"

    @app.route("/admin")
    def admin():
        return "ADMIN Dashboard – PART 3"

    return app

# -----------------------------
# CREATE APP INSTANCE
# -----------------------------
app = create_app()

# -----------------------------
# RUN DEV SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
