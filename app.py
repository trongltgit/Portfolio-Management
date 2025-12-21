# app.py
from flask import Flask, redirect, url_for
from config import Config
from db import init_db, seed_admin

# -----------------------------
# IMPORT BLUEPRINTS
# -----------------------------
from auth import auth_bp
from user import user_bp
from market import market_bp

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
    app.register_blueprint(auth_bp)   # /login, /logout, /change-password
    app.register_blueprint(user_bp)   # /user/*
    app.register_blueprint(market_bp) # /market/*

    # -----------------------------
    # PLACEHOLDER ROUTES
    # -----------------------------
    @app.route("/")
    def index():
        # Redirect root URL vào login page
        return redirect(url_for("auth.login"))  # auth_bp phải có route /login với endpoint 'login'

    @app.route("/dashboard")
    def dashboard():
        # Dashboard chỉ hiển thị sau khi login
        return "USER Dashboard – PART 4"

    @app.route("/admin")
    def admin():
        # Admin dashboard
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
