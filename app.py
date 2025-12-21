# app.py
from flask import Flask, redirect, url_for, render_template, session
from config import Config
from db import init_db, seed_admin

# -----------------------------
# IMPORT BLUEPRINTS
# -----------------------------
from auth import auth_bp       # /login, /logout, /change-password
from user import user_bp       # /user/*
from market import market_bp   # /market/*

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
    app.register_blueprint(user_bp)
    app.register_blueprint(market_bp)

    # -----------------------------
    # ROOT ROUTE
    # -----------------------------
    @app.route("/")
    def index():
        # Nếu chưa login → login page
        if "user_id" not in session:
            return redirect(url_for("auth.login"))

        # Redirect theo role
        role = session.get("role")
        if role == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("user_dashboard"))

    # -----------------------------
    # USER DASHBOARD
    # -----------------------------
    @app.route("/dashboard")
    def user_dashboard():
        if "user_id" not in session:
            return redirect(url_for("auth.login"))
        if session.get("role") != "user":
            return "Access denied", 403
        return render_template("user/dashboard.html")

    # -----------------------------
    # ADMIN DASHBOARD
    # -----------------------------
    @app.route("/admin")
    def admin_dashboard():
        if "user_id" not in session:
            return redirect(url_for("auth.login"))
        if session.get("role") != "admin":
            return "Access denied", 403
        return render_template("admin/dashboard.html")

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
