# app.py
from flask import Flask, redirect, url_for, render_template, session
from config import Config
from db import init_db, seed_admin

# -----------------------------
# IMPORT BLUEPRINTS
# -----------------------------
from auth import auth_bp       # phải có route /login, /logout
from user import user_bp       # các route user
from market import market_bp   # các route market

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
    # ROOT ROUTE
    # -----------------------------
    @app.route("/")
    def index():
        # Redirect root URL vào login page
        return redirect(url_for("auth.login"))

    # -----------------------------
    # USER DASHBOARD
    # -----------------------------
    @app.route("/user/templates")
    def dashboard():
        if "user_id" not in session:
            return redirect(url_for("auth.login"))

        # Chỉ cho phép user role 'user' truy cập
        if session.get("role") != "user":
            return "Access denied", 403

        return render_template("dashboard.html")  # template user dashboard

    # -----------------------------
    # ADMIN DASHBOARD
    # -----------------------------
    @app.route("/admin/templates")
    def admin_dashboard():
        if "user_id" not in session:
            return redirect(url_for("auth.login"))

        # Chỉ cho phép user role 'admin' truy cập
        if session.get("role") != "admin":
            return "Access denied", 403

        return render_template("dashboard.html")  # template admin dashboard

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
