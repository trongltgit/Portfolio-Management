from flask import (
    render_template, session,
    redirect, url_for, request, flash
)
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
from db import get_db
from . import user_bp


# =========================
# LOGIN REQUIRED DECORATOR
# =========================
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return wrapper


# =========================
# USER DASHBOARD
# =========================
@user_bp.route("/dashboard")
@login_required
def dashboard():
    if session.get("role") != "user":
        return redirect("/admin")  # admin lạc route

    return render_template("user/dashboard.html")


# =========================
# CHANGE PASSWORD
# =========================
@user_bp.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        old = request.form["old_password"]
        new = request.form["new_password"]
        confirm = request.form["confirm_password"]

        if new != confirm:
            flash("Password confirmation does not match", "danger")
            return redirect(url_for("user.change_password"))

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE id=?",
            (session["user_id"],)
        ).fetchone()

        if not check_password_hash(user["password_hash"], old):
            flash("Old password incorrect", "danger")
            conn.close()
            return redirect(url_for("user.change_password"))

        conn.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (generate_password_hash(new), session["user_id"])
        )
        conn.commit()
        conn.close()

        flash("Password changed successfully", "success")
        return redirect(url_for("user.dashboard"))

    return render_template("user/change_password.html")
