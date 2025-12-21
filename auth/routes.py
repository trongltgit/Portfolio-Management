# auth/routes.py
from flask import (
    Blueprint, request, render_template_string,
    redirect, url_for, session, flash
)
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
from db import get_db

auth_bp = Blueprint("auth", __name__)

# -----------------------------
# DECORATORS
# -----------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return wrapper


def role_required(role):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if session.get("role") != role:
                return "Forbidden", 403
            return f(*args, **kwargs)
        return wrapper
    return decorator

# -----------------------------
# LOGIN
# -----------------------------
LOGIN_HTML = """
<h3>Login</h3>
<form method="post">
  <input name="username" placeholder="Username" required><br>
  <input type="password" name="password" placeholder="Password" required><br>
  <button>Login</button>
</form>
{% for m in get_flashed_messages() %}
  <p style="color:red">{{ m }}</p>
{% endfor %}
"""

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=?",
            (username,)
        ).fetchone()
        conn.close()

        if not user:
            flash("Invalid credentials")
        elif user["locked"]:
            flash("Account is locked")
        elif check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["role"] = user["role"]

            # ROLE REDIRECT
            if user["role"] == "admin":
                return redirect("/admin")
            return redirect("/dashboard")
        else:
            flash("Invalid credentials")

    return render_template_string(LOGIN_HTML)

# -----------------------------
# LOGOUT
# -----------------------------
@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))

# -----------------------------
# CHANGE PASSWORD (USER)
# -----------------------------
CHANGE_PASS_HTML = """
<h3>Change Password</h3>
<form method="post">
  <input type="password" name="old" placeholder="Old password" required><br>
  <input type="password" name="new" placeholder="New password" required><br>
  <button>Change</button>
</form>
{% for m in get_flashed_messages() %}
  <p style="color:red">{{ m }}</p>
{% endfor %}
"""

@auth_bp.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        old = request.form["old"]
        new = request.form["new"]

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE id=?",
            (session["user_id"],)
        ).fetchone()

        if not check_password_hash(user["password_hash"], old):
            flash("Old password incorrect")
        else:
            conn.execute(
                "UPDATE users SET password_hash=? WHERE id=?",
                (generate_password_hash(new), session["user_id"])
            )
            conn.commit()
            flash("Password changed successfully")

        conn.close()

    return render_template_string(CHANGE_PASS_HTML)
