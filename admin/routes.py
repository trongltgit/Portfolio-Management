from flask import (
    render_template, request, redirect,
    url_for, flash, session, abort
)
from werkzeug.security import generate_password_hash
from . import admin_bp
from db import get_db


# -------------------------
# ADMIN GUARD
# -------------------------
def admin_required():
    if session.get("role") != "admin":
        abort(403)


def log_action(admin_id, action, target):
    conn = get_db()
    conn.execute(
        "INSERT INTO admin_logs (admin_id, action, target_user) VALUES (?, ?, ?)",
        (admin_id, action, target)
    )
    conn.commit()
    conn.close()


# -------------------------
# ADMIN DASHBOARD
# -------------------------
@admin_bp.route("/", methods=["GET", "POST"])
def dashboard():
    admin_required()
    conn = get_db()

    action = request.form.get("action")
    admin_id = session["user_id"]

    # ADD USER
    if action == "add":
        username = request.form["username"].strip()
        password = request.form["password"]
        role = request.form["role"]

        try:
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, generate_password_hash(password), role)
            )
            conn.commit()
            log_action(admin_id, "ADD_USER", username)
            flash("User added successfully", "success")
        except:
            flash("Username already exists", "danger")

    # RESET PASSWORD
    if action == "reset":
        uid = request.form["user_id"]
        conn.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (generate_password_hash("Test@123"), uid)
        )
        conn.commit()
        log_action(admin_id, "RESET_PASSWORD", uid)
        flash("Password reset to Test@123", "warning")

    # LOCK / UNLOCK
    if action == "lock":
        uid = request.form["user_id"]
        conn.execute("UPDATE users SET locked=1 WHERE id=?", (uid,))
        conn.commit()
        log_action(admin_id, "LOCK_USER", uid)

    if action == "unlock":
        uid = request.form["user_id"]
        conn.execute("UPDATE users SET locked=0 WHERE id=?", (uid,))
        conn.commit()
        log_action(admin_id, "UNLOCK_USER", uid)

    # DELETE STEP 1
    if action == "confirm_delete":
        uid = request.form["user_id"]
        user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        conn.close()
        return render_template("confirm_delete.html", user=user)

    # DELETE FINAL
    if action == "delete":
        uid = request.form["user_id"]
        conn.execute("DELETE FROM users WHERE id=? AND role!='admin'", (uid,))
        conn.commit()
        log_action(admin_id, "DELETE_USER", uid)
        flash("User deleted", "danger")

    users = conn.execute("SELECT * FROM users").fetchall()
    logs = conn.execute(
        "SELECT * FROM admin_logs ORDER BY timestamp DESC LIMIT 50"
    ).fetchall()
    conn.close()

    return render_template(
        "admin_dashboard.html",
        users=users,
        logs=logs
    )
