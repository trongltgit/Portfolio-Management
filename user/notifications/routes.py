# user/notifications/routes.py
from flask import Blueprint, render_template, session
from db import get_db

notifications_bp = Blueprint("notifications", __name__, url_prefix="/user/notifications")

@notifications_bp.route("/")
def index():
    user_id = session.get("user_id")
    if not user_id:
        return "Unauthorized", 401

    conn = get_db()
    alerts = conn.execute(
        "SELECT * FROM notifications WHERE user_id=? ORDER BY created_at DESC LIMIT 50",
        (user_id,)
    ).fetchall()
    conn.close()

    return render_template("user/notifications/index.html", alerts=alerts)
