# user/consultancy/routes.py
from flask import Blueprint, render_template, session
from .services import generate_recommendations

consultancy_bp = Blueprint("consultancy", __name__, url_prefix="/user/consultancy")

@consultancy_bp.route("/", methods=["GET"])
def dashboard():
    user_id = session.get("user_id")
    if not user_id:
        return "Unauthorized", 401

    recommendations = generate_recommendations(user_id)
    return render_template("user/consultancy/index.html", recommendations=recommendations)
