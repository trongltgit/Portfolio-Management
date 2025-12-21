# user/investment/routes.py
from flask import Blueprint, render_template, request, session, flash, redirect, url_for
from .services import add_order, get_portfolio

investment_bp = Blueprint("investment", __name__, url_prefix="/user/investment")

@investment_bp.route("/")
def dashboard():
    user_id = session.get("user_id")
    portfolio = get_portfolio(user_id)
    return render_template("user/investment/index.html", portfolio=portfolio)

@investment_bp.route("/add", methods=["GET", "POST"])
def add():
    if request.method == "POST":
        user_id = session.get("user_id")
        asset_code = request.form["asset_code"]
        asset_type = request.form["asset_type"]
        side = request.form["side"]
        amount = float(request.form["amount"])

        success, msg = add_order(user_id, asset_code, asset_type, side, amount)
        flash(msg, "success" if success else "danger")
        return redirect(url_for("investment.dashboard"))

    return render_template("user/investment/add_order.html")
