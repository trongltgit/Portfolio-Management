from flask import render_template, request
from . import market_bp
from user.routes import login_required
from .services import get_vcbf_nav, get_stock_price, get_vcbs_priceboard

@market_bp.route("/", methods=["GET", "POST"])
@login_required
def market_dashboard():
    vcbf_result = None
    stock_result = None
    priceboard = []

    if request.method == "POST":
        if request.form.get("type") == "vcbf":
            vcbf_result = get_vcbf_nav(request.form["fund_code"])

        if request.form.get("type") == "stock":
            stock_result = get_stock_price(
                request.form["symbol"],
                request.form.get("exchange", "HOSE")
            )

    priceboard = get_vcbs_priceboard()

    return render_template(
        "user/market.html",
        vcbf=vcbf_result,
        stock=stock_result,
        priceboard=priceboard
    )
