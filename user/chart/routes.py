# user/chart/routes.py
from flask import Blueprint, render_template, request, session, flash
from .services import get_historical_data, calculate_macd, calculate_rsi

chart_bp = Blueprint("chart", __name__, url_prefix="/user/chart")

@chart_bp.route("/", methods=["GET", "POST"])
def dashboard():
    data = None
    symbol = request.form.get("symbol", "AAPL")
    period = request.form.get("period", "6mo")

    if request.method == "POST":
        df = get_historical_data(symbol, period=period)
        df = calculate_macd(df)
        df = calculate_rsi(df)
        data = df.to_dict(orient="records")

    return render_template("user/chart/index.html", data=data, symbol=symbol, period=period)
