# user/consultancy/services.py
import pandas as pd
from user.market.services import get_vcbf_priceboard  # ví dụ service đã có
from user.portfolio.services import get_user_portfolio

def generate_recommendations(user_id):
    """
    Trả về danh sách gợi ý dựa trên portfolio & market data
    """
    portfolio = get_user_portfolio(user_id)
    priceboard = get_vcbf_priceboard()  # Giả định trả về dict {symbol: price}

    recommendations = []
    for asset in portfolio:
        symbol = asset['asset_code']
        current_price = priceboard.get(symbol, None)
        if current_price is None:
            continue

        # Rule-based example:
        # Nếu giá hiện tại < avg_price -> BUY, > avg_price -> SELL, bằng -> HOLD
        if current_price < asset['avg_price']:
            action = "BUY"
        elif current_price > asset['avg_price']:
            action = "SELL"
        else:
            action = "HOLD"

        recommendations.append({
            "symbol": symbol,
            "quantity": asset['quantity'],
            "avg_price": asset['avg_price'],
            "current_price": current_price,
            "action": action
        })

    return recommendations
