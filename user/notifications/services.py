# user/notifications/services.py
from datetime import datetime
from user.portfolio.services import get_user_portfolio
from user.market.services import get_vcbf_priceboard
from db import get_db

def check_price_alerts(user_id, threshold_pct=2.0):
    """
    Kiểm tra portfolio của user, tạo alert nếu giá thay đổi > threshold_pct %
    Lưu vào DB bảng notifications
    """
    portfolio = get_user_portfolio(user_id)
    priceboard = get_vcbf_priceboard()
    alerts = []

    for asset in portfolio:
        symbol = asset['asset_code']
        current_price = priceboard.get(symbol, None)
        if not current_price:
            continue
        change_pct = ((current_price - asset['avg_price']) / asset['avg_price']) * 100

        if abs(change_pct) >= threshold_pct:
            action = "ALERT"
            message = f"{symbol} changed {change_pct:.2f}% from avg price {asset['avg_price']:.0f} → current {current_price:.0f}"

            # Lưu vào DB
            conn = get_db()
            conn.execute(
                "INSERT INTO notifications (user_id, symbol, change_pct, message, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, symbol, change_pct, message, datetime.now())
            )
            conn.commit()
            conn.close()

            alerts.append({"symbol": symbol, "change_pct": change_pct, "message": message})

    return alerts
