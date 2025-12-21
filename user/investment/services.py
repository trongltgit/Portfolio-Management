# user/investment/services.py
from db import get_db
from datetime import datetime
import requests
from bs4 import BeautifulSoup

VCBF_FUNDS = {
    "VCBF-TBF": "https://www.vcbf.com/vn/funds/vcbf-tbf",
    "VCBF-MGF": "https://www.vcbf.com/vn/funds/vcbf-mgf",
    "VCBF-BCF": "https://www.vcbf.com/vn/funds/vcbf-bcf",
}

VCBS_PRICEBOARD = "https://priceboard.vcbs.com.vn/Priceboard"

HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_nav_ccq(code):
    url = VCBF_FUNDS.get(code)
    if not url:
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        label = soup.find("span", string=lambda x: x and "NAV" in x)
        value = label.find_next("span").text.strip()
        nav = float(value.replace(",", ""))
        return nav
    except:
        return None

def get_stock_price(symbol):
    # Lấy giá từ VCBS Priceboard
    try:
        r = requests.get(VCBS_PRICEBOARD, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        # Giả sử có table data-priceboard với symbol
        td = soup.find("td", string=symbol)
        if td:
            price = td.find_next_sibling("td").text.strip()
            return float(price.replace(",", ""))
        return None
    except:
        return None

def add_order(user_id, asset_code, asset_type, side, amount):
    """
    Tính quantity = amount / price
    Lưu vào orders
    """
    price = None
    if asset_type == "VCBF":
        price = get_nav_ccq(asset_code)
    elif asset_type == "STOCK":
        price = get_stock_price(asset_code)

    if not price:
        return False, "Cannot fetch price"

    quantity = round(amount / price, 4)

    conn = get_db()
    conn.execute(
        """
        INSERT INTO orders (user_id, asset_code, asset_type, side, price, amount, quantity)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, asset_code, asset_type, side, price, amount, quantity)
    )
    conn.commit()
    conn.close()
    return True, "Order added"

def get_portfolio(user_id):
    conn = get_db()
    rows = conn.execute(
        """
        SELECT asset_code, asset_type, SUM(
            CASE WHEN side='BUY' THEN quantity
                 WHEN side='SELL' THEN -quantity
                 ELSE 0 END
        ) as quantity,
        AVG(price) as avg_price
        FROM orders
        WHERE user_id=?
        GROUP BY asset_code, asset_type
        """,
        (user_id,)
    ).fetchall()
    conn.close()
    return rows
