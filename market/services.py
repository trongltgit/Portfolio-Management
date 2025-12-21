import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from flask import render_template
from . import market_bp
from .services import (
    get_vcbf_funds,
    get_hose_index,
    get_vcbs_priceboard
)


@market_bp.route("/")
def market_dashboard():
    return render_template(
        "market/index.html",
        vcbf=get_vcbf_funds(),
        hose=get_hose_index(),
        board=get_vcbs_priceboard()
    )


HEADERS = {"User-Agent": "Mozilla/5.0"}

# =========================
# VCBF NAV
# =========================
VCBF_FUNDS = {
    "VCBFTBF": "https://www.vcbf.com/vn/funds/vcbf-tbf",
    "VCBFMGF": "https://www.vcbf.com/vn/funds/vcbf-mgf",
    "VCBFBCF": "https://www.vcbf.com/vn/funds/vcbf-bcf",
    "VCBFFIF": "https://www.vcbf.com/vn/funds/vcbf-fif",
    "VCBFAIF": "https://www.vcbf.com/vn/funds/vcbf-aif",
}

def get_vcbf_nav(fund_code: str):
    url = VCBF_FUNDS.get(fund_code.upper())
    if not url:
        return None

    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        label = soup.find("span", string=lambda x: x and "NAV" in x)
        value = label.find_next("span").text.strip()

        return {
            "fund": fund_code,
            "nav": float(value.replace(",", "")),
            "date": (datetime.now() - timedelta(days=1)).date()
        }
    except Exception:
        return None


# =========================
# STOCK PRICE – HOSE / HNX
# =========================
VIETSTOCK_API = "https://api-finance.vietstock.vn/data/stock-price"

def get_stock_price(symbol: str, exchange="HOSE"):
    params = {
        "symbol": symbol.upper(),
        "exchange": exchange,
        "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "to": datetime.now().strftime("%Y-%m-%d"),
    }

    try:
        r = requests.get(VIETSTOCK_API, params=params, headers=HEADERS, timeout=10)
        data = r.json().get("data", [])
        if not data:
            return None

        df = pd.DataFrame(data)
        last = df.iloc[-1]

        return {
            "symbol": symbol.upper(),
            "exchange": exchange,
            "close": last["close"],
            "date": last["date"]
        }
    except Exception:
        return None


# =========================
# VCBS PRICEBOARD SNAPSHOT
# =========================
VCBS_PRICEBOARD = "https://priceboard.vcbs.com.vn/Priceboard"

def get_vcbs_priceboard(limit=20):
    try:
        res = requests.get(VCBS_PRICEBOARD, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        table = soup.find("table")
        rows = table.find_all("tr")[1:limit+1]

        data = []
        for r in rows:
            cols = [c.text.strip() for c in r.find_all("td")]
            if len(cols) >= 5:
                data.append({
                    "symbol": cols[0],
                    "price": cols[4]
                })
        return data
    except Exception:
        return []
