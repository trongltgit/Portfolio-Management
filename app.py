# =========================================================
# PART F – PRODUCTION FOUNDATION (MUST BE AT TOP)
# AdminLTE UI | Persistent DB | Security | Render/VPS Ready
# =========================================================

# -----------------------------
# CORE IMPORTS
# -----------------------------
import os
import sqlite3
import secrets
from datetime import timedelta
from flask import Flask, g

# -----------------------------
# APP & ENV CONFIG
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "investment_app.db")

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", secrets.token_hex(32)),
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# -----------------------------
# DATABASE (SQLITE – PROD READY)
# -----------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, check_same_thread=False)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db:
        db.close()


def init_db():
    db = get_db()
    cur = db.cursor()

    # USERS
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """)

    # PORTFOLIO
    cur.execute("""
    CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        asset TEXT,
        amount REAL,
        FOREIGN KEY(username) REFERENCES users(username)
    )
    """)

    db.commit()


# -----------------------------
# ADMIN BOOTSTRAP (IMMUTABLE)
# -----------------------------

def bootstrap_admin():
    from werkzeug.security import generate_password_hash

    db = get_db()
    cur = db.cursor()

    cur.execute("SELECT username FROM users WHERE username='admin'")
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO users VALUES (?, ?, ?)",
            ("admin", generate_password_hash("Test@123456"), "admin")
        )
        db.commit()


# -----------------------------
# ADMINLTE BASE TEMPLATE
# -----------------------------

ADMINLTE_BASE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/admin-lte@3.2/dist/css/adminlte.min.css">
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css">
</head>
<body class="hold-transition sidebar-mini">
<div class="wrapper">

<nav class="main-header navbar navbar-expand navbar-white navbar-light">
  <ul class="navbar-nav ml-auto">
    <li class="nav-item">
      <a class="nav-link" href="/logout">Logout</a>
    </li>
  </ul>
</nav>

<aside class="main-sidebar sidebar-dark-primary elevation-4">
  <a href="/dashboard" class="brand-link">
    <span class="brand-text font-weight-light">Investment ML</span>
  </a>
  <div class="sidebar">
    <nav class="mt-2">
      <ul class="nav nav-pills nav-sidebar flex-column">
        <li class="nav-item"><a href="/dashboard" class="nav-link">Dashboard</a></li>
        <li class="nav-item"><a href="/invest" class="nav-link">Investment</a></li>
        <li class="nav-item"><a href="/portfolio" class="nav-link">Portfolio</a></li>
        <li class="nav-item"><a href="/advisor" class="nav-link">ML Advisor</a></li>
        <li class="nav-item"><a href="/chart" class="nav-link">Charts</a></li>
        <li class="nav-item"><a href="/market" class="nav-link">Market Info</a></li>
        <li class="nav-item"><a href="/export_pdf" class="nav-link">Export PDF</a></li>
      </ul>
    </nav>
  </div>
</aside>

<div class="content-wrapper p-4">
  {{ content|safe }}
</div>

</div>
</body>
</html>
"""

# -----------------------------
# SECURITY HARDENING
# -----------------------------

def harden_response(resp):
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.after_request
def apply_security(resp):
    return harden_response(resp)


# -----------------------------
# INIT HOOK
# -----------------------------

with app.app_context():
    init_db()
    bootstrap_admin()

# =========================================================
# END PART F – MUST STAY AT TOP
# =========================================================



# ============================
# BASIC IMPORTS
# ============================
import os
import re
import json
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functools import wraps

# ============================
# FLASK
# ============================
from flask import (
    Flask, request, redirect, url_for,
    render_template_string, session,
    abort, jsonify, send_from_directory
)

from werkzeug.security import generate_password_hash, check_password_hash

# ============================
# APP INIT
# ============================
app = Flask(__name__)
app.secret_key = "CHANGE_THIS_SECRET_KEY_IN_PRODUCTION"

# ============================
# CONSTANTS
# ============================
ADMIN_ID = "admin"
ADMIN_DEFAULT_PASSWORD = "Test@123456"
USER_DEFAULT_PASSWORD = "Test@1234"

PASSWORD_REGEX = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z\d]).{8,}$"
)

# ============================
# IN-MEMORY DATABASE (PHASE 1)
# ============================
USERS = {
    ADMIN_ID: {
        "password": generate_password_hash(ADMIN_DEFAULT_PASSWORD),
        "role": "admin"
    }
}

USER_PORTFOLIOS = {}      # {username: [ {...}, {...} ]}
USER_RISK_PROFILE = {}   # {username: Conservative / Balanced / Aggressive}

# ============================
# AUTH DECORATORS
# ============================
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("username") != ADMIN_ID:
            abort(403)
        return f(*args, **kwargs)
    return wrapper

# ============================
# PASSWORD VALIDATION
# ============================
def validate_password(pw: str) -> bool:
    return bool(PASSWORD_REGEX.match(pw))

# ============================
# BASE HTML TEMPLATE (BOOTSTRAP)
# ============================
BASE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <style>
        body { background:#f4f6f9; }
        .sidebar {
            width: 240px;
            position: fixed;
            top: 0; left: 0;
            height: 100vh;
            background: #212529;
            padding: 20px;
            color: white;
        }
        .sidebar a {
            color: #ddd;
            display: block;
            margin: 10px 0;
            text-decoration: none;
        }
        .sidebar a:hover {
            color: #fff;
        }
        .content {
            margin-left: 260px;
            padding: 30px;
        }
    </style>
</head>
<body>

{% if session.get("username") %}
<div class="sidebar">
    <h5>AI Investment</h5>
    <hr>
    <p><b>User:</b> {{ session.get("username") }}</p>
    <a href="/dashboard">Dashboard</a>
    <a href="/investment">Investment</a>
    <a href="/portfolio">Portfolio</a>
    <a href="/advisor">AI Advisor</a>
    {% if session.get("username") == 'admin' %}
        <a href="/admin">Admin Panel</a>
    {% endif %}
    <hr>
    <a href="/logout" class="text-danger">Logout</a>
</div>
{% endif %}

<div class="content">
    {{ content | safe }}
</div>

</body>
</html>
"""

# ============================
# AUTH ROUTES
# ============================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = USERS.get(username)
        if not user:
            return "Invalid credentials", 401

        if not check_password_hash(user["password"], password):
            return "Invalid credentials", 401

        session["username"] = username
        return redirect("/dashboard")

    content = """
    <h3>Login</h3>
    <form method="post">
        <div class="mb-3">
            <input class="form-control" name="username" placeholder="User ID">
        </div>
        <div class="mb-3">
            <input class="form-control" type="password" name="password" placeholder="Password">
        </div>
        <button class="btn btn-primary">Login</button>
    </form>
    """
    return render_template_string(BASE_HTML, title="Login", content=content)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ============================
# ADMIN – USER MANAGEMENT
# ============================
@app.route("/admin", methods=["GET", "POST"])
@login_required
@admin_required
def admin_panel():
    msg = ""

    if request.method == "POST":
        action = request.form.get("action")
        username = request.form.get("username")

        if action == "add":
            if username in USERS:
                msg = "User already exists"
            else:
                USERS[username] = {
                    "password": generate_password_hash(USER_DEFAULT_PASSWORD),
                    "role": "user"
                }
                msg = f"User {username} created (default password)"

        elif action == "delete":
            if username == ADMIN_ID:
                msg = "Cannot delete admin"
            else:
                USERS.pop(username, None)
                USER_PORTFOLIOS.pop(username, None)
                USER_RISK_PROFILE.pop(username, None)
                msg = f"User {username} deleted"

        elif action == "reset":
            if username in USERS:
                USERS[username]["password"] = generate_password_hash(USER_DEFAULT_PASSWORD)
                msg = f"Password reset for {username}"

    rows = ""
    for u, info in USERS.items():
        rows += f"<tr><td>{u}</td><td>{info['role']}</td></tr>"

    content = f"""
    <h3>Admin Panel</h3>
    <p class="text-info">{msg}</p>

    <form method="post" class="row g-2">
        <input class="form-control col" name="username" placeholder="Username">
        <select class="form-select col" name="action">
            <option value="add">Add</option>
            <option value="delete">Delete</option>
            <option value="reset">Reset Password</option>
        </select>
        <button class="btn btn-danger col">Execute</button>
    </form>

    <hr>
    <table class="table table-bordered">
        <tr><th>User</th><th>Role</th></tr>
        {rows}
    </table>
    """
    return render_template_string(BASE_HTML, title="Admin", content=content)

# ============================
# DASHBOARD PLACEHOLDER
# (SẼ ĐƯỢC MỞ RỘNG Ở PART D)
# ============================
@app.route("/dashboard")
@login_required
def dashboard():
    content = """
    <h2>Investment Dashboard</h2>
    <p>System initialized successfully.</p>
    <ul>
        <li>VCBF NAV Forecast</li>
        <li>HOSE / HNX Stocks</li>
        <li>AI Advisor</li>
        <li>Portfolio Management</li>
    </ul>
    """
    return render_template_string(BASE_HTML, title="Dashboard", content=content)


# =========================================================
# PART B – DATA ENGINE
# Crawl real price data: HOSE / HNX / VCBF NAV T+1
# =========================================================

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# =========================================================
# CONFIG
# =========================================================

VCBF_FUNDS = {
    "VCBF-TBF": "https://www.vcbf.com/vn/funds/vcbf-tbf",
    "VCBF-MGF": "https://www.vcbf.com/vn/funds/vcbf-mgf",
    "VCBF-BCF": "https://www.vcbf.com/vn/funds/vcbf-bcf",
    "VCBF-FIF": "https://www.vcbf.com/vn/funds/vcbf-fif",
}

HOSE_API = "https://api-finance.vietstock.vn/data/stock-price"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# =========================================================
# 1. VCBF NAV CRAWLER (T+1)
# =========================================================

def crawl_vcbf_nav(fund_code: str):
    """
    Crawl NAV T+1 from official VCBF website
    """
    url = VCBF_FUNDS.get(fund_code)
    if not url:
        return None

    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        nav_label = soup.find("span", string=lambda x: x and "NAV" in x)
        nav_value = nav_label.find_next("span").text.strip()

        nav = float(nav_value.replace(",", ""))
        nav_date = datetime.now().date() - timedelta(days=1)

        return {
            "fund": fund_code,
            "nav": nav,
            "nav_date": nav_date
        }
    except Exception as e:
        print(f"[VCBF NAV ERROR] {e}")
        return None


# =========================================================
# 2. STOCK PRICE CRAWLER (HOSE / HNX)
# =========================================================

def crawl_stock_price(symbol: str, exchange="HOSE", days=365):
    """
    Crawl historical prices from Vietstock public endpoint
    """
    end = datetime.now()
    start = end - timedelta(days=days)

    payload = {
        "symbol": symbol,
        "exchange": exchange,
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d")
    }

    try:
        r = requests.get(HOSE_API, params=payload, headers=HEADERS, timeout=10)
        data = r.json()

        df = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        print(f"[STOCK CRAWL ERROR] {symbol} - {e}")
        return None


# =========================================================
# 3. DATA NORMALIZATION
# =========================================================

def normalize_price_series(df: pd.DataFrame):
    """
    Normalize price series for ML input
    """
    df = df.copy()
    for col in ["open", "high", "low", "close"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


# =========================================================
# 4. VOLATILITY & RETURN METRICS
# =========================================================

def calculate_metrics(df: pd.DataFrame):
    """
    Calculate daily return & volatility
    """
    df = df.copy()
    df["return"] = df["close"].pct_change()
    volatility = df["return"].std() * np.sqrt(252)
    annual_return = df["return"].mean() * 252

    return {
        "annual_return": annual_return,
        "volatility": volatility
    }


# =========================================================
# 5. FUND RISK CLASSIFICATION
# =========================================================

def classify_risk(volatility):
    if volatility < 0.10:
        return "Low"
    elif volatility < 0.20:
        return "Medium"
    else:
        return "High"


# =========================================================
# 6. PORTFOLIO DATA LOADER
# =========================================================

def load_asset_data(asset_type, code):
    """
    Unified loader for ML engine
    """
    if asset_type == "VCBF":
        nav = crawl_vcbf_nav(code)
        if not nav:
            return None

        df = pd.DataFrame([{
            "date": nav["nav_date"],
            "close": nav["nav"]
        }])
        df.set_index("date", inplace=True)
        return df

    elif asset_type == "STOCK":
        return crawl_stock_price(code)

    return None


# =========================================================
# 7. DATA CACHE (IN-MEMORY)
# =========================================================

DATA_CACHE = {}

def get_cached_data(key, loader_func, ttl_minutes=30):
    now = datetime.now()
    cached = DATA_CACHE.get(key)

    if cached and (now - cached["time"]).seconds < ttl_minutes * 60:
        return cached["data"]

    data = loader_func()
    DATA_CACHE[key] = {
        "data": data,
        "time": now
    }
    return data


# =========================================================
# END PART B
# =========================================================


# =========================================================
# PART C – ML ENGINE + PORTFOLIO + BACKTEST
# Real ML: Prediction + Risk-aware Allocation + Backtesting
# =========================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# =========================================================
# 1. FEATURE ENGINEERING
# =========================================================

def build_features(df: pd.DataFrame, window=14):
    """
    Build ML features from price series
    """
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["ma"] = df["close"].rolling(window).mean()
    df["std"] = df["close"].rolling(window).std()
    df["momentum"] = df["close"] - df["close"].shift(window)

    df.dropna(inplace=True)
    return df


# =========================================================
# 2. ML PRICE PREDICTION MODEL
# =========================================================

def train_price_model(df: pd.DataFrame):
    """
    Train RandomForest for next-period price prediction
    """
    df_feat = build_features(df)

    X = df_feat[["return", "ma", "std", "momentum"]]
    y = df_feat["close"].shift(-1)

    X, y = X[:-1], y[:-1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    model.fit(X_scaled, y)

    return model, scaler, df_feat


def predict_next_price(model, scaler, df_feat):
    """
    Predict next price
    """
    last_row = df_feat.iloc[-1][["return", "ma", "std", "momentum"]].values.reshape(1, -1)
    last_scaled = scaler.transform(last_row)
    return float(model.predict(last_scaled)[0])


# =========================================================
# 3. BUY / SELL SIGNAL ENGINE
# =========================================================

def generate_signal(current_price, predicted_price, threshold=0.02):
    """
    Generate trading signal
    """
    delta = (predicted_price - current_price) / current_price

    if delta > threshold:
        return "BUY"
    elif delta < -threshold:
        return "SELL"
    else:
        return "HOLD"


# =========================================================
# 4. RISK PROFILE CONFIG
# =========================================================

RISK_PROFILE = {
    "Conservative": {"max_vol": 0.10, "equity_ratio": 0.30},
    "Balanced": {"max_vol": 0.18, "equity_ratio": 0.60},
    "Aggressive": {"max_vol": 0.30, "equity_ratio": 0.80},
}


# =========================================================
# 5. PORTFOLIO OPTIMIZATION (MEAN–VARIANCE)
# =========================================================

def portfolio_performance(weights, returns, cov):
    port_return = np.sum(returns * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return port_return, port_vol


def optimize_portfolio(returns, cov, risk_profile="Balanced"):
    """
    Optimize portfolio under risk constraint
    """
    n = len(returns)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: RISK_PROFILE[risk_profile]["max_vol"]
         - np.sqrt(np.dot(w.T, np.dot(cov, w)))}
    ]

    def objective(w):
        ret, vol = portfolio_performance(w, returns, cov)
        return -ret / vol

    w0 = np.array([1 / n] * n)

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x if result.success else w0


# =========================================================
# 6. BACKTEST ENGINE
# =========================================================

def backtest_strategy(df: pd.DataFrame, initial_cash=100_000_000):
    """
    Simple ML-based backtest
    """
    cash = initial_cash
    shares = 0
    history = []

    model, scaler, df_feat = train_price_model(df)

    for i in range(len(df_feat) - 1):
        row = df_feat.iloc[: i + 1]
        current_price = row.iloc[-1]["close"]
        predicted = predict_next_price(model, scaler, row)
        signal = generate_signal(current_price, predicted)

        if signal == "BUY" and cash > current_price:
            shares = cash // current_price
            cash -= shares * current_price

        elif signal == "SELL" and shares > 0:
            cash += shares * current_price
            shares = 0

        portfolio_value = cash + shares * current_price
        history.append(portfolio_value)

    return {
        "initial": initial_cash,
        "final": history[-1] if history else initial_cash,
        "return_pct": (history[-1] / initial_cash - 1) * 100 if history else 0,
        "history": history
    }


# =========================================================
# 7. ASSET RANKING ENGINE
# =========================================================

def rank_assets(asset_data: dict):
    """
    Rank funds/stocks by risk-adjusted return
    """
    ranking = []

    for code, df in asset_data.items():
        metrics = calculate_metrics(df)
        score = metrics["annual_return"] / metrics["volatility"]
        ranking.append({
            "code": code,
            "return": metrics["annual_return"],
            "volatility": metrics["volatility"],
            "score": score
        })

    ranking.sort(key=lambda x: x["score"], reverse=True)
    return ranking


# =========================================================
# 8. USER INVESTMENT ADVISOR
# =========================================================

def investment_advisor(amount, assets: dict, risk_profile="Balanced"):
    """
    Full advisory pipeline
    """
    returns = []
    cov_matrix = []

    price_frames = []

    for df in assets.values():
        price_frames.append(df["close"].pct_change().dropna())

    combined = pd.concat(price_frames, axis=1).dropna()
    returns = combined.mean().values * 252
    cov_matrix = combined.cov().values * 252

    weights = optimize_portfolio(returns, cov_matrix, risk_profile)

    allocation = {}
    for (code, _), w in zip(assets.items(), weights):
        allocation[code] = {
            "weight": float(w),
            "amount": float(amount * w)
        }

    return allocation


# =========================================================
# END PART C
# =========================================================

# =========================================================
# PART D – CHART + TECHNICAL INDICATORS + SIGNALS
# =========================================================

import matplotlib
matplotlib.use("Agg")  # for server / Render
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# =========================================================
# 1. TECHNICAL INDICATORS
# =========================================================

def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["close"].ewm(span=fast).mean()
    ema_slow = df["close"].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    hist = macd - macd_signal
    return macd, macd_signal, hist


def compute_stochastic(df, k_period=14, d_period=3):
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()

    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    d = k.rolling(d_period).mean()
    return k, d


# =========================================================
# 2. TIMEFRAME RESAMPLER
# =========================================================

def resample_price(df, timeframe="daily"):
    if timeframe == "weekly":
        return df.resample("W").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
    elif timeframe == "monthly":
        return df.resample("M").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
    elif timeframe == "yearly":
        return df.resample("Y").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
    return df


# =========================================================
# 3. BUY / SELL MARKERS
# =========================================================

def detect_signals(df):
    rsi = compute_rsi(df)
    macd, macd_signal, _ = compute_macd(df)

    buy = (rsi < 30) & (macd > macd_signal)
    sell = (rsi > 70) & (macd < macd_signal)

    return buy, sell


# =========================================================
# 4. PRICE CHART GENERATOR (BASE64)
# =========================================================

def generate_price_chart(df, symbol, timeframe="daily"):
    df = resample_price(df, timeframe)
    df = df.dropna()

    rsi = compute_rsi(df)
    macd, macd_signal, hist = compute_macd(df)
    buy, sell = detect_signals(df)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # ---- PRICE ----
    axes[0].plot(df.index, df["close"], label="Close Price")
    axes[0].scatter(df.index[buy], df["close"][buy], marker="^", color="green", label="BUY")
    axes[0].scatter(df.index[sell], df["close"][sell], marker="v", color="red", label="SELL")
    axes[0].set_title(f"{symbol} Price Chart ({timeframe})")
    axes[0].legend()

    # ---- RSI ----
    axes[1].plot(df.index, rsi, label="RSI")
    axes[1].axhline(30)
    axes[1].axhline(70)
    axes[1].set_ylabel("RSI")
    axes[1].legend()

    # ---- MACD ----
    axes[2].plot(df.index, macd, label="MACD")
    axes[2].plot(df.index, macd_signal, label="Signal")
    axes[2].bar(df.index, hist, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()

    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return img_base64


# =========================================================
# 5. ML + TECHNICAL CONSENSUS
# =========================================================

def combined_signal(df):
    model, scaler, df_feat = train_price_model(df)
    predicted = predict_next_price(model, scaler, df_feat)
    current = df_feat.iloc[-1]["close"]

    ml_signal = generate_signal(current, predicted)

    rsi = compute_rsi(df)
    tech_signal = "HOLD"

    if rsi.iloc[-1] < 30:
        tech_signal = "BUY"
    elif rsi.iloc[-1] > 70:
        tech_signal = "SELL"

    if ml_signal == tech_signal:
        return ml_signal
    return "HOLD"


# =========================================================
# END PART D
# =========================================================

# =========================================================
# PART E – DASHBOARD UI + ADMIN / USER + PDF EXPORT
# =========================================================

from flask import Flask, request, session, redirect, url_for, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

# =========================================================
# 1. FLASK APP INIT
# =========================================================

app.secret_key = "SUPER_SECRET_KEY_CHANGE_ME"

# =========================================================
# 2. USER STORE (IN-MEMORY – PROD: REPLACE DB)
# =========================================================

USERS = {
    "admin": {
        "password": generate_password_hash("Test@123456"),
        "role": "admin"
    }
}

USER_PORTFOLIO = {}  # {username: {asset: amount}}

# =========================================================
# 3. AUTH DECORATORS
# =========================================================

def login_required(role=None):
    def wrapper(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            if "user" not in session:
                return redirect("/")
            if role and session.get("role") != role:
                return "Unauthorized", 403
            return fn(*args, **kwargs)
        return decorated
    return wrapper


# =========================================================
# 4. PASSWORD POLICY
# =========================================================

def valid_password(pw):
    if len(pw) < 8:
        return False
    return any(c.isupper() for c in pw) and any(c.islower() for c in pw) \
        and any(c.isdigit() for c in pw) and any(not c.isalnum() for c in pw)


# =========================================================
# 5. LOGIN / LOGOUT
# =========================================================

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        user = USERS.get(u)
        if user and check_password_hash(user["password"], p):
            session["user"] = u
            session["role"] = user["role"]
            return redirect("/dashboard")

    return """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
    <div class="container mt-5">
      <h3>Login</h3>
      <form method="post">
        <input class="form-control mb-2" name="username" placeholder="Username">
        <input class="form-control mb-2" name="password" type="password" placeholder="Password">
        <button class="btn btn-primary">Login</button>
      </form>
    </div>
    """


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# =========================================================
# 6. DASHBOARD
# =========================================================

@app.route("/dashboard")
@login_required()
def dashboard():
    if session["role"] == "admin":
        return redirect("/admin")

    return f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
    <div class="container mt-4">
      <h4>User Dashboard – {session["user"]}</h4>
      <ul>
        <li><a href="/invest">Investment</a></li>
        <li><a href="/portfolio">Portfolio</a></li>
        <li><a href="/advisor">ML Advisor</a></li>
        <li><a href="/chart">Charts</a></li>
        <li><a href="/market">Market Info</a></li>
        <li><a href="/export_pdf">Export PDF</a></li>
        <li><a href="/logout">Logout</a></li>
      </ul>
    </div>
    """


# =========================================================
# 7. ADMIN PANEL
# =========================================================

@app.route("/admin", methods=["GET", "POST"])
@login_required(role="admin")
def admin_panel():
    msg = ""
    if request.method == "POST":
        action = request.form["action"]
        username = request.form["username"]

        if action == "add":
            pw = "Test@1234"
            USERS[username] = {
                "password": generate_password_hash(pw),
                "role": "user"
            }
            msg = f"User {username} added"

        elif action == "delete":
            USERS.pop(username, None)
            USER_PORTFOLIO.pop(username, None)
            msg = f"User {username} deleted"

        elif action == "reset":
            USERS[username]["password"] = generate_password_hash("Test@1234")
            msg = f"Password reset"

    user_list = "<br>".join(USERS.keys())

    return f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
    <div class="container mt-4">
      <h3>Admin Panel</h3>
      <form method="post">
        <input name="username" placeholder="Username" class="form-control mb-2">
        <select name="action" class="form-control mb-2">
          <option value="add">Add User</option>
          <option value="delete">Delete User</option>
          <option value="reset">Reset Password</option>
        </select>
        <button class="btn btn-danger">Execute</button>
      </form>
      <p>{msg}</p>
      <h5>Users</h5>
      {user_list}
      <br><a href="/logout">Logout</a>
    </div>
    """


# =========================================================
# 8. PORTFOLIO
# =========================================================

@app.route("/portfolio")
@login_required()
def portfolio():
    pf = USER_PORTFOLIO.get(session["user"], {})
    items = "<br>".join([f"{k}: {v:,.0f}" for k, v in pf.items()]) or "Empty"
    return f"<h4>Portfolio</h4>{items}<br><a href='/dashboard'>Back</a>"


# =========================================================
# 9. PDF EXPORT
# =========================================================

@app.route("/export_pdf")
@login_required()
def export_pdf():
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.drawString(50, 800, f"Investment Report – {session['user']}")

    y = 760
    pf = USER_PORTFOLIO.get(session["user"], {})
    for k, v in pf.items():
        pdf.drawString(50, y, f"{k}: {v:,.0f}")
        y -= 20

    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True,
                     download_name="investment_report.pdf",
                     mimetype="application/pdf")


# =========================================================
# 10. DEPLOY ENTRY
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
