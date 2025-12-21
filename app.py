# =====================================================
# PART 0 – CORE APP / CONFIG / DB (FIXED)
# =====================================================
import os
import sqlite3
from functools import wraps
from flask import (
    Flask, request, redirect, url_for,
    render_template_string, session, flash, abort
)
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "app.db")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "CHANGE_ME_IN_PROD")

# ---------------- DB ----------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    # USERS
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT,
            locked INTEGER DEFAULT 0
        )
    """)

    # ADMIN LOGS
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id INTEGER,
            action TEXT,
            target_user TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def seed_admin():
    conn = get_db()
    cur = conn.cursor()

    admin = cur.execute(
        "SELECT * FROM users WHERE username='admin'"
    ).fetchone()

    if not admin:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, locked) VALUES (?, ?, ?, 0)",
            (
                "admin",
                generate_password_hash("Test@123"),
                "admin"
            )
        )
        conn.commit()

    conn.close()


# INIT DB ON BOOT
init_db()
seed_admin()


# =====================================================
# PART 1 – AUTH / ADMIN (FIXED)
# =====================================================
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def log_admin_action(admin_id, action, target_user):
    conn = get_db()
    conn.execute(
        "INSERT INTO admin_logs (admin_id, action, target_user) VALUES (?, ?, ?)",
        (admin_id, action, target_user)
    )
    conn.commit()
    conn.close()


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=?",
            (username,)
        ).fetchone()
        conn.close()

        if user:
            if user["locked"]:
                flash("Account is locked. Contact admin.", "danger")
            elif check_password_hash(user["password_hash"], password):
                session["user_id"] = user["id"]
                session["role"] = user["role"]
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid credentials", "danger")
        else:
            flash("Invalid credentials", "danger")

    return render_template_string(LOGIN_HTML)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/admin", methods=["GET", "POST"])
@login_required
def admin():
    if session.get("role") != "admin":
        abort(403)

    conn = get_db()
    admin_id = session["user_id"]

    action = request.form.get("action")

    # ADD USER
    if request.method == "POST" and action == "add":
        username = request.form["username"].strip()
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, generate_password_hash("Test@123"), "user")
            )
            conn.commit()
            log_admin_action(admin_id, "ADD_USER", username)
            flash("User added (default password: Test@123)", "success")
        except sqlite3.IntegrityError:
            flash("Username already exists", "danger")

    # RESET PASSWORD
    if request.method == "POST" and action == "reset":
        uid = request.form["user_id"]
        conn.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (generate_password_hash("Test@123"), uid)
        )
        conn.commit()
        log_admin_action(admin_id, "RESET_PASSWORD", uid)
        flash("Password reset to Test@123", "warning")

    # CHANGE PASSWORD
    if request.method == "POST" and action == "change_pass":
        uid = request.form["user_id"]
        new_pass = request.form["new_password"]
        conn.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (generate_password_hash(new_pass), uid)
        )
        conn.commit()
        log_admin_action(admin_id, "CHANGE_PASSWORD", uid)
        flash("Password changed", "success")

    # LOCK / UNLOCK
    if request.method == "POST" and action == "lock":
        uid = request.form["user_id"]
        conn.execute("UPDATE users SET locked=1 WHERE id=?", (uid,))
        conn.commit()
        log_admin_action(admin_id, "LOCK_USER", uid)
        flash("User locked", "danger")

    if request.method == "POST" and action == "unlock":
        uid = request.form["user_id"]
        conn.execute("UPDATE users SET locked=0 WHERE id=?", (uid,))
        conn.commit()
        log_admin_action(admin_id, "UNLOCK_USER", uid)
        flash("User unlocked", "success")

    # DELETE STEP 1 – CONFIRM
    if request.method == "POST" and action == "confirm_delete":
        uid = request.form["user_id"]
        user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        conn.close()
        return render_template_string(DELETE_CONFIRM_HTML, user=user)

    # DELETE FINAL
    if request.method == "POST" and action == "delete":
        uid = request.form["user_id"]
        conn.execute("DELETE FROM users WHERE id=? AND role!='admin'", (uid,))
        conn.commit()
        log_admin_action(admin_id, "DELETE_USER", uid)
        flash("User deleted", "danger")

    users = conn.execute("SELECT * FROM users").fetchall()
    logs = conn.execute(
        "SELECT * FROM admin_logs ORDER BY timestamp DESC LIMIT 50"
    ).fetchall()
    conn.close()

    return render_template_string(ADMIN_HTML, users=users, logs=logs)




# =====================================================
# PART 2 – MARKET DATA ENGINE
# =====================================================
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

HEADERS = {"User-Agent": "Mozilla/5.0"}

VCBF_FUNDS = {
    "VCBF-TBF": "https://www.vcbf.com/vn/funds/vcbf-tbf",
    "VCBF-MGF": "https://www.vcbf.com/vn/funds/vcbf-mgf",
    "VCBF-BCF": "https://www.vcbf.com/vn/funds/vcbf-bcf",
}

HOSE_API = "https://api-finance.vietstock.vn/data/stock-price"


def crawl_vcbf_nav(fund_code: str):
    url = VCBF_FUNDS.get(fund_code)
    if not url:
        return None

    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        label = soup.find("span", string=lambda x: x and "NAV" in x)
        value = label.find_next("span").text.strip()
        nav = float(value.replace(",", ""))
        return {
            "date": datetime.now().date() - timedelta(days=1),
            "close": nav
        }
    except Exception:
        return None


def crawl_stock_price(symbol, exchange="HOSE", days=365):
    end = datetime.now()
    start = end - timedelta(days=days)

    params = {
        "symbol": symbol,
        "exchange": exchange,
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
    }

    try:
        r = requests.get(HOSE_API, params=params, headers=HEADERS, timeout=10)
        data = r.json()["data"]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return None


def load_asset(asset_type, code):
    if asset_type == "VCBF":
        nav = crawl_vcbf_nav(code)
        if not nav:
            return None
        df = pd.DataFrame([nav]).set_index("date")
        return df

    if asset_type == "STOCK":
        return crawl_stock_price(code)

    return None

# =====================================================
# PART 3 – ML ENGINE
# =====================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


def build_features(df, window=14):
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["ma"] = df["close"].rolling(window).mean()
    df["std"] = df["close"].rolling(window).std()
    df["momentum"] = df["close"] - df["close"].shift(window)
    df.dropna(inplace=True)
    return df


def train_model(df):
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


def predict_price(model, scaler, df_feat):
    row = df_feat.iloc[-1][["return", "ma", "std", "momentum"]].values.reshape(1, -1)
    row_scaled = scaler.transform(row)
    return float(model.predict(row_scaled)[0])


def generate_signal(current, predicted, threshold=0.02):
    delta = (predicted - current) / current
    if delta > threshold:
        return "BUY"
    if delta < -threshold:
        return "SELL"
    return "HOLD"

# =====================================================
# PART 4 – BACKTEST ENGINE
# =====================================================
def backtest(df, initial_cash=100_000_000):
    cash = initial_cash
    shares = 0
    history = []

    model, scaler, df_feat = train_model(df)

    for i in range(len(df_feat) - 1):
        current_price = df_feat.iloc[i]["close"]
        predicted = predict_price(model, scaler, df_feat.iloc[: i + 1])
        signal = generate_signal(current_price, predicted)

        if signal == "BUY" and cash >= current_price:
            shares = cash // current_price
            cash -= shares * current_price

        elif signal == "SELL" and shares > 0:
            cash += shares * current_price
            shares = 0

        history.append(cash + shares * current_price)

    return {
        "initial": initial_cash,
        "final": history[-1] if history else initial_cash,
        "return_pct": (history[-1] / initial_cash - 1) * 100 if history else 0
    }

# =====================================================
# PART 5 – CHART ENGINE
# =====================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def generate_chart(df, title="Price Chart"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df["close"])
    ax.set_title(title)

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    plt.close()

    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# =====================================================
# PART 6 – PORTFOLIO + PDF EXPORT
# =====================================================
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io


def export_portfolio_pdf(username, portfolio: dict):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.drawString(50, 800, f"Investment Report – {username}")

    y = 760
    for asset, amount in portfolio.items():
        pdf.drawString(50, y, f"{asset}: {amount:,.0f}")
        y -= 20

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Login</title>
    <style>
        body { font-family: Arial; background:#f4f6f8; }
        .box { width:300px; margin:120px auto; padding:20px; background:#fff; border-radius:8px; }
        input, button { width:100%; padding:10px; margin-top:10px; }
        button { background:#2c7be5; color:white; border:none; cursor:pointer; }
        .err { color:red; margin-top:10px; }
    </style>
</head>
<body>
<div class="box">
<h3>Login</h3>
<form method="post">
    <input name="username" placeholder="Username" required>
    <input type="password" name="password" placeholder="Password" required>
    <button type="submit">Login</button>
</form>
{% with msgs = get_flashed_messages(with_categories=true) %}
{% for c,m in msgs %}
<div class="err">{{ m }}</div>
{% endfor %}
{% endwith %}
</div>
</body>
</html>
"""

DASHBOARD_HTML = """
<h2>Dashboard</h2>
<p>Welcome {{ user }}</p>
<ul>
    <li><a href="/market">Market</a></li>
    <li><a href="/backtest">Backtest</a></li>
    <li><a href="/logout">Logout</a></li>
</ul>
"""
ADMIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Admin</title>
    <style>
        body { font-family: Arial; background:#f4f6f8; }
        .box { width:600px; margin:40px auto; padding:20px; background:#fff; border-radius:8px; }
        table { width:100%; border-collapse: collapse; margin-top:20px; }
        th, td { border:1px solid #ddd; padding:8px; }
        th { background:#eee; }
    </style>
</head>
<body>
<div class="box">
<h3>Admin – User Management</h3>

<h4>Admin Logs</h4>
<table>
<tr><th>Time</th><th>Admin</th><th>Action</th><th>Target</th></tr>
{% for l in logs %}
<tr>
  <td>{{ l.timestamp }}</td>
  <td>{{ l.admin_id }}</td>
  <td>{{ l.action }}</td>
  <td>{{ l.target_user }}</td>
</tr>
{% endfor %}
</table>


<form method="post">
    <input name="username" placeholder="New username" required>
    <button type="submit">Add user (default: Test@123)</button>
</form>

<table>
<tr><th>ID</th><th>Username</th><th>Role</th></tr>
{% for u in users %}
<tr>
    <td>{{ u.id }}</td>
    <td>{{ u.username }}</td>
    <td>{{ u.role }}</td>
</tr>
{% endfor %}
</table>

<br>
<a href="/logout">Logout</a>
</div>
</body>
</html>
"""


DELETE_CONFIRM_HTML = """
<h3>Confirm delete</h3>
<p>Delete user: <b>{{ user.username }}</b>?</p>

<form method="post">
  <input type="hidden" name="action" value="delete">
  <input type="hidden" name="user_id" value="{{ user.id }}">
  <button style="color:red">YES – DELETE</button>
</form>

<a href="/admin">Cancel</a>
"""



# =====================================================
# PART 7 – BUSINESS ROUTES
# =====================================================
from flask import send_file, render_template_string

# -----------------------------
# SIMPLE TEMPLATES (INLINE)
# -----------------------------
DASHBOARD_HTML = """
<h2>Investment Dashboard</h2>
<ul>
  <li><a href="/market/VCBF/VCBF-TBF">VCBF NAV</a></li>
  <li><a href="/market/STOCK/VCB">Stock VCB</a></li>
  <li><a href="/backtest/STOCK/VCB">Backtest VCB</a></li>
  <li><a href="/chart/STOCK/VCB">Chart VCB</a></li>
  <li><a href="/export_pdf">Export Portfolio PDF</a></li>
</ul>
<a href="/logout">Logout</a>
"""

RESULT_HTML = """
<h3>{{ title }}</h3>
<pre>{{ data }}</pre>
<a href="/">Back</a>
"""

CHART_HTML = """
<h3>{{ title }}</h3>
<img src="data:image/png;base64,{{ img }}">
<br><a href="/">Back</a>
"""

# -----------------------------
# DASHBOARD
# -----------------------------
@app.route("/")
@login_required
def dashboard():
    return render_template_string(DASHBOARD_HTML)

# -----------------------------
# MARKET DATA
# -----------------------------
@app.route("/market/<asset_type>/<code>")
@login_required
def market(asset_type, code):
    df = load_asset(asset_type, code)
    if df is None:
        return "Data not available", 404

    return render_template_string(
        RESULT_HTML,
        title=f"Market Data – {code}",
        data=df.tail().to_string()
    )

# -----------------------------
# ML SIGNAL
# -----------------------------
@app.route("/advisor/<asset_type>/<code>")
@login_required
def advisor(asset_type, code):
    df = load_asset(asset_type, code)
    if df is None or len(df) < 50:
        return "Not enough data", 400

    model, scaler, df_feat = train_model(df)
    predicted = predict_price(model, scaler, df_feat)
    current = df_feat.iloc[-1]["close"]
    signal = generate_signal(current, predicted)

    return render_template_string(
        RESULT_HTML,
        title=f"AI Advisor – {code}",
        data=f"Current: {current}\nPredicted: {predicted}\nSignal: {signal}"
    )

# -----------------------------
# BACKTEST
# -----------------------------
@app.route("/backtest/<asset_type>/<code>")
@login_required
def backtest_view(asset_type, code):
    df = load_asset(asset_type, code)
    if df is None or len(df) < 50:
        return "Not enough data", 400

    result = backtest(df)
    return render_template_string(
        RESULT_HTML,
        title=f"Backtest – {code}",
        data=result
    )

# -----------------------------
# CHART
# -----------------------------
@app.route("/chart/<asset_type>/<code>")
@login_required
def chart(asset_type, code):
    df = load_asset(asset_type, code)
    if df is None:
        return "Data not available", 404

    img = generate_chart(df, title=f"{code} Price Chart")
    return render_template_string(
        CHART_HTML,
        title=f"Chart – {code}",
        img=img
    )

# -----------------------------
# PDF EXPORT
# -----------------------------
@app.route("/export_pdf")
@login_required
def export_pdf():
    # DEMO portfolio – PROD: load từ DB
    portfolio = {
        "VCBF-TBF": 50_000_000,
        "VCB": 30_000_000
    }

    buffer = export_portfolio_pdf(
        username=session.get("user_id"),
        portfolio=portfolio
    )

    return send_file(
        buffer,
        as_attachment=True,
        download_name="investment_report.pdf",
        mimetype="application/pdf"
    )

if __name__ == "__main__":
    app.run()






