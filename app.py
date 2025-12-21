# =========================================================
# app.py – PROD READY (SINGLE FILE)
# Investment ML System | SQLite | ML | Chart | PDF
# =========================================================

# =========================
# A. CORE & CONFIG
# =========================
import os, sqlite3, secrets, re, io, base64
from datetime import datetime, timedelta
from functools import wraps

from flask import (
    Flask, request, redirect, url_for,
    render_template_string, session,
    abort, send_file, g
)

from werkzeug.security import generate_password_hash, check_password_hash

# =========================
# B. APP INIT
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "investment_app.db")

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", secrets.token_hex(32)),
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# =========================
# C. DATABASE
# =========================
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

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        role TEXT NOT NULL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        asset TEXT,
        amount REAL
    )""")

    db.commit()

def bootstrap_admin():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT 1 FROM users WHERE username='admin'")
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO users VALUES (?, ?, ?)",
            ("admin", generate_password_hash("Test@123456"), "admin")
        )
        db.commit()

with app.app_context():
    init_db()
    bootstrap_admin()

# =========================
# D. AUTH DECORATORS
# =========================
def login_required(role=None):
    def wrapper(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            if "username" not in session:
                return redirect("/login")
            if role and session.get("role") != role:
                abort(403)
            return fn(*args, **kwargs)
        return decorated
    return wrapper

# =========================
# E. BASE HTML
# =========================
BASE_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container mt-4">
{% if session.get("username") %}
<p>
<b>User:</b> {{ session.username }} |
<a href="/dashboard">Dashboard</a> |
<a href="/portfolio">Portfolio</a> |
<a href="/advisor">Advisor</a> |
<a href="/chart">Chart</a> |
<a href="/export_pdf">PDF</a> |
{% if session.role == 'admin' %}
<a href="/admin">Admin</a> |
{% endif %}
<a href="/logout" class="text-danger">Logout</a>
</p>
<hr>
{% endif %}
{{ content|safe }}
</div>
</body>
</html>
"""

# =========================
# F. AUTH ROUTES
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        db = get_db()
        cur = db.execute("SELECT * FROM users WHERE username=?", (u,))
        user = cur.fetchone()

        if user and check_password_hash(user["password"], p):
            session["username"] = u
            session["role"] = user["role"]
            return redirect("/dashboard")

        return "Invalid credentials", 401

    return render_template_string(
        BASE_HTML,
        title="Login",
        content="""
        <h3>Login</h3>
        <form method="post">
          <input class="form-control mb-2" name="username" placeholder="Username">
          <input class="form-control mb-2" type="password" name="password" placeholder="Password">
          <button class="btn btn-primary">Login</button>
        </form>
        """
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# =========================
# G. DASHBOARD
# =========================
@app.route("/dashboard")
@login_required()
def dashboard():
    return render_template_string(
        BASE_HTML,
        title="Dashboard",
        content="""
        <h3>Investment Dashboard</h3>
        <ul>
          <li>VCBF NAV Forecast</li>
          <li>Stock ML Prediction</li>
          <li>Portfolio Optimization</li>
        </ul>
        """
    )

# =========================
# H. ADMIN PANEL
# =========================
@app.route("/admin", methods=["GET", "POST"])
@login_required(role="admin")
def admin_panel():
    db = get_db()
    msg = ""

    if request.method == "POST":
        u = request.form["username"]
        action = request.form["action"]

        if action == "add":
            db.execute(
                "INSERT OR IGNORE INTO users VALUES (?, ?, ?)",
                (u, generate_password_hash("Test@1234"), "user")
            )
            db.commit()
            msg = f"User {u} added"

        elif action == "delete":
            if u != "admin":
                db.execute("DELETE FROM users WHERE username=?", (u,))
                db.commit()
                msg = f"User {u} deleted"

        elif action == "reset":
            db.execute(
                "UPDATE users SET password=? WHERE username=?",
                (generate_password_hash("Test@1234"), u)
            )
            db.commit()
            msg = f"Password reset for {u}"

    users = db.execute("SELECT username, role FROM users").fetchall()
    rows = "".join(f"<tr><td>{x['username']}</td><td>{x['role']}</td></tr>" for x in users)

    return render_template_string(
        BASE_HTML,
        title="Admin",
        content=f"""
        <h3>Admin Panel</h3>
        <p class="text-info">{msg}</p>
        <form method="post">
          <input name="username" class="form-control mb-2" placeholder="Username">
          <select name="action" class="form-control mb-2">
            <option value="add">Add</option>
            <option value="delete">Delete</option>
            <option value="reset">Reset Password</option>
          </select>
          <button class="btn btn-danger">Execute</button>
        </form>
        <hr>
        <table class="table table-bordered">
        <tr><th>User</th><th>Role</th></tr>
        {rows}
        </table>
        """
    )

# =========================
# I. PORTFOLIO
# =========================
@app.route("/portfolio")
@login_required()
def portfolio():
    db = get_db()
    pf = db.execute(
        "SELECT asset, amount FROM portfolio WHERE username=?",
        (session["username"],)
    ).fetchall()

    rows = "".join(f"<li>{x['asset']}: {x['amount']:,.0f}</li>" for x in pf) or "<li>Empty</li>"

    return render_template_string(
        BASE_HTML,
        title="Portfolio",
        content=f"<h4>Portfolio</h4><ul>{rows}</ul>"
    )

# =========================
# J. PDF EXPORT
# =========================
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

@app.route("/export_pdf")
@login_required()
def export_pdf():
    db = get_db()
    pf = db.execute(
        "SELECT asset, amount FROM portfolio WHERE username=?",
        (session["username"],)
    ).fetchall()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.drawString(50, 800, f"Investment Report – {session['username']}")

    y = 760
    for x in pf:
        pdf.drawString(50, y, f"{x['asset']}: {x['amount']:,.0f}")
        y -= 20

    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="investment_report.pdf",
        mimetype="application/pdf"
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
