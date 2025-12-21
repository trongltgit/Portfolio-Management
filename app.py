import os
import sqlite3
from functools import wraps
from datetime import datetime

import pandas as pd
import numpy as np

from flask import (
    Flask, request, redirect, url_for,
    render_template_string, session, flash
)

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# =========================
# APP CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "CHANGE_ME_IN_PROD")

# =========================
# DATABASE
# =========================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT
            )
        """)
        # default admin
        cur = db.execute("SELECT * FROM users WHERE username='admin'")
        if not cur.fetchone():
            db.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
                (
                    "admin",
                    generate_password_hash("Admin@123"),
                    "admin",
                    datetime.utcnow().isoformat()
                )
            )


init_db()

# =========================
# AUTH DECORATORS
# =========================
def login_required(role=None):
    def wrapper(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            if "user" not in session:
                return redirect(url_for("login"))
            if role and session.get("role") != role:
                return "Forbidden", 403
            return fn(*args, **kwargs)
        return decorated
    return wrapper


# =========================
# ROUTES – AUTH
# =========================
@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")

        with get_db() as db:
            cur = db.execute("SELECT * FROM users WHERE username=?", (u,))
            user = cur.fetchone()

        if user and check_password_hash(user["password_hash"], p):
            session["user"] = u
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))

        flash("Invalid credentials")

    return render_template_string("""
        <h2>Login</h2>
        <form method="post">
            <input name="username" placeholder="Username" required>
            <input name="password" type="password" placeholder="Password" required>
            <button>Login</button>
        </form>
        {% for m in get_flashed_messages() %}<p>{{m}}</p>{% endfor %}
    """)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# =========================
# DASHBOARD
# =========================
@app.route("/dashboard")
@login_required()
def dashboard():
    return render_template_string("""
        <h1>Dashboard</h1>
        <p>User: {{session.user}} ({{session.role}})</p>

        <ul>
            <li><a href="/upload-csv">CSV Analysis</a></li>
            <li><a href="/upload-image">Image Analysis</a></li>
            {% if session.role == 'admin' %}
                <li><a href="/admin/users">User Management</a></li>
            {% endif %}
            <li><a href="/logout">Logout</a></li>
        </ul>
    """)


# =========================
# CSV ANALYSIS
# =========================
@app.route("/upload-csv", methods=["GET", "POST"])
@login_required()
def upload_csv():
    result = None

    if request.method == "POST":
        f = request.files.get("file")
        if f:
            path = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
            f.save(path)
            df = pd.read_csv(path)
            result = df.describe().to_html()

    return render_template_string("""
        <h2>CSV Analysis</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <button>Upload</button>
        </form>
        <div>{{result|safe}}</div>
        <a href="/dashboard">Back</a>
    """, result=result)


# =========================
# IMAGE ANALYSIS (PLACEHOLDER)
# =========================
@app.route("/upload-image", methods=["GET", "POST"])
@login_required()
def upload_image():
    msg = None
    if request.method == "POST":
        f = request.files.get("file")
        if f:
            path = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
            f.save(path)
            msg = "Image uploaded successfully (model hook ready)"

    return render_template_string("""
        <h2>Image Upload</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <button>Upload</button>
        </form>
        {% if msg %}<p>{{msg}}</p>{% endif %}
        <a href="/dashboard">Back</a>
    """, msg=msg)


# =========================
# ADMIN – USER MANAGEMENT
# =========================
@app.route("/admin/users", methods=["GET", "POST"])
@login_required(role="admin")
def admin_users():
    with get_db() as db:
        if request.method == "POST":
            u = request.form["username"]
            p = request.form["password"]
            r = request.form["role"]
            try:
                db.execute(
                    "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
                    (u, generate_password_hash(p), r, datetime.utcnow().isoformat())
                )
            except sqlite3.IntegrityError:
                flash("User already exists")

        users = db.execute("SELECT username, role FROM users").fetchall()

    return render_template_string("""
        <h2>User Management</h2>
        <form method="post">
            <input name="username" placeholder="Username" required>
            <input name="password" placeholder="Password" required>
            <select name="role">
                <option value="user">User</option>
                <option value="admin">Admin</option>
            </select>
            <button>Add</button>
        </form>

        <ul>
            {% for u in users %}
                <li>{{u.username}} - {{u.role}}</li>
            {% endfor %}
        </ul>

        <a href="/dashboard">Back</a>
    """, users=users)


# =========================
# ENTRY POINT (LOCAL ONLY)
# =========================
if __name__ == "__main__":
    app.run(debug=True)
