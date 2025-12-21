# app.py
from flask import Flask, redirect, url_for
from config import Config
from db import init_db, seed_admin

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # INIT DB
    with app.app_context():
        init_db()
        seed_admin()

    # AUTH ROUTES (TEMP – PART 2 sẽ tách)
    from flask import request, render_template_string, session, flash
    from werkzeug.security import check_password_hash
    from db import get_db

    LOGIN_HTML = """
    <h3>Login</h3>
    <form method="post">
        <input name="username" placeholder="Username" required><br>
        <input type="password" name="password" placeholder="Password" required><br>
        <button>Login</button>
    </form>
    {% for m in get_flashed_messages() %}
      <p style="color:red">{{ m }}</p>
    {% endfor %}
    """

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

            if not user:
                flash("Invalid credentials")
            elif user["locked"]:
                flash("Account locked")
            elif check_password_hash(user["password_hash"], password):
                session["user_id"] = user["id"]
                session["role"] = user["role"]
                return redirect("/dashboard")
            else:
                flash("Invalid credentials")

        return render_template_string(LOGIN_HTML)

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/dashboard")
    def dashboard():
        return "Dashboard placeholder – PART 4"

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
