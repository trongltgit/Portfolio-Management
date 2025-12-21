# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "CHANGE_ME_IN_PROD")
    DB_PATH = os.path.join(BASE_DIR, "app.db")
