# market/__init__.py
from flask import Blueprint

market_bp = Blueprint(
    "market",
    __name__,
    url_prefix="/market"
)

from . import routes
