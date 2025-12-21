from flask import Blueprint

market_bp = Blueprint(
    "market",
    __name__,
    template_folder="templates",
    url_prefix="/market"
)

# import routes cuối cùng để tránh circular import
from . import routes
