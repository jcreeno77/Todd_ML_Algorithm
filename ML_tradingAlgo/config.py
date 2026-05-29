import os
from dotenv import load_dotenv
load_dotenv()

BROKERAGE_ACCOUNT_ID = os.environ["BROKERAGE_ACCOUNT_ID"]
TD_AMERITRADE_CLIENT_ID = os.environ["TD_AMERITRADE_CLIENT_ID"]

# --- Alerting (Twilio removed) ---
# Channel-agnostic notifier (ML_tradingAlgo/data/notify.py) reads this directly.
# Unset -> alerts log only; set to a Discord webhook URL to also POST there.
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")

# --- Trading execution (Schwab) ---
# Real-money safety switch. Default OFF: orders are logged (dry-run) but NOT sent
# to Schwab (which has no paper-trading API). Set TRADING_ENABLED=true for live orders.
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "false")

# --- Schwab API / S3 storage / Fundamentals (TFT data pipeline) ---
# Optional at import time (os.getenv -> None when unset) so the module loads
# in dev without the new pipeline configured.
SCHWAB_APP_KEY = os.getenv("SCHWAB_APP_KEY")
SCHWAB_APP_SECRET = os.getenv("SCHWAB_APP_SECRET")
SCHWAB_CALLBACK_URL = os.getenv("SCHWAB_CALLBACK_URL")
SCHWAB_TOKEN_PATH = os.getenv("SCHWAB_TOKEN_PATH")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "")

FUNDAMENTALS_PROVIDER = os.getenv("FUNDAMENTALS_PROVIDER", "yahoo")  # fmp | yahoo
FMP_API_KEY = os.getenv("FMP_API_KEY")
