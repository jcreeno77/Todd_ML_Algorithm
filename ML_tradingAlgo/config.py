import os
from dotenv import load_dotenv
load_dotenv()

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
BROKERAGE_ACCOUNT_ID = os.environ["BROKERAGE_ACCOUNT_ID"]
WHATSAPP_FROM = os.environ["WHATSAPP_FROM"]
WHATSAPP_TO = os.environ["WHATSAPP_TO"]
TD_AMERITRADE_CLIENT_ID = os.environ["TD_AMERITRADE_CLIENT_ID"]

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
