import os
from dotenv import load_dotenv
load_dotenv()

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
BROKERAGE_ACCOUNT_ID = os.environ["BROKERAGE_ACCOUNT_ID"]
WHATSAPP_FROM = os.environ["WHATSAPP_FROM"]
WHATSAPP_TO = os.environ["WHATSAPP_TO"]
TD_AMERITRADE_CLIENT_ID = os.environ["TD_AMERITRADE_CLIENT_ID"]
