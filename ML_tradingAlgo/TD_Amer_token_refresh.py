import requests
import pandas as pd
import time
import json
import os
from tda import auth, client
import tda
from TDAmConfig import client_id
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

token_path = 'token.pickle'
api_key = client_id
redirect_uri = 'http://localhost:8080'

try:
    c = auth.client_from_token_file(token_path, api_key)
except FileNotFoundError:
    from selenium import webdriver
    from chromedriver_py import binary_path # this will get you the path variable

    driver = webdriver.Chrome(executable_path=binary_path)
    #time.sleep(5)
    #driver.get("http://www.python.org")
    #assert "Python" in driver.title
    c = auth.client_from_login_flow(
        driver, api_key, redirect_uri, token_path)
    print("done")