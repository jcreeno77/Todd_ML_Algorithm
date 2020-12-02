import requests
import pandas as pd
import json
import os
import time
import functools
import tensorflow as tf
import numpy as np
import tda
from tensorflow import keras
from functools import partial
from tensorflow.keras import layers
from keras.layers.advanced_activations import LeakyReLU
from tda import auth, client
from TDAmConfig import client_id
from TD_Ameritrade_Data import arrange_fundamentals
from Todd_tradingAlgo1 import Todd_predict

#Sets client for SMS messaging
from twilio.rest import Client
account_id = 'AC5581f6879047db6004b52f7412c61755'
token = '6f5a1f0fd68c32d21900a5f77729e7a8'
client = Client(account_id, token)

#money to spend per trade
trade_amount = 25



def main():

    #all info for OAuth 2.0
    token_path = 'token.pickle'
    api_key = client_id
    redirect_uri = 'http://localhost:8080'

    #All beginning information for creating candles
    ticker = input("Ticker: ")
    Market_open = input("9:30am Time for the day in epoch (seconds): ")
    Market_open = int(Market_open)
    running = True
    runningPremarket = True
    end_time = Market_open + 10800

    #GETS PREVIOUS DAY CLOSE
    endpointHist = r"https://api.tdameritrade.com/v1/marketdata/{}/pricehistory".format(ticker)

    #define payload
    payloadHist = {'apikey': client_id, 'periodType': 'day', 'period': '1', 'frequencyType': 'minute', 'frequency': '30', 'needExtendedHoursData' : 'false'} #need to be able to set time period

    #make a request
    contentHist = requests.get(url = endpointHist, params = payloadHist)

    dataHist = contentHist.json()
    dataHist = json.dumps(dataHist)
    stockData = pd.read_json(dataHist)

    previous_day_close = stockData['candles'].iloc[-1]['close']


    oneMinContent = []
    fiveMinContent = []
    preMarketContent = []



    #One minute lists (to be assembled into data later)
    temp_OneMinCandle = []
    all_OneMinCandles = []

    #Five minute lists (to be assembled into data later)
    temp_FiveMinCandle = []
    all_FiveMinCandles = []


    #Authorizes for live data gathering
    try:
        c = auth.client_from_token_file(token_path, api_key)
    except FileNotFoundError:
        from selenium import webdriver
        from chromedriver_py import binary_path # this will get you the path variable

        driver = webdriver.Chrome(executable_path=binary_path)
        #driver.get("http://www.python.org")
        assert "Python" in driver.title
        c = auth.client_from_login_flow(
            driver, api_key, redirect_uri, token_path)

    #Get starting volume
    r = c.get_quote(ticker)
    assert r.ok, r.raise_for_status()
    quote = json.dumps(r.json())
    stockData = pd.read_json(quote)

    print(stockData[ticker])
    start_vol = stockData[ticker]['totalVolume']
    start_vol_5min = stockData[ticker]['totalVolume']
    ratio_premarketHigh = 0
    ratio_premarketLow = 0

    floatShares = arrange_fundamentals(ticker)[2]
    print(floatShares)

    sleep_seconds = 1.4


    #The vibe check for preMarket
    if time.time() >= Market_open:
        runningPremarket = False

    while runningPremarket == True:

        print("running premarket")
        time.sleep(sleep_seconds)
        try:
            r = c.get_quote(ticker)
            quote = json.dumps(r.json())
            stockData = pd.read_json(quote)
            print(stockData[ticker]['lastPrice'])
            stockPrice = stockData[ticker]['lastPrice']
            
        except:
            print("Error. Something wrong in the pandas import. Keep trying?")

        preMarketContent.append(stockPrice)
        preMarketHigh = max(preMarketContent)
        preMarketLow = min(preMarketContent)

        if time.time() >= Market_open:
            ratio_premarketHigh = 1 - preMarketContent[-1]/preMarketHigh
            ratio_premarketLow = 1 - preMarketContent[-1]/preMarketLow
            print(ratio_premarketHigh)
            runningPremarket = False
            print("ending premarket")



    #All time info
    start_time_oneMin = int(time.time())
    start_time_fiveMin = int(time.time())
    
    bought = False
    sold1 = False
    sold2 = False
    noted1 = False
    noted2 = False

    #Gathers data
    while running == True:
        
        if time.time() > end_time:
            running = False
        
        current_time_oneMin = int(time.time()) - start_time_oneMin
        current_time_fiveMin = int(time.time()) - start_time_fiveMin
        print(current_time_oneMin)
        print(current_time_fiveMin)

        time.sleep(sleep_seconds)

        
        try:
            r = c.get_quote(ticker)
            quote = json.dumps(r.json())
            stockData = pd.read_json(quote)
            print(stockData[ticker]['lastPrice'])
            stockPrice = stockData[ticker]['lastPrice']
            
        except:
            print("Error. Something wrong in the pandas import. Keep trying?")




        #adds to the main data
        
        oneMinContent.append(stockPrice)
        fiveMinContent.append(stockPrice)


        if current_time_oneMin >= 60:
            #calculates data values for candles
            candle_high = max(oneMinContent)
            candle_low = min(oneMinContent)
            candle_open = oneMinContent[0]
            candle_close = oneMinContent[-1]
            try:
                candle_volume = stockData[ticker]['totalVolume'] - start_vol
            except:
                print("error getting candle volume")
                try:
                    candle_volume = stockData[ticker]['totalVolume'] - start_vol
                except:
                    candle_volume = 1
            

            #adds data to lists
            temp_OneMinCandle.append(candle_open)
            temp_OneMinCandle.append(candle_high)
            temp_OneMinCandle.append(candle_low)
            temp_OneMinCandle.append(candle_close)
            temp_OneMinCandle.append(candle_volume)

            all_OneMinCandles.append(temp_OneMinCandle)
            print(all_OneMinCandles)

            #cancel if stock falls too low
            currentCandleOpen = all_OneMinCandles[-1][0]

            percent_change = (currentCandleOpen - previous_day_close) / previous_day_close * 100
            if percent_change <= 10:
                running = False


            #resets the candle
            oneMinContent = []
            temp_OneMinCandle = []
            start_vol = start_vol + candle_volume
            start_time_oneMin += 60

        #sets the five minute candle
        if current_time_fiveMin >= 300:
            #calculates data values for candles
            candle_high_5min = max(fiveMinContent)
            candle_low_5min = min(fiveMinContent)
            candle_open_5min = fiveMinContent[0]
            candle_close_5min = fiveMinContent[-1]
            try:
                candle_volume_5min = stockData[ticker]['totalVolume'] - start_vol_5min
            except:
                print("failure for 5min, trying again.")
                try:
                    candle_volume_5min = stockData[ticker]['totalVolume'] - start_vol_5min
                except:
                    print("failed")
            
            #adds data to lists
            temp_FiveMinCandle.append(candle_open_5min)
            temp_FiveMinCandle.append(candle_high_5min)
            temp_FiveMinCandle.append(candle_low_5min)
            temp_FiveMinCandle.append(candle_close_5min)
            temp_FiveMinCandle.append(candle_volume_5min)


            all_FiveMinCandles.append(temp_FiveMinCandle) #1open, 2high, 3low, 4close, 5volume
            print(all_FiveMinCandles)

            #resets the candle
            fiveMinContent = []
            temp_FiveMinCandle = []
            start_vol_5min = start_vol_5min + candle_volume_5min
            start_time_fiveMin += 300


            #HERE BEGETS THE ARRANGING OF INFORMATION: BEGIN!
            if len(all_FiveMinCandles) >= 8 and bought == False and time.time() < (end_time - 900):

                #Gets daily high
                daily_highs = []
                daily_lows = []
                for item in all_FiveMinCandles:
                    daily_highs.append(item[1])
                    daily_lows.append(item[2])
                daily_high = max(daily_highs)
                daily_low = min(daily_lows)

                daily_high_ratio = 1 - all_FiveMinCandles[-1][3]/daily_high
                daily_low_ratio = 1 - daily_low/all_FiveMinCandles[-1][3]


                #get 52 week data and ratios
                try:
                    fiftyTwo_week_high = stockData[ticker]['52WkHigh']
                    fiftyTwo_week_low = stockData[ticker]['52WkLow']
                    high52ratio = 1 - stockPrice / fiftyTwo_week_high
                    low52ratio = 1 - fiftyTwo_week_low / stockPrice
                    
                    stock_data = arrange_from_live(all_FiveMinCandles, all_OneMinCandles, floatShares,high52ratio,low52ratio, ratio_premarketHigh, ratio_premarketLow, daily_high_ratio, daily_low_ratio)
                    write_to_csv_for_prediction(stock_data)
                except:
                    print("problem importing")

                #here is where the algo makes a prediction
                prediction1, prediction2 = Todd_predict()
                prediction1 = prediction1[0][0]
                prediction2 = prediction2[0][0]
                print("PREDICTION: ")
                
                if prediction1 == 1 and prediction2 == 1:
                    prediction = 1
                    print(prediction)
                else:
                    prediction = 0
                    print(prediction)
                #This is where the buy occurs
                if prediction == 1:
                    bought = True
                    buy_price = stockPrice
                    buy_time = int(time.time())
                    buyTime_since_open = (buy_time - Market_open) / 60
                    sms_text = "Todd just bought " + str(ticker) + " for " + str(buy_price) + " at " + str(buy_time)
                    
                    
                    

                    #organizes the buy amount
                    buy_quantity = round(trade_amount/stockPrice)
                    buy_remainder = 2 - (buy_quantity % 2)
                    buy_quantity += buy_remainder
                    
                    #places trade
                    account_id = '426767497'
                    x = tda.orders.equities.equity_buy_market(ticker, buy_quantity)
                    r_place_order = c.place_order(account_id, x)
                    #assert r_place_order.ok, r_place_order.raise_for_status()
                    #order_id = tda.utils.Utils(client, account_id).extract_order_id(r_place_order)


        #All following code pertains to the sell mechanism - includes a trailing stop loss of 1%
        if bought == False:
            stop_loss_set = False
            begin_stoploss_trail = False
            stop_loss_trail = -2.5
        if bought == True:
            try:
                assert r_place_order.ok, r_place_order.raise_for_status()
                order_id = tda.utils.Utils(client, account_id).extract_order_id(r_place_order)
            except:
                print("order did not go through (normally not enough money in account)")
                
            percent_change = (stockPrice - buy_price) / buy_price * 100
            print("percent change")
            print(percent_change)

            time_since_bought = (int(time.time()) - buy_time) / 60
            

            if time_since_bought > 15:
                profit = (stockPrice - buy_price)
                sold1 = True
                sold2 = True

            begin_stoploss_trail = True
            
            if begin_stoploss_trail == True:
                stop_loss_trail = max(stop_loss_trail, (percent_change - 2.5))
                if percent_change <= stop_loss_trail:
                    profit = (stockPrice - buy_price)
                    sold1 = True
            
            if percent_change <= -2:
                profit = (stockPrice - buy_price)
                sold2 = True
                
            if percent_change >= 3:
                profit = (stockPrice - buy_price)
                sold2 = True


            
        if sold1 == True and noted1 == False:
            print("profit equals: ", profit)
            filename = 'trade_history.txt'
            time_since_open = (int(time.time()) - Market_open) / 60
            to_write = str(ticker) + " " + str(profit) + "Time Bought: " + str(buyTime_since_open) +  " Time Sold: " + str(time_since_bought) + " Percent: " + str(percent_change) + " SOLD 2" + "\n"
            with open(filename, "a") as file:
                file.write(to_write)
            noted1 = True
            
            sms_text = "Todd just sold " + str(ticker) + " for a percent change of " + str(percent_change) + " SOLD 2"
            message = client.messages.create(
                body=sms_text, 
                from_='whatsapp:+14155238886',
                to='whatsapp:+18134096031')

            #sell code
            sell_quantity = buy_quantity/2
            account_id = '426767497'
            x = tda.orders.equities.equity_sell_market(ticker, sell_quantity)
            r_place_order = c.place_order(account_id, x)

        if sold2 == True and noted2 == False:
            print("profit equals: ", profit)
            filename = 'trade_history.txt'
            time_since_open = (int(time.time()) - Market_open) / 60
            to_write = str(ticker) + " " + str(profit) + "Time Bought: " + str(buyTime_since_open) +  " Time Sold: " + str(time_since_bought) + " Percent: " + str(percent_change) + " SOLD 3" + "\n"
            with open(filename, "a") as file:
                file.write(to_write)
            noted2 = True

            sms_text = "Todd just sold " + str(ticker) + " for a percent change of " + str(percent_change) + " SOLD 3"
            message = client.messages.create(
                body=sms_text, 
                from_='whatsapp:+14155238886',
                to='whatsapp:+18134096031')

            #sell code
            sell_quantity = buy_quantity/2
            account_id = '426767497'
            x = tda.orders.equities.equity_sell_market(ticker, sell_quantity)
            r_place_order = c.place_order(account_id, x)

        if sold1 == True and sold2 == True:
            bought = False
            sold1 = False
            sold2 = False
            noted1 = False
            noted2 = False

def arrange_from_live(fiveMinCandlesList, oneMinCandlesList, float_volume, high52ratio, low52ratio, ratio_premarketHigh, ratio_premarketLow, daily_high_ratio, daily_low_ratio):
    

    fiveMinData = pd.DataFrame({"goodNews": [],"Earnings": [],"52highRatio": [], "52lowRatio": [],"preMarket_high_ratio":[],"premarket_low_ratio":[],"ratioToDailyHigh":[],"ratioToDailyLow":[],"5min8":[],"5min8Unweighted":[],"5min8Squar":[],"5min7":[],"5min7Unweighted":[],"5min7Squar":[],"5min6":[],"5min6Unweighted":[],"5min6Squar":[],"5min5":[],"5min5Unweighted":[],"5min5Squar":[],"5min4":[],"5min4Unweighted":[],"5min4Squar":[],"5min3":[],"5min3Unweighted":[],"5min3Squar":[],"5min2":[],"5min2Unweighted":[],"5min2Squar":[],"5min1":[],"5min1Unweighted":[],"5min1Squar":[],"1min5":[],"1min5Unweighted":[],"1min5Squar":[],"1min4":[],"1min4Unweighted":[],"1min4Squar":[],"1min3":[],"1min3Unweighted":[],"1min3Squar":[],"1min2":[],"1min2Unweighted":[],"1min2Squar":[],"1min1":[],"1min1Unweighted":[],"1min1Squar":[]})

    fiveMinFeats = []
    for i in range(0,8):
        add_location_5min = i + len(fiveMinCandlesList) - 8
        cande_feature = fiveMinCandlesList[add_location_5min]
        feature_open = cande_feature[0]
        feature_high = cande_feature[1]
        feature_low = cande_feature[2]
        feature_close = cande_feature[3]
        feature_vol = cande_feature[4]

        try:
            feature = (((feature_close - feature_low) - (feature_high - feature_close))/feature_open * 1000) * (feature_vol/float_volume*100)
        except:
            try:
                feature = (((feature_close - feature_low) - (feature_high - feature_close))/feature_open * 1000) * (feature_vol/float_volume*100)
            except:
                feature = 0
        feature_unweighted = (((feature_close - feature_low) - (feature_high - feature_close))/feature_open * 1000)
        feature_squared = feature ** 2

        fiveMinFeats.append(feature)
        fiveMinFeats.append(feature_unweighted)
        fiveMinFeats.append(feature_squared)

    for o in range(0,5):
        add_location_1min = o + len(oneMinCandlesList) - 5
        cande_feature = oneMinCandlesList[add_location_1min]
        feature_open = cande_feature[0]
        feature_high = cande_feature[1]
        feature_low = cande_feature[2]
        feature_close = cande_feature[3]
        feature_vol = cande_feature[4]
        
        feature = (((feature_close - feature_low) - (feature_high - feature_close))/feature_open * 1000) * (feature_vol/float_volume*100)
        feature_unweighted = (((feature_close - feature_low) - (feature_high - feature_close))/feature_open * 1000)
        feature_squared = feature ** 2

        fiveMinFeats.append(feature)
        fiveMinFeats.append(feature_unweighted)
        fiveMinFeats.append(feature_squared)
    
    earnings = 0
    good_news = 1
    
    fiveMinFeats.insert(0,daily_low_ratio)
    fiveMinFeats.insert(0,daily_high_ratio)
    fiveMinFeats.insert(0,ratio_premarketLow)
    fiveMinFeats.insert(0,ratio_premarketHigh)
    fiveMinFeats.insert(0,low52ratio)
    fiveMinFeats.insert(0,high52ratio)
    fiveMinFeats.insert(0,earnings)
    fiveMinFeats.insert(0,good_news)
    new_row = pd.Series(fiveMinFeats, index = fiveMinData.columns)
    fiveMinData = fiveMinData.append(new_row, ignore_index = True)
    y_list = [0]
    fiveMinData["y_list"] = y_list
    return fiveMinData


def write_to_csv_for_prediction(fiveMinData):
    filename = 'DataToPredict1.csv'
    with open(filename, 'w') as file:
        fiveMinData.to_csv(file)


if __name__ == '__main__':
    main()