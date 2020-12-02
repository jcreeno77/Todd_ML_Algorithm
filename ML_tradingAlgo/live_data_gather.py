import requests
import pandas as pd
import json
import os
import time
import numpy as np
import tda
from tda import auth, client
from TDAmConfig import client_id
from TD_Ameritrade_Data import arrange_fundamentals
#from Todd_tradingAlgo import Todd_predict


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

    sleep_seconds = 1.8


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

    #Gathers data
    while running == True:
        
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
            candle_volume = stockData[ticker]['totalVolume'] - start_vol
            

            #adds data to lists
            temp_OneMinCandle.append(candle_open)
            temp_OneMinCandle.append(candle_high)
            temp_OneMinCandle.append(candle_low)
            temp_OneMinCandle.append(candle_close)
            temp_OneMinCandle.append(candle_volume)

            all_OneMinCandles.append(temp_OneMinCandle)
            print(all_OneMinCandles)

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
            candle_volume_5min = stockData[ticker]['totalVolume'] - start_vol_5min

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
            if len(all_FiveMinCandles) >= 8:

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
                fiftyTwo_week_high = stockData[ticker]['52WkHigh']
                fiftyTwo_week_low = stockData[ticker]['52WkLow']
                high52ratio = 1 - stockPrice / fiftyTwo_week_high
                low52ratio = 1 - fiftyTwo_week_low / stockPrice

                #HERE IS WHERE PREDICTIONS HAPPEN
                stock_data = arrange_from_live(all_FiveMinCandles, all_OneMinCandles, floatShares,high52ratio,low52ratio, ratio_premarketHigh, ratio_premarketLow, daily_high_ratio, daily_low_ratio)
                write_to_csv_for_prediction(stock_data)




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


        feature = (((feature_close - feature_low) - (feature_high - feature_close))/feature_open * 1000) * (feature_vol/float_volume*100)
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
    filename = 'DataToPredict.csv'
    with open(filename, 'w') as file:
        fiveMinData.to_csv(file)


if __name__ == '__main__':
    main()