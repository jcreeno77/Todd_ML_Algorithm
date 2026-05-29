import sys
import pandas as pd
import os
import time
import numpy as np
import config  # noqa: F401 - imported for the load_dotenv() side-effect (populates os.environ)
from Todd_tradingAlgo1 import Todd_predict

# Channel-agnostic alerting (Twilio removed). Defaults to logging; POSTs to a
# Discord webhook when ALERT_WEBHOOK_URL is set. Import works whether the loop is
# run from the repo root (as a module) or from within ML_tradingAlgo/.
try:
    from ML_tradingAlgo.data.notify import notify
except ImportError:  # run with CWD inside ML_tradingAlgo/
    from data.notify import notify

# Schwab data-source / order-execution layer (replaces the dead TD Ameritrade
# API). Same dual-import idiom as notify above.
try:
    from ML_tradingAlgo.data import schwab_client, schwab_trader
    from ML_tradingAlgo.data.token_health import check_token_freshness
except ImportError:  # run with CWD inside ML_tradingAlgo/
    from data import schwab_client, schwab_trader
    from data.token_health import check_token_freshness

#money to spend per trade
trade_amount = 25

# Instance ID determines CSV filename (e.g. DataToPredict1.csv, DataToPredict2.csv)
instance_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# Optional non-blocking warehouse tee. Guarded so the trading loop is completely
# unaffected if S3 / the tee is unconfigured or fails to construct: on any error
# tee stays None and every `if tee:` call site below is a no-op.
tee = None
try:
    from ML_tradingAlgo.data.tee import LiveTee
    tee = LiveTee(instance_id)
    tee.start()
except Exception as _tee_exc:  # noqa: BLE001 - never let the tee break trading
    print(f"LiveTee disabled: {_tee_exc}")
    tee = None


def main():

    # Best-effort Schwab token freshness check; never block startup on failure.
    try:
        check_token_freshness(notify=notify)
    except Exception as _token_exc:  # noqa: BLE001
        print(f"token freshness check skipped: {_token_exc}")

    #All beginning information for creating candles
    ticker = input("Ticker: ")
    Market_open = input("9:30am Time for the day in epoch (seconds): ")
    Market_open = int(Market_open)
    running = True
    runningPremarket = True
    end_time = Market_open + 10800


    #GETS PREVIOUS DAY CLOSE
    previous_day_close = schwab_client.get_prior_close(ticker)


    oneMinContent = []
    fiveMinContent = []
    preMarketContent = []



    #One minute lists (to be assembled into data later)
    temp_OneMinCandle = []
    all_OneMinCandles = []

    #Five minute lists (to be assembled into data later)
    temp_FiveMinCandle = []
    all_FiveMinCandles = []


    #Get starting volume
    q = schwab_client.get_live_quote(ticker)
    print(q)
    start_vol = q["total_volume"]
    start_vol_5min = q["total_volume"]
    ratio_premarketHigh = 0
    ratio_premarketLow = 0

    #Float lookup from Schwab fundamentals
    try:
        _f = schwab_client.get_fundamentals([ticker])
        _match = _f.loc[_f["symbol"] == ticker, "float_shares"]
        floatShares = float(_match.iloc[0]) if not _match.empty else None
    except Exception:
        floatShares = None
    if not floatShares:
        notify("Could not resolve float shares for " + str(ticker) + "; defaulting to 0", level="warning")
        floatShares = 0
    print(floatShares)

    sleep_seconds = 1.4


    #The vibe check for preMarket
    if time.time() >= Market_open:
        runningPremarket = False

    while runningPremarket == True:

        print("running premarket")
        time.sleep(sleep_seconds)
        try:
            q = schwab_client.get_live_quote(ticker)
            print(q["last_price"])
            stockPrice = q["last_price"]

        except Exception:
            print("Error. Something wrong in the quote fetch. Keep trying?")

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
    sold3 = False
    noted1 = False
    noted2 = False
    noted3 = False

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
            q = schwab_client.get_live_quote(ticker)
            print(q["last_price"])
            stockPrice = q["last_price"]

        except Exception:
            print("Error. Something wrong in the quote fetch. Keep trying?")




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
                candle_volume = q["total_volume"] - start_vol
            except Exception:
                print("error getting candle volume")
                try:
                    candle_volume = q["total_volume"] - start_vol
                except Exception:
                    candle_volume = 1


            #adds data to lists
            temp_OneMinCandle.append(candle_open)
            temp_OneMinCandle.append(candle_high)
            temp_OneMinCandle.append(candle_low)
            temp_OneMinCandle.append(candle_close)
            temp_OneMinCandle.append(candle_volume)

            all_OneMinCandles.append(temp_OneMinCandle)
            print(all_OneMinCandles)

            if tee: tee.put_candle(ticker, temp_OneMinCandle, "1min", None)


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
                candle_volume_5min = q["total_volume"] - start_vol_5min
            except Exception:
                print("failure for 5min, trying again.")
                try:
                    candle_volume_5min = q["total_volume"] - start_vol_5min
                except Exception:
                    print("failed")

            #adds data to lists
            temp_FiveMinCandle.append(candle_open_5min)
            temp_FiveMinCandle.append(candle_high_5min)
            temp_FiveMinCandle.append(candle_low_5min)
            temp_FiveMinCandle.append(candle_close_5min)
            temp_FiveMinCandle.append(candle_volume_5min)


            all_FiveMinCandles.append(temp_FiveMinCandle) #1open, 2high, 3low, 4close, 5volume
            print(all_FiveMinCandles)

            if tee: tee.put_candle(ticker, temp_FiveMinCandle, "5min", None)

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
                    fiftyTwo_week_high = q["high_52wk"]
                    fiftyTwo_week_low = q["low_52wk"]
                    high52ratio = 1 - stockPrice / fiftyTwo_week_high
                    low52ratio = 1 - fiftyTwo_week_low / stockPrice
                    stock_data = arrange_from_live(all_FiveMinCandles, all_OneMinCandles, floatShares,high52ratio,low52ratio, ratio_premarketHigh, ratio_premarketLow, daily_high_ratio, daily_low_ratio)
                    write_to_csv_for_prediction(stock_data)
                except Exception:
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
                    notify("Todd just bought " + str(ticker) + " for " + str(buy_price) + " at " + str(time.time()))

                    #organizes the buy amount
                    buy_quantity = round(trade_amount/stockPrice)
                    buy_remainder = 2 - (buy_quantity % 2)
                    buy_quantity += buy_remainder

                    #places trade
                    order_id = schwab_trader.buy_market(ticker, buy_quantity)


        #All following code pertains to the sell mechanism - includes a trailing stop loss of 1%
        if bought == False:
            stop_loss_set = False
            begin_stoploss_trail = False
            stop_loss_trail = -2.5
        if bought == True:
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

            notify("Todd just sold " + str(ticker) + " for a percent change of " + str(percent_change) + " SOLD 2")

            #sell code
            sell_quantity = buy_quantity/2
            schwab_trader.sell_market(ticker, sell_quantity)

        if sold2 == True and noted2 == False:
            print("profit equals: ", profit)
            filename = 'trade_history.txt'
            time_since_open = (int(time.time()) - Market_open) / 60
            to_write = str(ticker) + " " + str(profit) + "Time Bought: " + str(buyTime_since_open) +  " Time Sold: " + str(time_since_bought) + " Percent: " + str(percent_change) + " SOLD 3" + "\n"
            with open(filename, "a") as file:
                file.write(to_write)
            noted2 = True

            notify("Todd just sold " + str(ticker) + " for a percent change of " + str(percent_change) + " SOLD 3")

            #sell code
            sell_quantity = buy_quantity/2
            schwab_trader.sell_market(ticker, sell_quantity)

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
        except Exception:
            try:
                feature = (((feature_close - feature_low) - (feature_high - feature_close))/feature_open * 1000) * (feature_vol/float_volume*100)
            except Exception:
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
    fiveMinData = pd.concat([fiveMinData, new_row.to_frame().T], ignore_index=True)
    y_list = [0]
    fiveMinData["y_list"] = y_list
    return fiveMinData


def write_to_csv_for_prediction(fiveMinData):
    filename = f'DataToPredict{instance_id}.csv'
    with open(filename, 'w') as file:
        fiveMinData.to_csv(file)


if __name__ == '__main__':
    main()
