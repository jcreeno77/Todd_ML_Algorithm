import requests
import pandas as pd
import json
import os
import time
from TDAmConfig import client_id
import addDataFromLog as dataLog
import convertCSVyToSigmoid as ccsvts
import BalanceDataSigmoid as BDS
import upload_csv_toAWS_s3 as U_s3



def main():
    print("Running Main")
    select_input()
    #runs convert to sigmoid automatically
    ccsvts.main()
    #runs balance data automatically
    BDS.main()
    #uploads files
    U_s3.main()
    



def select_input():
    selection = input("Input manually or from log? Input 1 for manual and 0 for log: ")
    if selection == "1":
        interactive_data_adding()
        #adds two week data automatically
        data_adding_from_log("trained_2weeks.txt", 'past_two_weekstrainData.csv')
    if selection == "0":
        Train_or_crossVal = input("Create Log from training data from last 2 weeks with 0 and CrossValidation data with 1: ")
        if Train_or_crossVal == "0":
            data_adding_from_log("trained_2weeks.txt", 'past_two_weekstrainData.csv')
        elif Train_or_crossVal == "1":
            data_adding_from_log("Cross_val_data_log.txt", 'crossValData.csv')

    #runs convertCSVyToSigmoid automatically
    if selection == "0":
        ccsvts.main()


#the method for inputting data through log, in case I want to input data that way or if I change anything in the data
def interactive_data_adding():
    same_day = False
    repeat = True
    difference = 9000

    while repeat:
        try:
            ticker = str(input("Enter a Ticker: " ))
            if same_day == False:
                startDate = str(int(input("Enter a start Date in Epochs: " )) * 1000)
                endDate = str(int(startDate) + difference)
                

            frequencyFive = '5'
            frequencyOne = '1'
            frequencyType = 'minute'

            #to be set later by input
            good_news = 1
            earnings = 0
            fundamentals = arrange_fundamentals(ticker)        #returns dict of "tickers" : [high52week, low52week, and floeatshares] in a list

            #These return data from TD Ameritrade for a start date and end date: one for five min candles and one for 1 min candles
            stockHist5min = requestsTDAm_Hist(ticker, frequencyFive, frequencyType, startDate, endDate, 'False')  
            stockHist5min_ext_hrs = requestsTDAm_Hist(ticker, frequencyFive, frequencyType, startDate, endDate, 'True')
            stockHist1min = requestsTDAm_Hist(ticker, frequencyOne, frequencyType,startDate,endDate, 'False')

            #This function arranges the data into trainable features for the ML
            trainable_data = arrange_Hist(stockHist5min, stockHist1min, stockHist5min_ext_hrs, fundamentals, good_news, earnings)
            
            filename = 'trainableData.csv'
            #This function adds the arranged data to a csv housing all training data
            add_to_training_data(trainable_data, ticker, filename,startDate, False, True)

            #The following code checks if more data should be entered
            repeat_check = input("Would you like to repeat? 1 for yes, 0 for no: ")
            if repeat_check == '1':
                repeat = True
                same_day_check = input("Would you like to input stock data for the same day? 1 for yes, 0 for no: ")
                if same_day_check == '1':
                    same_day = True
                else:
                    same_day = False
            else:
                repeat = False
        except:
            print("Problem... could be a few reasons... Usually no float identified.")
            

def data_adding_from_log(logFile, filename):

    frequencyFive = '5'
    frequencyOne = '1'
    frequencyType = 'minute'
    good_news = 1
    earnings = 0 
    difference = 9000
    

    x, y = dataLog.get_info_from_log(logFile)
    i = -1
    for ticker, epochTime in zip(x,y):
        i = i + 1
        if i % 15 == 0 and i != 0:
            print("Pausing for 30 seconds")
            time.sleep(30)

        fundamentals = arrange_fundamentals(ticker)
        startDate = epochTime
        endDate = str(int(startDate) + difference)

        #get stock history details
        stockHist5min = requestsTDAm_Hist(ticker, frequencyFive, frequencyType, startDate, endDate, 'False')  
        stockHist5min_ext_hrs = requestsTDAm_Hist(ticker, frequencyFive, frequencyType, startDate, endDate, 'True')
        stockHist1min = requestsTDAm_Hist(ticker, frequencyOne, frequencyType,startDate,endDate, 'False')

        #Arrange that history into the features
        trainable_data = arrange_Hist(stockHist5min, stockHist1min, stockHist5min_ext_hrs, fundamentals, good_news, earnings)

        #adds to training data
        if i <= 1:
            add_to_training_data(trainable_data, ticker, filename,startDate, True, False)
        else:
            add_to_training_data(trainable_data, ticker, filename,startDate, False, False)

 


#function arranges fundamental info into usable ML inputs for the csv append to add it
def arrange_fundamentals(ticker):
    #the return dictionary
    #iterates over the ticker list
    print(ticker)
    #uses the quotes and fundamentals request functions to return a dictionary
    fundamentals_df = requestsTDAm_Fundamentals(ticker)
    #sifts through dictionary for important items
    floatShares = fundamentals_df[ticker]['fundamental']['marketCapFloat'] * 1000000
    high52week = fundamentals_df[ticker]['fundamental']['high52']
    low52week = fundamentals_df[ticker]['fundamental']['low52']

    
    ticker_fund_info = [high52week, low52week, floatShares]

    return ticker_fund_info


#returns dataframe to be added
def arrange_Hist(fiveMinDataFrame, oneMinDataFrame, ext_hrs_data, fundamentals, good_news, earnings):

    float_volume = fundamentals[2]
    fiveMinData = pd.DataFrame({"goodNews": [],"Earnings": [],"52highRatio": [], "52lowRatio": [],"preMarket_high_ratio":[],"premarket_low_ratio":[],"ratioToDailyHigh":[],"ratioToDailyLow":[],"5min8":[],"5min8Unweighted":[],"5min8Squar":[],"5min7":[],"5min7Unweighted":[],"5min7Squar":[],"5min6":[],"5min6Unweighted":[],"5min6Squar":[],"5min5":[],"5min5Unweighted":[],"5min5Squar":[],"5min4":[],"5min4Unweighted":[],"5min4Squar":[],"5min3":[],"5min3Unweighted":[],"5min3Squar":[],"5min2":[],"5min2Unweighted":[],"5min2Squar":[],"5min1":[],"5min1Unweighted":[],"5min1Squar":[],"1min5":[],"1min5Unweighted":[],"1min5Squar":[],"1min4":[],"1min4Unweighted":[],"1min4Squar":[],"1min3":[],"1min3Unweighted":[],"1min3Squar":[],"1min2":[],"1min2Unweighted":[],"1min2Squar":[],"1min1":[],"1min1Unweighted":[],"1min1Squar":[]})
    #fiveMinData = pd.DataFrame({"52highRatio": [], "52lowRatio": [], "5min8":[],"5min7":[],"5min6":[],"5min5":[],"5min4":[],"5min3":[],"5min2":[],"5min1":[]}) #In case 1 minute is bad
    
    y_list = []
    #features_squared = []
    data_size_5min = fiveMinDataFrame.shape[0]
    data_size_1min = oneMinDataFrame.shape[0]
    
    data_isgood = True
    #calculates ratio for 5 to 1 minute candles so that the data stays relatively in-time

    five_to_one_minRatio = data_size_1min / data_size_5min

    ###### This is all to get the premarket information ####

    #These get the range for the premarket candlesticks
    premarket_limit = -1
    for item in ext_hrs_data['candles']:
        premarket_limit = premarket_limit + 1
        
        if item['volume'] == fiveMinDataFrame['candles'][0]['volume']:
            break

    #this is for the high of premarket
    premarket_high = 0
    
    for item in range(0,premarket_limit):
        candle_high = ext_hrs_data['candles'][item]['high']
        if candle_high > premarket_high:
            premarket_high = candle_high

    #this is for the low
    premarket_low = premarket_high
    for item in range(0,premarket_limit):
        candle_low = ext_hrs_data['candles'][item]['low']
        if candle_low < premarket_low:
            premarket_low = candle_low


    stockOpen = fiveMinDataFrame['candles'][0]['high']
    
    #These are the values to append
    try:
        ratio_premarketHigh = 1 - stockOpen/premarket_high
        ratio_premarketLow = 1 - premarket_low/stockOpen
    except:
        print("Ratio for premarket data is trying to divide by zero. As in it cannot request premarket data.")
        ratio_premarketHigh = None
        ratio_premarketLow = None





    for i in range(8,round(data_size_5min/2)+10):        #This is for the 5minute candle calculation
        fiveMinFeats = []
        #fiveMinFeats.append(fundamentals[0])
        #fiveMinFeats.append(fundamentals[1])
        
        for f in range(i-8,i):              #To get X five minute
            row = fiveMinDataFrame['candles'][f]
            rowOpen = row['open']
            rowClose = row['close']
            rowHigh = row['high']
            rowLow = row['low']
            rowVolume = row['volume']
            
            feature = (((rowClose - rowLow) - (rowHigh - rowClose))/rowOpen * 1000) * (rowVolume/float_volume*100)
            feature_unweighted = (((rowClose - rowLow) - (rowHigh - rowClose))/rowOpen * 100)
            feature_squared = feature**2
        
        
            fiveMinFeats.append(feature)
            fiveMinFeats.append(feature_unweighted)
            fiveMinFeats.append(feature_squared)       
            
        
        #obtains 1 minute candle features
        for o in range(round(i*five_to_one_minRatio) + round(five_to_one_minRatio) -5, round(i*five_to_one_minRatio) + round(five_to_one_minRatio)):
            rowOneMin = oneMinDataFrame['candles'][o]
            rowOpen = rowOneMin['open']
            rowClose = rowOneMin['close']
            rowHigh = rowOneMin['high']
            rowLow = rowOneMin['low']
            rowVolume = rowOneMin['volume']
            feature = (((rowClose - rowLow) - (rowHigh - rowClose))/rowOpen * 1000) * (rowVolume/float_volume*100)
            feature_unweighted = (((rowClose - rowLow) - (rowHigh - rowClose))/rowOpen * 100)
            feature_squared = feature**2

            fiveMinFeats.append(feature)
            fiveMinFeats.append(feature_unweighted)
            fiveMinFeats.append(feature_squared)

        daily_high = 0
        for candle in range(0,i+premarket_limit):
            candle_high = ext_hrs_data['candles'][candle]['high']
            if candle_high > daily_high:
                daily_high = candle_high

        daily_low = daily_high

        for candle in range(0,i+premarket_limit):
            candle_low = ext_hrs_data['candles'][candle]['low']
            if candle_low < daily_low:
                daily_low = candle_low

        daily_high_ratio = 1 - fiveMinDataFrame['candles'][i]['open']/daily_high
        daily_low_ratio = 1 - daily_low/fiveMinDataFrame['candles'][i]['open']


        #Add persistent / fundamental features here
        high52ratio = 1 - fiveMinDataFrame['candles'][i]['close']/fundamentals[0]
        low52ratio = 1 - fundamentals[1]/fiveMinDataFrame['candles'][i]['close']

        if data_isgood == True:
        #bad way of doing this but whatever
            fiveMinFeats.insert(0,daily_low_ratio)
            fiveMinFeats.insert(0,daily_high_ratio)
            fiveMinFeats.insert(0,ratio_premarketLow)
            fiveMinFeats.insert(0,ratio_premarketHigh)
            fiveMinFeats.insert(0,low52ratio)
            fiveMinFeats.insert(0,high52ratio)
            fiveMinFeats.insert(0,earnings)
            fiveMinFeats.insert(0,good_news)
            new_row = pd.Series(fiveMinFeats, index = fiveMinData.columns)
            fiveMinData = fiveMinData.append(new_row, ignore_index = True)  #append data rows

        
        
        y = 0
        sold = False
        future_candle_count = 4
        
        sell_limit = 5
        sell_stop = -5
 
        for y in range(i+1,i+future_candle_count):      #finds y
            base_price_candle = fiveMinDataFrame['candles'][i]
            future_price_candle = fiveMinDataFrame['candles'][y]
            base_price_close = base_price_candle['close']
            future_price_close = future_price_candle['close']
            future_price_high = future_price_candle['high']
            future_price_low = future_price_candle['low']
            percent_gain = (future_price_close - base_price_close)/base_price_close * 100
            
            #possible future use
            #percent_gain_high = (future_price_high - base_price_close)/base_price_close * 100
            #percent_gain_low = (future_price_low - base_price_close)/base_price_close * 100
            
            
            
            if percent_gain >= sell_limit:
                y = 1
                sold = True
                break
            elif percent_gain <= sell_stop:
                y = 0
                sold = True
                break
        
        if sold == False:
            base_price_candle = fiveMinDataFrame['candles'][i]
            future_price_candle = fiveMinDataFrame['candles'][i+future_candle_count-1]
            base_price_close = base_price_candle['close']
            future_price_close = future_price_candle['close']
            percent_gain = (future_price_close - base_price_close)/base_price_close * 100
            y = (abs(sell_stop) + percent_gain) / (sell_limit + abs(sell_stop)) 
        
        
        
        y_list.append(str(y))
    fiveMinData["y_list"] = y_list
    print(fiveMinData)
    return fiveMinData
      


#data requests (returns a dataframe for each item)
def requestsTDAm_Hist(ticker, frequency, frequencyType, startDate, endDate, extHours):  #eventually add start and end date info

    endpointHist = r"https://api.tdameritrade.com/v1/marketdata/{}/pricehistory".format(ticker)

    #define payload
    payloadHist = {'apikey': client_id, 'frequencyType': frequencyType, 'frequency': frequency, 'startDate': startDate,'endDate': endDate, 'needExtendedHoursData' : extHours} #need to be able to set time period
    
    #make a request
    contentHist = requests.get(url = endpointHist, params = payloadHist)

    dataHist = contentHist.json()
    dataHist = json.dumps(dataHist)
    stockData = pd.read_json(dataHist)
    
    #gets nodes
    return stockData

def requestsTDAm_Fundamentals(tickerList):
    #set endpoint url for request
    endpointFund = r"https://api.tdameritrade.com/v1/instruments"
    
    #define payload
    payloadFund = {'apikey':client_id, 'symbol': tickerList,'projection': 'fundamental'}
    

    #make request
    contentFund = requests.get(url = endpointFund, params = payloadFund)
    

    #convert json request return to panda dataframe
    dataFund = contentFund.json()
    dataFund = json.dumps(dataFund)

    stockFundamentals = pd.read_json(dataFund)    #this is the name of the stock fundamentals data frame

    return stockFundamentals
    
def requestsTDAm_liveQuote(tickerList):
    endpointQuote = r"https://api.tdameritrade.com/v1/marketdata/quotes"

    payloadQuote = {'apikey': client_id,'symbol': tickerList}

    contentQuote = requests.get(url = endpointQuote, params = payloadQuote)

    dataQuote = contentQuote.json()
    dataQuote = json.dumps(dataQuote)

    stockQuotes = pd.read_json(dataQuote)   #this is the stock quote dataframe

    return stockQuotes



#appends data to csv file
def add_to_training_data(fiveMinDataFrame, ticker, filename, startDate, overwrite, addToLog): #receives pandas dataframe and adds it to the training data csv
    if overwrite == True:
        with open(filename, 'w') as file:
            fiveMinDataFrame.to_csv(file)
    if overwrite == False:
        with open(filename, 'a') as file:
            fiveMinDataFrame.to_csv(file, header= False)

    #filename = 'crossValData.csv'
    #with open(filename, 'a') as file:
    #    fiveMinDataFrame.to_csv(file, header=False)
    #adds to log and 2week log
    if addToLog == True:
        logname = 'trained.txt'
        with open(logname, 'a') as file:
            file.write(ticker + " " + startDate + "\n")


        #Adds to 2week, and removes previous line
        logname2week = open("trained_2weeks.txt", "r")
        lines = logname2week.readlines()
        logname2week.close()
        #deletes line from line list
        del lines[1]
        #appends new line to line list
        lines.append(ticker + " " + startDate + "\n")
        print(lines)
        #writes line list to file
        logname2week = open("trained_2weeks.txt", "w+")
        for line in lines:
            logname2week.write(line)
        logname2week.close()



#Run code
if __name__ == '__main__':
    main()