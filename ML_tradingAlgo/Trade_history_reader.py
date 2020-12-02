
    #Steady Scanner TickerConv v1.0.0
	#9/11/2020
import pandas as pd
import numpy as np
import re
import os

all_trade_details = []
sum_list = []

filepath = "trade_history_test.txt"
percent_change_search = r"([,-9]+)\w+"
sell_type_search = r"([0-9]+)"


file = open(filepath,"r+") #gets filepath


for i, line in enumerate(file):   #goes through .txt file and gather necessary info, then arranges it into nested lists
    single_trade_details = []
    
    if re.search(percent_change_search, line):
        percent_result = re.findall(percent_change_search, line)
        sell_type_result = re.findall(sell_type_search, line)
        #stock = result.group(0)
        
        percent = percent_result[-1]
        sell_type = sell_type_result[-1]
        
        single_trade_details.append(float(percent))
        single_trade_details.append(int(sell_type))
        all_trade_details.append(single_trade_details)
        
        sum_list.append(float(percent))
file.close()

#parses lists into seperate lists pertaining to each sell item
sell_1 = []
sell_2 = []
sell_3 = []

for trade_details in all_trade_details:
    sell_type = int(trade_details[1])
    if sell_type == 1:
        sell_1.append(float(trade_details[0]))
    elif sell_type == 2:
        sell_2.append(float(trade_details[0]))
    else:
        sell_3.append(float(trade_details[0]))

print("Sell 1 results:")
print(sum(sell_1))
print("Sell 2 results:")
print(sum(sell_2))
print("Sell 3 results:")
print(sum(sell_3))

print("Totals:")
print(sum(sum_list))
input()