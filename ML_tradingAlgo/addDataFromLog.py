import re


def main():
    x, y = get_info_from_log("trained.txt")
    print(x, y)


def get_info_from_log(txt_file):

    #regex info
    ticker_regex = r"([A-Z])\w+"
    epoch_regex = r"([0-9])\w+"

    ticker_list = []
    epoch_time_list = []

    #reading the file
    log = txt_file
    with open(log, 'r') as file:
        for line in file:
            x = re.search(ticker_regex,line)
            
            y = re.search(epoch_regex,line)

            if x != None:
                ticker_list.append(x.group(0))
            
            if y != None:
                
                epoch_time_list.append(str(y.group(0)))

    ticker_list.remove("Ticker")
    #print(ticker_list)
    #print(epoch_time_list)
    return ticker_list, epoch_time_list




#Run code
if __name__ == '__main__':
    main()