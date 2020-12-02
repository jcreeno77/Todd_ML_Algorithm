import pandas as pd

def main():

    #Balances data from total data and past2weeks of data
    data = pd.read_csv("trainableDataSigmoid.csv")
    data_past2weeks = pd.read_csv("past_two_weekstrainDataSigmoid.csv")

    amount_of_pos = 0
    amount_of_neg = 0
    for item in range(0,len(data['y_list'])):
        if data['y_list'][item] == 1:
            amount_of_pos += 1
        if data['y_list'][item] == 0:
            amount_of_neg += 1

    print("Total Length")
    print(len(data['y_list']))
    print("Positive length")
    print(amount_of_pos)
    print("negative length")
    print(amount_of_neg)

    amount_of_weekly_pos = 0
    amount_of_weekly_neg = 0
    for item in range(0,len(data_past2weeks['y_list'])):
        if data_past2weeks['y_list'][item] == 1:
            amount_of_weekly_pos += 1
        if data_past2weeks['y_list'][item] == 0:
            amount_of_weekly_neg += 1

    locations = []
    locations_weekly = []

    for item in range(0,len(data['y_list'])):
        if data['y_list'][item] == 1:
            locations.append(item)

    #weekly locations
    for item in range(0,len(data_past2weeks['y_list'])):
        if data_past2weeks['y_list'][item] == 1:
            locations_weekly.append(item)

    #for item in locations:
    positives = data.loc[locations, :]
    timesToMultiply = amount_of_neg/amount_of_pos

    #for item in weekly locations:
    positives_weekly = data_past2weeks.loc[locations_weekly, :]
    timesToMultiply_weekly = amount_of_weekly_neg/amount_of_weekly_pos
    
    print("trainableDataSigmoid_balanced_test.csv")
    print("For total data: ")
    for multiples in range(round(timesToMultiply-3)):
        data = data.append(positives, ignore_index = True)
    print("length of final list:")
    print(len(data['y_list']))
    print()

    #Past 2 weeks data
    print("past_two_weekstrainDataSigmoid_balanced.csv")
    print("For past 2 weeks data: ")
    print("Total Length")
    print(len(data_past2weeks['y_list']))
    for multiples in range(round(timesToMultiply_weekly-3)):
        data_past2weeks = data_past2weeks.append(positives_weekly, ignore_index = True)

    print("length of final list:")
    print(len(data_past2weeks['y_list']))


    filename = 'trainableDataSigmoid_balanced_test.csv'
    with open(filename, 'w') as file:
        data.to_csv(file)

    filename = 'past_two_weekstrainDataSigmoid_balanced.csv'
    with open(filename, 'w') as file:
        data_past2weeks.to_csv(file)


if __name__ == '__main__':
    main()