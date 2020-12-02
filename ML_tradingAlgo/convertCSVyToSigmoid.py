import pandas as pd

def main():
    train_or_crossVal = input("FOR CCSVTS: 0 for training data 1 for crossval data: ")
    train_or_crossVal = int(train_or_crossVal)
    if train_or_crossVal == 0:
        importData = pd.read_csv("trainableData.csv")
        importDataWeekly = pd.read_csv("past_two_weekstrainData.csv")
    else:
        importData = pd.read_csv("crossValData.csv")
        importDataWeekly = pd.read_csv("crossValData.csv")
    #print(importData)
    y_list_sigmoid = []
    y_list_weekly_sigmoid = []

    for i in importData['y_list']:
        if i >= .7:
            y_list_sigmoid.append(1)
        else:
            y_list_sigmoid.append(0)

    
    for i in importDataWeekly['y_list']:
        if i >= .7:
            y_list_weekly_sigmoid.append(1)
        else:
            y_list_weekly_sigmoid.append(0)

    #print(y_list_sigmoid)

    SigmoidData = importData.drop(columns=['y_list'])
    SigmoidData = SigmoidData.drop(SigmoidData.columns[0], axis=1)
    SigmoidData["y_list"] = y_list_sigmoid

    SigmoidData_weekly = importDataWeekly.drop(columns=['y_list'])
    SigmoidData_weekly = SigmoidData_weekly.drop(SigmoidData_weekly.columns[0], axis=1)
    SigmoidData_weekly["y_list"] = y_list_weekly_sigmoid

    print(SigmoidData)
    print(SigmoidData_weekly)

    if train_or_crossVal == 0:
        filename = 'trainableDataSigmoid.csv'
        filename_weekly = 'past_two_weekstrainDataSigmoid.csv'
    else:
        filename = 'crossValSigmoid.csv'
    
    add_to_training_data(SigmoidData, filename)
    add_to_training_data(SigmoidData_weekly, filename_weekly)
        



#appends data to csv file
def add_to_training_data(pandasDataFrame, filename): #receives pandas dataframe and adds it to the training data csv
    #filename = 'trainableDataSigmoid.csv'
    with open(filename, 'w') as file:
        pandasDataFrame.to_csv(file)



if __name__ == '__main__':
    main()