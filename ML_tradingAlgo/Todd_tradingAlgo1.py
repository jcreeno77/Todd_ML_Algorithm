import functools
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class TradingModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.05),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def normalize_numeric_data(data, mean, std):
    return (data - mean) / std


def Todd_predict():
    test_filepath = "DataToPredict1.csv"
    train_filepath_sigmoid = "trainableDataSigmoid_balanced_test.csv"
    train_filepath_sigmoid2 = "past_two_weekstrainDataSigmoid_balanced.csv"

    train_filepath_inuse = train_filepath_sigmoid
    train_filepath_inuse2 = train_filepath_sigmoid2

    SELECT_COLUMNS = ["52highRatio", "52lowRatio","preMarket_high_ratio","premarket_low_ratio","ratioToDailyHigh","ratioToDailyLow","5min8","5min8Unweighted","5min8Squar","5min7","5min7Unweighted","5min7Squar","5min6","5min6Unweighted","5min6Squar","5min5","5min5Unweighted","5min5Squar","5min4","5min4Unweighted","5min4Squar","5min3","5min3Unweighted","5min3Squar","5min2","5min2Unweighted","5min2Squar","5min1","5min1Unweighted","5min1Squar","1min5","1min5Unweighted","1min4","1min4Unweighted","1min3","1min3Unweighted","1min2","1min2Unweighted","1min1","1min1Unweighted"]

    NUMERIC_FEATURES = SELECT_COLUMNS

    # Compute mean/std from training data
    desc = pd.read_csv(train_filepath_inuse)[NUMERIC_FEATURES].describe()
    desc2 = pd.read_csv(train_filepath_inuse2)[NUMERIC_FEATURES].describe()
    MEAN = np.array(desc.T['mean'])
    STD = np.array(desc.T['std'])
    MEAN2 = np.array(desc2.T['mean'])
    STD2 = np.array(desc2.T['std'])

    # Load test data
    test_data = pd.read_csv(test_filepath)
    test_features = test_data[NUMERIC_FEATURES].values.astype(np.float32)

    # Normalize
    test_norm = normalize_numeric_data(test_features, MEAN, STD)
    test_norm2 = normalize_numeric_data(test_features, MEAN2, STD2)

    # Build models
    input_size = len(NUMERIC_FEATURES)
    model_tanh = TradingModel(input_size)
    model_2wk = TradingModel(input_size)

    # Load weights (NOTE: existing TF weights are NOT compatible — model needs retraining)
    model_tanh.load_state_dict(torch.load('current_weights', weights_only=True))
    model_2wk.load_state_dict(torch.load('current_weights_2wk', weights_only=True))

    model_tanh.eval()
    model_2wk.eval()

    # Predict
    with torch.no_grad():
        pred_tanh = (model_tanh(torch.tensor(test_norm)) > 0.5).int()
        pred_2wk = (model_2wk(torch.tensor(test_norm2)) > 0.5).int()

    predictions = pred_2wk, pred_tanh

    print("PREDICTIONS")
    print(predictions)

    return predictions


if __name__ == '__main__':
    Todd_predict()
