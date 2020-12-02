import functools
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from functools import partial
from tensorflow.keras import layers
from keras.layers.advanced_activations import LeakyReLU


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels
    
def get_dataset(file_path, **kwargs):
    BATCH_SIZE = 1806 #len(train_filepath_inuse)
    LABEL_COLUMN = 'y_list'
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=BATCH_SIZE, 
        na_value="?",
        label_name=LABEL_COLUMN,
        num_epochs=1,
        ignore_errors=True, 
        **kwargs)
    return dataset
    
def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data-mean)/std

def Todd_predict():
    test_filepath = "DataToPredict1.csv"
    train_filepath_sigmoid = "trainableDataSigmoid_balanced_test.csv"
    train_filepath_sigmoid2 = "past_two_weekstrainDataSigmoid_balanced.csv"

    test_filepath_inuse = test_filepath
    train_filepath_inuse = train_filepath_sigmoid
    train_filepath_inuse2 = train_filepath_sigmoid2

    raw_test_data = get_dataset(test_filepath_inuse)
    raw_train_data = get_dataset(train_filepath_inuse)



    SELECT_COLUMNS = ["52highRatio", "52lowRatio","preMarket_high_ratio","premarket_low_ratio","ratioToDailyHigh","ratioToDailyLow","5min8","5min8Unweighted","5min8Squar","5min7","5min7Unweighted","5min7Squar","5min6","5min6Unweighted","5min6Squar","5min5","5min5Unweighted","5min5Squar","5min4","5min4Unweighted","5min4Squar","5min3","5min3Unweighted","5min3Squar","5min2","5min2Unweighted","5min2Squar","5min1","5min1Unweighted","5min1Squar","1min5","1min5Unweighted","1min4","1min4Unweighted","1min3","1min3Unweighted","1min2","1min2Unweighted","1min1","1min1Unweighted"]

    packed_train_data = raw_train_data.map(
        PackNumericFeatures(SELECT_COLUMNS))
    packed_test_data = raw_test_data.map(
        PackNumericFeatures(SELECT_COLUMNS))


    NUMERIC_FEATURES = ["52highRatio", "52lowRatio","preMarket_high_ratio","premarket_low_ratio","ratioToDailyHigh","ratioToDailyLow","5min8","5min8Unweighted","5min8Squar","5min7","5min7Unweighted","5min7Squar","5min6","5min6Unweighted","5min6Squar","5min5","5min5Unweighted","5min5Squar","5min4","5min4Unweighted","5min4Squar","5min3","5min3Unweighted","5min3Squar","5min2","5min2Unweighted","5min2Squar","5min1","5min1Unweighted","5min1Squar","1min5","1min5Unweighted","1min4","1min4Unweighted","1min3","1min3Unweighted","1min2","1min2Unweighted","1min1","1min1Unweighted"]


    desc = pd.read_csv(train_filepath_inuse)[NUMERIC_FEATURES].describe()
    desc2 = pd.read_csv(train_filepath_inuse2)[NUMERIC_FEATURES].describe()
    MEAN = np.array(desc.T['mean'])
    STD = np.array(desc.T['std'])

    MEAN2 = np.array(desc2.T['mean'])
    STD2 = np.array(desc2.T['std'])

    normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)
    normalizer2 = functools.partial(normalize_numeric_data, mean=MEAN2, std=STD2)


    numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
    numeric_column2 = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer2, shape=[len(NUMERIC_FEATURES)])

    numeric_columns = [numeric_column]
    numeric_columns2 = [numeric_column2]

    numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
    numeric_layer2 = tf.keras.layers.DenseFeatures(numeric_columns2)

    preprocessing_layer = tf.keras.layers.DenseFeatures(numeric_columns)

    regularization_rate = .03

    #Sets learning rate for "adam"
    tf.keras.optimizers.Adam(
        learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')



    model_2wk = tf.keras.Sequential([
        numeric_layer2,
        tf.keras.layers.Dense(128, activation="tanh"),
        tf.keras.layers.Dense(128, activation=partial(tf.nn.leaky_relu, alpha=0.05)),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model_tanh = tf.keras.Sequential([
        numeric_layer,
        tf.keras.layers.Dense(128, activation="tanh"),
        tf.keras.layers.Dense(128, activation=partial(tf.nn.leaky_relu, alpha=0.05)),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model_2wk.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='Adam',
        metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    model_tanh.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='Adam',
        metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    test_data = packed_test_data


    model_2wk.load_weights('current_weights_2wk')
    model_tanh.load_weights('current_weights')

    predictions = model_2wk.predict_classes(test_data), model_tanh.predict_classes(test_data)

    print("PREDICTIONS")
    print(predictions)

    return predictions
    
    
if __name__ == '__main__':
    Todd_predict()