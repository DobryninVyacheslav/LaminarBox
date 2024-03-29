import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ml.src.main.python.utils.tf_utils import pretty_print

import tensorflow as tf
import tensorflowjs as tfjs

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

dataset_path = "ml/src/resources/filter_data/train_and_val_filter_data.csv"
raw_dataset = pd.read_csv(dataset_path, na_values="?", comment='\t',
                          sep=",", skipinitialspace=True)
dataset = raw_dataset.copy()
dataset.isna().sum()
dataset = dataset.dropna()

train_dataset = dataset.sample(frac=1, random_state=0)
pretty_print(train_dataset, "Train dataset")
test_dataset = dataset.drop(dataset.sample(frac=0.8, random_state=0).index)
pretty_print(test_dataset, "Test dataset")

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('time')
test_labels = test_features.pop('time')

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


dnn_model = build_and_compile_model(normalizer)

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=1000)

test_predictions = dnn_model.predict(test_features).flatten()
pretty_print(test_features, "test_features")
pretty_print(test_predictions, "test_predictions")
