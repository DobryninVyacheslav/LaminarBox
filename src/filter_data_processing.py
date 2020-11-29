import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

dataset_path = "/home/slava/Common/PycharmProjects/LaminarBox/resources/filter_data.csv"
raw_dataset = pd.read_csv(dataset_path, na_values="?", comment='\t',
                          sep=",", skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.tail())
print('=====================')
dataset = dataset.dropna()
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_stats = train_dataset.describe()
train_stats.pop('time')
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('time')
test_labels = test_dataset.pop('time')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()
print(model.summary())
print('=====================')
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)
print('=====================')


# Выведем прогресс обучения в виде точек после каждой завершенной эпохи
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


EPOCHS = 1000

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

# plot_history(history)
#
# loss, mae, mse = model.evaluate(normed_train_data, train_labels, verbose=2)
#
# print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# test_predictions = model.predict(normed_test_data).flatten()
# print(normed_test_data)
# print('===================')
# print(test_predictions)

# plt.scatter(test_labels, test_predictions)
# plt.xlabel('True Values [MPG]')
# plt.ylabel('Predictions [MPG]')
# plt.axis('equal')
# plt.axis('square')
# plt.xlim([0, plt.xlim()[1]])
# plt.ylim([0, plt.ylim()[1]])
# _ = plt.plot([-100, 100], [-100, 100])
# plt.show()

data = {'glass': [0.0],
        'air': [0.0],
        'pressure': [250]}
test_df = pd.DataFrame(data)
test_predictions = model.predict(norm(test_df)).flatten()
print(test_df)
print('===================')
print(test_predictions)
