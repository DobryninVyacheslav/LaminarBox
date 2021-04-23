import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
from tensorflow.keras import layers
from ml.src.main.python.utils.tf_utils import pretty_print

# Load data
dataset_path = "ml/src/resources/filter_data.csv"
raw_dataset = pd.read_csv(dataset_path, na_values="?", comment='\t',
                          sep=",", skipinitialspace=True)
dataset = raw_dataset.copy()
pretty_print(dataset.tail(), "Part of init data:")

# Delete missing values
dataset = dataset.dropna()

# Split the data into train and test
train_dataset = dataset.sample(frac=1, random_state=0)
test_dataset = dataset.drop(dataset.sample(frac=0.8, random_state=0).index)

# Inspect the data
train_stats = train_dataset.describe()
train_stats.pop('time')
train_stats = train_stats.transpose()
pretty_print(train_stats, "Train_stats")

# Split features from labels
train_labels = train_dataset.pop('time')
test_labels = test_dataset.pop('time')


# Normalize data
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# Build model
def build_model():
    dnn_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    dnn_model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.RMSprop(0.001),
                      metrics=['mae', 'mse'])
    return dnn_model


model = build_model()
model.summary()

# Выведем прогресс обучения в виде точек после каждой завершенной эпохи
EPOCHS = 1000
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

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

data = {'glass': [0],
        'air': [1],
        'pressure': [300.0]}
test_df = pd.DataFrame(data)
print('===================')
print(norm(test_df))
print('===================')
test_predictions = model.predict(norm(test_df)).flatten()
print(test_df)
print('===================')
print(test_predictions)
print('===================')
print(test_dataset)
print('===================')
print(normed_test_data)
print('===================')
test_predictions = model.predict(normed_test_data).flatten()
print(test_predictions)
print('===================')

tfjs.converters.save_keras_model(model, "src/main/resources/static/model/filter")
