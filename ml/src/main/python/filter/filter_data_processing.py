import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
from tensorflow.keras import layers
import ml.src.main.python.utils.tf_utils as tf_utils
from ml.src.main.python.utils.tf_utils import pretty_print
from ml.src.main.python.utils.tf_utils import read_csv

# Load data
train_and_val_ds = read_csv(csv_path="ml/src/resources/filter_data/train_and_val_filter_data.csv", do_copy=True)
test_ds = read_csv(csv_path="ml/src/resources/filter_data/test_filter_data.csv", do_copy=True)
test_ds = tf_utils.get_unique_columns(test_ds, train_and_val_ds, keep=False)
pretty_print(train_and_val_ds.tail(), "Part of train and val data:")
pretty_print(test_ds.tail(), "Part of test data:")

# Shuffle values and delete
train_and_val_ds = train_and_val_ds.sample(frac=1, random_state=0).dropna()
test_ds = test_ds.sample(frac=1, random_state=0).dropna()

# Inspect the data
train_stats = train_and_val_ds.describe()
train_stats.pop('time')
train_stats = train_stats.transpose()
pretty_print(train_stats, "Train_stats")

# Split features from labels
train_features = train_and_val_ds.copy()
test_features = test_ds.copy()
train_labels = train_features.pop('time')
test_labels = test_features.pop('time')


# Normalize data
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_features)
normed_test_data = norm(test_features)


# Build model
def build_model():
    dnn_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_features.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    dnn_model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.RMSprop(0.001),
                      metrics=['mae', 'mse'])
    return dnn_model


model = build_model()
model.summary()

# Fit model
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

pretty_print(test_features, "Test features")
pretty_print(normed_test_data, "Normalized test features")
test_predictions = model.predict(normed_test_data).flatten()
pretty_print(test_predictions, "Predict result (test features)", line_length=50)

tfjs.converters.save_keras_model(model, "src/main/resources/static/model/filter")
