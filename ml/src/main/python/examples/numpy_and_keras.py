import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History

history = History()
seed = 7
np.random.seed(seed)

dataset_path = keras.utils.get_file("iris.data",
                                    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
dataset = np.loadtxt(dataset_path,
                     delimiter=",",
                     skiprows=1)

X = dataset[:, 0:(dataset.shape[1] - 1)]
# Y = dataset[:, dataset.shape[1] - 1]
print(X)
print("+++++++++++++++++++++")
# print(Y)

# create model
# model = Sequential()
# model.add(Dense(100, input_dim=(dataset.shape[1] - 1), activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # Fit the model
# model.fit(X, Y, validation_data=(X, Y), epochs=20, batch_size=128)
