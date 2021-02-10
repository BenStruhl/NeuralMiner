import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import models
from keras.layers import Dense, Dropout
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

my_data = genfromtxt('bitcoin3.csv', delimiter=',')
# scaler = MinMaxScaler(feature_range=(0, 1))
# my_data = scaler.fit_transform(my_data)
# print(my_data[0][-1:])
PERCENT_HELD = 0.75
train_x = my_data[:,:5][:int(len(my_data)*PERCENT_HELD)]
train_y = my_data[:,5][:int(len(my_data)*PERCENT_HELD)]
test_x = my_data[:,:5][int(len(my_data)*PERCENT_HELD):]
test_y = my_data[:,5][int(len(my_data)*PERCENT_HELD):]

scaler = MinMaxScaler(feature_range=(0, 1))
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)

train_y= list(map(lambda x: [x], train_y))
test_y= list(map(lambda x: [x], test_y))
scaler2 = MinMaxScaler(feature_range=(0, 1))
train_y = scaler2.fit_transform(train_y)
test_y = scaler2.fit_transform(test_y)
train_y= np.array(list(map(lambda x: x[0], train_y)))
test_y= np.array(list(map(lambda x: x[0], test_y)))
# print(test_x[0])
# print(train_x[0].shape)


# Build neural network
model = models.Sequential()
model.add(Dense(100, activation="sigmoid", input_shape=(5,)))
model.add(Dense(100, activation="sigmoid", input_shape=(5,)))
model.add(Dense(100, activation="sigmoid", input_shape=(5,)))
model.add(Dense(1, activation="sigmoid"))

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanAbsoluteError(
    reduction="auto", name="mean_absolute_error"
),
              metrics=['accuracy'])

model.fit(train_x, train_y,
          batch_size=200,
          epochs=3,
          verbose=1,
          validation_data=(test_x, test_y))

score = model.evaluate(test_x, test_y, verbose=0)
test1 = scaler2.inverse_transform(np.array(model.predict(test_x)))
test_y= list(map(lambda x: [x], test_y))
output1 = scaler2.inverse_transform(np.array(test_y))

average = 0
for i in range(len(output1)):
    average += abs(output1[i][0] - test1[i][0])
print("offby in test: " + str(average / len(output1)))
print("percent of search space " + str((average / len(output1)) / (pow(2,31) - 1)))

print('Test loss:', score[0])
print('Test accuracy:', score[1])
