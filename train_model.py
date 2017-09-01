import pickle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten

# Constants
x_train_file_name = "train_pairs_x.npy"
y_train_file_name = "train_pairs_y.npy"

x_test_file_name = "test_pairs_x.npy"
y_test_file_name = "test_pairs_y.npy"

batch_size = 100
num_classes = 2
num_epoch = 100

img_rows = 64
img_cols = 64

img_channels = 1


# Load images
x_train = np.load(x_train_file_name)
y_train = np.load(y_train_file_name)

x_test = np.load(x_test_file_name)
y_test = np.load(y_test_file_name)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Set up output data in categorical matrices

y_train = np_utils.to_categorical(y_train, num_classes)
y_test  = np_utils.to_categorical(y_test, num_classes)

# Build the network model

model = Sequential()

model.add(Convolution2D(32, 7, 7, input_shape=x_train.shape[1:]))
model.add(Activation("relu"))

# Had to add this because we had too many dimensions when we hit the below Dense network.
# We should experiment with adding more convolutional layers so that we can
# decrease the dimensionality without losing features.
model.add(MaxPooling2D(pool_size=(2,2))) 

model.add(Convolution2D(64, 5, 5))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(25536))
model.add(Activation("relu"))
model.add(Dense(num_classes))

model.add(Activation("softmax"))

# Compile the model and put data between 0 and 1

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(x_train.shape)
print(y_train.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape)
print(y_train.shape)

# Train the model

model.fit(x_train, y_train,
              batch_size=batch_size,
              nb_epoch=num_epoch,
              validation_data=(x_test, y_test),
              shuffle=True)

model.save("finished_model.hdf5")