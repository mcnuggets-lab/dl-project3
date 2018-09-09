import os
import scipy
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge, TimeDistributed, Maximum, Lambda, Concatenate, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, MaxPooling1D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.optimizers import Adam
import keras.backend as K

NUM_CLASS = 2
# cheange your data folder root here
DATA_FOLDER = os.path.join(".", "Dataset_C")

imgs = []
file_ids = []
for root, dirnames, filenames in os.walk(DATA_FOLDER):
    for filename in filenames:
        file_ids.append(filename.split("_")[0])
        imgs.append(os.path.join(root, filename))

CROP_SIZE = 227
NUM_PATCHES = 270

output_array = np.zeros((len(imgs), NUM_PATCHES, CROP_SIZE, CROP_SIZE, 1))

for idx, fn in enumerate(imgs):
    counter = 0
    raw_img = scipy.misc.imread(fn, mode="L")

    for i in range(raw_img.shape[0]//CROP_SIZE):
        for j in range(raw_img.shape[1]//CROP_SIZE):
            cropped = raw_img[i*CROP_SIZE:i*CROP_SIZE+CROP_SIZE, j*CROP_SIZE:j*CROP_SIZE+CROP_SIZE]
            output_array[idx, counter, :, :, 0] = cropped
            counter += 1

# define shared layers
conv1 = Conv2D(64, (3, 3), strides=3, activation='relu', name='conv1_1')
conv2 = Conv2D(64, (3, 3), strides=3, activation='relu', name='conv1_2')
max_pool1 = MaxPooling2D((2, 2), strides=(2, 2))
flatten1 = Flatten()
dense1 = Dense(NUM_CLASS, name='dense_1', activation='sigmoid')
reshape1 = Reshape((NUM_CLASS, 1))

def conv_layer(X_input):
    """
    The convolution layer used for all instances, weights are all shared.
    :param X_input: an instance (slice of an image), shape (CROP_SIZE, CROP_SIZE, 1)
    :return: the convolution layer output, shape (NUM_CLASS, 1).
    """
    X = conv1(X_input)
    X = conv2(X)
    X = max_pool1(X)
    X = flatten1(X)
    X = dense1(X)
    X = reshape1(X)
    return X


# todo edge case, data image generator
X_inputs = [Input(shape=(CROP_SIZE, CROP_SIZE, 1), name="input_" + str(i)) for i in range(NUM_PATCHES)]
X_outputs = [conv_layer(X_inputs[i]) for i in range(NUM_PATCHES)]
Instance_Probs = concatenate(X_outputs)
X = Lambda(lambda x: K.logsumexp(x, axis=2))(Instance_Probs)
#X = Lambda(lambda x: K.sum(x, axis=2))(Instance_Probs)  # combine histograms, c.f. paper of Option 2
print(X.shape)
X = Dense(1, name="dense_output", activation="sigmoid")(X)
model = Model(inputs=X_inputs, outputs=X, name='MIL')

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])

print(model.output_shape)
model.summary()
#plot_model(model, "model.png")
model.fit({'input_' + str(i): output_array[:,i,:,:,:] for i in range(NUM_PATCHES)},
          np.array([0]*len(imgs)), epochs=2)

