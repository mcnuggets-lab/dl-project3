import os
import scipy
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, to_categorical
from keras.optimizers import Adam
import keras.backend as K


######################################
# define constants and hyperparameters
######################################

NUM_CLASS = 2

# constants for the training
USE_EM = True
NUM_EPOCHS = 20
BATCH_SIZE = 32

# the following are used only for the EM method
IMAGE_BASED_THRESHOLD = 75        # percentile of patches to trim, based on image
CLASS_BASED_THRESHOLD = [75, 25]  # percentile of patches to trim, based on class prob.
TISSUE_RATIO = 0.2                # if the amount of tissue in a patch is less than this ratio, we drop the patch
MIN_KEEP_PROB = 0                 # min prob to regard a patch as discriminative


######################################
# read-in data and labels
######################################

# TODO: read in data and image level labels
# change your data folder root here
DATA_FOLDER = os.path.join(".", "Dataset_C")

imgs_filenames = []
file_ids = []
for root, dirnames, filenames in os.walk(DATA_FOLDER):
    for filename in filenames:
        file_ids.append(filename.split("_")[0])
        imgs_filenames.append(os.path.join(root, filename))

MAX_WIDTH = 3328
MAX_HEIGHT = 4084
CROP_SIZE = 227
NUM_PATCH_WIDTH = MAX_WIDTH // CROP_SIZE + 1
NUM_PATCH_HEIGHT = MAX_HEIGHT // CROP_SIZE + 1
NUM_PATCHES = NUM_PATCH_WIDTH * NUM_PATCH_HEIGHT

imgs = []  # backup of all images for many croppings later
img_labels = []  # image level labels
#img_labels_onehot = []
train_data = []
train_label = []
discriminative_patches = []


def get_patch(img_id, ind_x, ind_y):
    """
    Return the cropped image patch from image with id: img_id and location index (ind_x, ind_y)
    :param img_id: ID of the imnage to be cropped
    :param ind_x: x index of the crop location
    :param ind_y: y index of the crop location
    :return: the cropped patch
    """
    cropped = imgs[img_id][ind_x*CROP_SIZE:ind_x*CROP_SIZE+CROP_SIZE, ind_y*CROP_SIZE:ind_y*CROP_SIZE+CROP_SIZE]
    if ind_x == imgs[img_id].shape[0]//CROP_SIZE or ind_y == imgs[img_id].shape[1]//CROP_SIZE:
        cropped = np.pad(cropped, ((0, CROP_SIZE-cropped.shape[0]), (0, CROP_SIZE-cropped.shape[1])), mode="constant")
    return cropped.reshape((CROP_SIZE, CROP_SIZE, 1))

def get_heatmap(ind):
    """
    Show the heatmap corresponding to the patches of the given image
    :param ind: index of the image to be shown
    :return: the heatmap which is P(H_i|X_i) of the paper
    """
    label = img_labels[ind]  # non-one-hot class label
    heatmap = np.zeros((imgs[ind].shape[0] // CROP_SIZE + 1,
                        imgs[ind].shape[1] // CROP_SIZE + 1))
    for patch in discriminative_patches[ind]:
        prob = model.predict(np.array([get_patch(ind, patch[0], patch[1])]))
        heatmap[patch] = prob[0][label]
    heatmap = gaussian_filter(heatmap, sigma=1)  # denoise using gaussian kernel
    return heatmap


for idx, fn in enumerate(imgs_filenames):

    raw_img = scipy.misc.imread(fn, mode="L") / 255  # divide 255 for normalization
    imgs.append(raw_img)
    img_label = 0 if idx < 8 else 1  # TODO: after reading the labels, refactor this line
    img_labels.append(img_label)

    discriminative_patch = []

    for i in range(raw_img.shape[0]//CROP_SIZE+1):
        for j in range(raw_img.shape[1]//CROP_SIZE+1):
            cropped = get_patch(idx, i, j)
            if np.sum(cropped > 0) >= TISSUE_RATIO * np.size(cropped):
                discriminative_patch.append((i, j))
                train_data.append(cropped)
                train_label.append(img_labels[idx])
    discriminative_patches.append(discriminative_patch)

train_data = np.array(train_data)
train_label = to_categorical(np.array(train_label))
img_labels = np.array(img_labels)
img_labels_onehot = to_categorical(img_labels)


######################################
# build patch-based model for the EM step
######################################

# X_input = Input(shape=(CROP_SIZE, CROP_SIZE, 1), name="input")
# X = Conv2D(64, (3, 3), strides=3, activation="relu", name="conv1_1")(X_input)
# X = MaxPooling2D((2, 2), strides=(2, 2))(X)
# X = Conv2D(64, (3, 3), strides=3, activation="relu", name="conv1_2")(X)
# X = MaxPooling2D((2, 2), strides=(2, 2))(X)
# X = Flatten()(X)
# X = Dense(2, name="dense_1", activation="softmax")(X)

# define patch-level CNN model for EM step. TODO: reconstruct a better CNN for training
X_input = Input(shape=(CROP_SIZE, CROP_SIZE, 1), name="input")
X = Conv2D(80, (7, 7), strides=2, activation="relu", name="conv1_1")(X_input)
X = BatchNormalization(name="batchnorm_1")(X)
X = MaxPooling2D((2, 2), strides=(2, 2))(X)
X = Conv2D(120, (5, 5), strides=2, activation="relu", name="conv1_2")(X)
X = BatchNormalization(name="batchnorm_2")(X)
X = MaxPooling2D((2, 2), strides=(2, 2))(X)
#X = Conv2D(160, (3, 3), strides=1, activation="relu", name="conv1_3")(X)
#X = Conv2D(200, (3, 3), strides=1, activation="relu", name="conv1_4")(X)
#X = MaxPooling2D((2, 2), strides=(2, 2))(X)
X = Flatten()(X)
X = Dense(320, activation="relu", name="dense_1")(X)
#X = Dropout(0.5, name="dropout_1")(X)
#X = Dense(320, activation="relu", name="dense_2")(X)
#X = Dropout(0.5, name="dropout_2")(X)
X = Dense(2, name="instance_output", activation="softmax")(X)

model = Model(inputs=X_input, outputs=X, name='patch_CNN')
adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
#plot_model(model, "model.png")

# TODO: fit in validation data for the model
if not USE_EM:
    model.fit(train_data, train_label, batch_size = BATCH_SIZE, epochs=NUM_EPOCHS)
else:
    for em_round in range(1, NUM_EPOCHS//2+1):
        print("EM round {}".format(em_round))

        # M-step: train the model for 2 epoches
        model.fit(train_data, train_label, batch_size = BATCH_SIZE, epochs=2)

        # E-step: construct discriminative patches for the next training cycle
        heatmaps = []
        threshold1 = []
        threshold2 = []
        for i in range(len(imgs)):
            heatmaps.append(get_heatmap(i))
            threshold1.append(np.percentile(heatmaps[i][heatmaps[i] >= MIN_KEEP_PROB], IMAGE_BASED_THRESHOLD))
        #heatmap_freqs = np.hstack([heatmaps[i].flatten() for i in range(len(imgs))])
        for c in range(NUM_CLASS):
            heatmap_freqs = np.hstack([heatmaps[i].flatten() for i in range(len(imgs)) if img_labels[i] == c])
            threshold2.append(np.percentile(heatmap_freqs, CLASS_BASED_THRESHOLD[c]))

        train_data = []
        train_label = []
        discriminative_patches = []
        for i in range(len(imgs)):
            discriminative_patch = []
            for idx in range(heatmaps[i].shape[0]):
                for idy in range(heatmaps[i].shape[1]):
                    if heatmaps[i][idx, idy] >= min(threshold1[i], threshold2[img_labels[i]]):
                        cropped = get_patch(i, idx, idy)
                        if np.sum(cropped > 0) >= TISSUE_RATIO * np.size(cropped):
                            train_data.append(cropped)
                            train_label.append(img_labels[i])
                            discriminative_patch.append((idx, idy))
            discriminative_patches.append(discriminative_patch)

        train_data = np.array(train_data)
        train_label = to_categorical(np.array(train_label))


######################################
# build image-level fusion model on histograms
######################################

# building histograms from discriminative patches
histograms = []
for ind in range(len(imgs)):
    patches = np.zeros((len(discriminative_patches[ind]), CROP_SIZE, CROP_SIZE, 1))
    for patch_ind, patch in enumerate(discriminative_patches[ind]):
        patches[patch_ind] = get_patch(ind, patch[0], patch[1])
    prob = model.predict(patches)
    hist = np.mean(prob, axis=0)
    histograms.append(hist)
histograms = np.array(histograms)

# build the fusion model. Currently this is essentially a logistic regression, may consider using sklearn.
X2_input = Input(shape=(2,), name="input_2o_1")
X2 = Dense(2, name="dense_2o_1", activation="softmax")(X2_input)

model2 = Model(inputs=X2_input, outputs=X2, name="second_order_model")
adam2 = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model2.compile(optimizer=adam2, loss="binary_crossentropy", metrics=["accuracy"])

model2.fit(histograms, img_labels_onehot, batch_size = 32, epochs=1000)

# TODO: make prediction using this 2-step model




