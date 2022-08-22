import keras
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, InputLayer, UpSampling2D, Conv2DTranspose, LeakyReLU, AveragePooling2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Input, RepeatVector, Reshape, concatenate
from keras.models import Model
import imageio
import glob
import numpy as np


train_set = []
skipped_loc = []
for image_path in glob.glob("train_set/*.png"):
    try:
        train_set.append(imageio.imread(image_path))
    except:
        skipped_loc.append(image_path)

label_set = []
skipped_loc_label = []
for image_path in glob.glob("train_npy_label/*.npy"):
    if not skipped_loc:
        label_set.append(np.load(image_path))
    else:
        if skipped_loc[0][15:-4] != image_path[15:-10]:
            label_set.append(np.load(image_path))
        else:
            skipped_loc_label.append(image_path)

test_set = []
skipped_loc_test = []
for image_path in glob.glob("test_set/*.png"):
    try:
        test_set.append(imageio.imread(image_path))
    except:
        skipped_loc_test.append(image_path)

test_label_set = []
for image_path in glob.glob("test_npy_label/*.npy"):
    test_label_set.append(np.load(image_path))


train_set = np.reshape(train_set, (-1, 64, 128, 4))
label_set = np.reshape(label_set, (-1, 64, 128, 1))
test_set = np.reshape(test_set, (-1, 64, 128, 4))
test_label_set = np.reshape(test_label_set, (-1, 64, 128, 1))


model = Sequential()
inputs = Input((64, 128, 4))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
up3 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2), conv1], axis=3)
conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
conv4 = Conv2D(1, (3, 3), activation='relu', padding='same')(conv3)

model = Model(inputs=[inputs], outputs=[conv4])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()


history = model.fit(x=train_set, y=label_set, validation_split=0.2, epochs=20, batch_size=16)
xx = model.predict(test_set)