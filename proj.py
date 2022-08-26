from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, InputLayer, UpSampling2D, Conv2DTranspose, LeakyReLU, AveragePooling2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Input, RepeatVector, Reshape, concatenate
from keras.models import Model
from tensorflow.keras.regularizers import l1_l2
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def dataset_prepare(train_set_path: str, test_set_path: str, train_label_path: str, test_label_path: str,
                    patch_size: np.array) -> list:
    train_set = []
    skipped_loc = []
    for image_path in glob.glob(f"{train_set_path}/*.npy"):
        try:
            train_set.append(np.load(image_path))
        except:
            skipped_loc.append(image_path)

    label_set = []
    skipped_loc_label = []
    for image_path in glob.glob(f"{train_label_path}/*.npy"):
        if not skipped_loc:
            label_set.append(np.load(image_path))
        else:
            if skipped_loc[0][15:-4] != image_path[15:-10]:
                label_set.append(np.load(image_path))
            else:
                skipped_loc_label.append(image_path)

    test_set = []
    skipped_loc_test = []
    for image_path in glob.glob(f"{test_set_path}/*.npy"):
        try:
            test_set.append(np.load(image_path))
        except:
            skipped_loc_test.append(image_path)

    test_label_set = []
    for image_path in glob.glob(f"{test_label_path}/*.npy"):
        test_label_set.append(np.load(image_path))

    train_set = np.reshape(train_set, (-1, patch_size[0], patch_size[1], 4))
    label_set = np.reshape(label_set, (-1, patch_size[0], patch_size[1], 5))
    test_set = np.reshape(test_set, (-1, patch_size[0], patch_size[1], 4))
    test_label_set = np.reshape(test_label_set, (-1, patch_size[0], patch_size[1], 5))

    return train_set, label_set, test_set, test_label_set


def unet_trial(input_size=(64, 128, 4), weight_file=None, kr=l1_l2(l1=0.1, l2=0.1)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    d4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(d4)

    up5 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5), conv2], axis=3)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv1], axis=3)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    conv8 = Conv2D(1, (1, 1), activation='softmax', kernel_regularizer=kr)(conv7)

    model = Model(inputs=inputs, outputs=conv8)

    if weight_file:
        model.load_weights(weight_file)
    return model


def unet_batchnorm(nclass=5, input_size=(64, 64, 4), weight_file=None,
                   kr=l1_l2(l1=0.1, l2=0.01), maps=[64, 128, 256, 512, 1024]):
    """
    UNet network using batch normalization features.
    """
    inputs = Input(input_size, name='Input')

    # Encoder
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c1)
    n1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(n1)

    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c2)
    n2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(n2)

    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c3)
    n3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(n3)

    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c4)
    n4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(n4)

    # Squeeze
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(p4)
    d5 = Dropout(0.30)(c5)
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(d5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    n6 = BatchNormalization()(u6)
    u6 = concatenate([n6, n4])
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    n7 = BatchNormalization()(u7)
    u7 = concatenate([n7, n3])
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    n8 = BatchNormalization()(u8)
    u8 = concatenate([n8, n2])
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    n9 = BatchNormalization()(u9)
    u9 = concatenate([n9, n1], axis=3)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c9)

    c10 = Conv2D(nclass, (1, 1), activation='softmax', kernel_regularizer=kr)(c9)
    model = Model(inputs=inputs, outputs=c10, name="UNetBatchNorm")

    if weight_file:
        model.load_weights(weight_file)
    return model


train_set, label_set, test_set, test_label_set = dataset_prepare("train_set_new_thresh", "test_set_new_thresh",
                                                                 "train_label_new", "test_label_new", (64, 64))
model = unet_batchnorm()
# model = unet_trial()

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])
model.summary()

history = model.fit(x=train_set, y=label_set, validation_split=0.2, epochs=100, batch_size=16, verbose=0)
loss, accuracy, f1_score, precision, recall = model.evaluate(test_set, test_label_set, verbose=0)
model.save_weights("my_model_1.h5")

y_pred_prob = model.predict(test_set)
y_pred = np.zeros((np.shape(y_pred_prob)[0], np.shape(y_pred_prob)[1], np.shape(y_pred_prob)[2]))
y_true = np.zeros((np.shape(y_pred_prob)[0], np.shape(y_pred_prob)[1], np.shape(y_pred_prob)[2]))
for i in range(np.shape(y_pred_prob)[0]):
    for j in range(np.shape(y_pred_prob)[1]):
        for k in range(np.shape(y_pred_prob)[2]):
            max_pxl_idx = np.argmax(y_pred_prob[i, j, k, :])
            y_pred[i, j, k] = max_pxl_idx
            max_label_idx = np.argmax(test_label_set[i, j, k, :])
            y_true[i, j, k] = max_label_idx

cm = confusion_matrix(np.ravel(y_true), np.ravel(y_pred), labels=[0, 1, 2, 3, 4])
print(cm)
df_cm = pd.DataFrame(cm, index=["unknown",  "agriculture", "forest", "city", "water"],
                     columns=["unknown",  "agriculture", "forest", "city", "water"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
fig = plt.figure()
h = history
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Model Loss')
plt.legend(['loss', 'val_loss'])
plt.axis([0, 10, 0, 5])
fig = plt.figure()
plt.show()
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['accuracy', 'val_acc'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Model Accuracy')
plt.show()
