import math
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint
from imgaug.augmentables.kps import KeypointsOnImage
from sklearn.model_selection import train_test_split

BATCH_SIZE = 16
MODELS_DIRECTORY = "../models"
TRAIN_FILE_PATH = "../data/training.csv"


def get_rgbimage_from_string(image_pixels_string):
    
    image_pixels_list = [int(point) for point in image_pixels_string.split(" ")]
    image_side = int(np.sqrt(len(image_pixels_list)))
    image = np.reshape(image_pixels_list, (image_side, image_side))
    image = np.stack([image, image, image], axis=2)
    
    return image

def display_image(image):
    
    plt.figure()
    plt.imshow(image.astype(np.uint8))
    
    return

def get_image_keypoints(index):
    
    keypoints = train_df.iloc[index][:-1].values
    
    return keypoints

def draw_keypoints(image, keypoints):
    
    for i in np.linspace(0, len(keypoints)-2, 15, dtype=int):
        
        if math.isnan(keypoints[i]) or math.isnan(keypoints[i+1]):
            continue
        if keypoints[i] < 0 or keypoints[i] > 95:
            continue
        if keypoints[i+1] < 0 or keypoints[i+1] > 95:
            continue
        image[int(keypoints[i+1]), int(keypoints[i])] = 255
        
    return image

class KeyPointsDataset(tf.keras.utils.Sequence):
    
    def __init__(self, images, keypoints, augmentations, batch_size, shuffle=True):
        
        self.images = images
        self.keypoints = keypoints
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))
        
        return
    
    def __len__(self):
        
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        images = [self.images[k] for k in indexes]
        keypoints = [self.keypoints[k] for k in indexes]
        images, keypoints = self.data_generation(images, keypoints)

        return (images, keypoints)

    def on_epoch_end(self):
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
        return
    
    def data_generation(self, images, keypoints):
        
        len_image_side = images[0].shape[0]
        len_keypoints = len(keypoints[0])
        batch_images = np.empty((self.batch_size, len_image_side, len_image_side, 3), dtype="float32")
        batch_keypoints = np.empty((self.batch_size, len_keypoints), dtype="float32")

        for i, (image, keypoint) in enumerate(zip(images, keypoints)):
            keypoint_pairs = []
            
            for j in np.arange(0, len(keypoint), 2):
                keypoint_pairs.append(Keypoint(x=int(keypoint[j]), y=int(keypoint[j+1])))
            
            kps_obj = KeypointsOnImage(keypoint_pairs, shape=image.shape)
            (new_images, new_kps_obj) = self.augmentations(image=image, keypoints=kps_obj)
            batch_images[i] = new_images
            kp_temp = []

            for keypoint in new_kps_obj:
                kp_x = np.nan_to_num(keypoint.x)
                kp_y = np.nan_to_num(keypoint.y) 
                if kp_x < 0 or kp_x > 95:
                    kp_x = 0
                if kp_y < 0 or kp_y > 95:
                    kp_y = 0
                kp_temp.append(kp_x)
                kp_temp.append(kp_y)

            batch_keypoints[i] = kp_temp
        
        return batch_images, batch_keypoints

train_df = pd.read_csv(TRAIN_FILE_PATH)
train_df.head()
train_df = train_df.fillna(0)
no_rows, no_columns = train_df.shape
labels = []

for i in range(no_rows):
    labels.append(train_df.iloc[i][:-1].values.astype(float))

images = []

for i in range(no_rows):
    images.append(get_rgbimage_from_string(train_df.iloc[i]["Image"]).astype(float))

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.1)

train_aug = iaa.Sequential([iaa.Resize(X_train[0].shape[0:2], interpolation="linear"), iaa.Fliplr(0.3), iaa.Sometimes(0.5, iaa.Affine(rotate=(-10, 10), scale=(0.75, 1.25)))])
test_aug = iaa.Sequential([iaa.Resize(X_train[0].shape[0:2], interpolation="linear")])

train_dataset = KeyPointsDataset(X_train, y_train, train_aug, BATCH_SIZE)
val_dataset = KeyPointsDataset(X_train, y_train, test_aug, BATCH_SIZE)

count = 0

for images, labels in train_dataset:
    count += 1
    
    for image, label in zip(images, labels):
        image_w_keypoints = draw_keypoints(image.copy(), label)
        display_image(image_w_keypoints)
    
    if count == 1:
        break

def get_model(base_trainable=False):
    
    base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
    base_model.trainable = base_trainable
    
    inputs = tf.keras.Input(shape=(96, 96, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(30)(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

model = get_model()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODELS_DIRECTORY + "/model-base-frozen.h5", monitor="val_loss", save_best_only=True, mode="min"),
]

model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001))

if not os.path.exists(MODELS_DIRECTORY):
    os.makedirs(MODELS_DIRECTORY)

history = model.fit(train_dataset, validation_data=val_dataset, callbacks=callbacks, epochs=5)

plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

model_file = os.listdir(MODELS_DIRECTORY)[-1]
model = tf.keras.models.load_model(MODELS_DIRECTORY + "/" + model_file)

X_val_tensor = tf.convert_to_tensor(X_val)
predictions = model.predict(X_val_tensor)
X_val_tensor.shape

count = 0

for image, prediction in zip(X_val, predictions):
    count += 1
        
    if count == 10:
        break
    
    display_image(draw_keypoints(image.copy(), prediction))

model.summary()
model.layers[3].trainable = True
model.summary()
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODELS_DIRECTORY + "/model-finetuned.h5", monitor="val_loss", save_best_only=True),
]

model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.0003))
history = model.fit(train_dataset, callbacks=callbacks, validation_data=val_dataset, epochs=1)

plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

predictions = model.predict(X_val_tensor)
count = 0

for image, prediction in zip(X_val, predictions):
    count += 1
    
    if count == 50:
        break
    
    display_image(draw_keypoints(image.copy(), prediction))

