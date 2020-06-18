DATA_DIR = 'C:\\Users\\Alice\\Desktop\\Data\\PetImages'
CLASSES = ['Cat', 'Dog']

import os  # operations with os ->> creating files...
import cv2  # image operations
import matplotlib.pyplot as plt

class_path = os.path.join(DATA_DIR, CLASSES[0]) # katuneri nkarneri chanarh
img_name = os.listdir(class_path)[6] # @ntrum enq katvi nkar@
#print(img_name)
img_path = os.path.join(class_path, img_name) # katvi nkari lriv chanaparh
#print(img_path)
img_pixels = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # nkari pixelnern en kardum

# plt.imshow(img_pixels)
# plt.show()

print(img_pixels.shape)  # width, height, rgb

IMG_WIDTH = 70
IMG_HEIGHT = 70

converted_img_px = cv2.resize(img_pixels, (IMG_HEIGHT, IMG_WIDTH)) # nkar@ ktrecinq

# plt.imshow(converted_img_px, cmap='gray')
# plt.show()

input_data = []

for cls in CLASSES: # verevi katarvac gorcoxutyunner
    class_path = os.path.join(DATA_DIR, cls)
    class_id = CLASSES.index(cls)
    for img_name in os.listdir(class_path):
        try:
            img_pixels = cv2.imread(os.path.join(class_path, img_name), cv2.IMREAD_GRAYSCALE)
            converted_img_px = cv2.resize(img_pixels, (IMG_HEIGHT, IMG_WIDTH))
            input_data.append([converted_img_px, class_id])
        except:
            pass

print(len(input_data))
print(input_data[0])

import random

random.shuffle(input_data)
#print('\n', input_data)

X = []
y = []

for features, label in input_data:
    X.append(features)
    y.append(label)

import numpy as np

X = np.array(X).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1) # 1 - da mek guina: sev yev spitak
X = X / 255.0

y = np.array(y)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 1))) # filtraciai patuhanner
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2))) # Subsample layer  - texnika sjatiya (poqracnuma, shat@ qicha dardznum)

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten()) # 2d dardznel 1d
model.add(Dense(64, activation='relu'))
model.add(Dense(1,  activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, verbose=1, epochs=3, batch_size=64, validation_split=0.1)
