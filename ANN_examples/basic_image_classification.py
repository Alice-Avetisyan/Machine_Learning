# Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__) ->> 2.0.0

# loading the dataset ->> returns NumPy arrays
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# the images are 28x28 np arrays, with pixel values ranging from 0 to 255

# class names are not included in the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# data exploration
print('-----Train Images: Shape-----\n', train_images.shape)
print('-----Train Labels: Length-----\n', len(train_labels))

print('-----Train Labels-----\n', train_labels)

print('-----Test Images: Shape-----\n', test_images.shape)
print('-----Test Labels: Length-----\n', len(test_labels))

# preprocess the data

# Will show a image of a shoe
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# normalization/scaling
train_images = train_images/255.0
test_images = test_images/255.0

# to verify that the data is in the correct format and we are ready to build and train the network,
# let's display the first 25 images from the training set, and also display the class name below each image

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)  # n_rows, n_cols, num ->> cannot be 0
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# building the model

# seting up the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # transforms images from two-dim array to one-dim array
    keras.layers.Dense(128, activation='relu'),  # Dense layers cannot take 3d arrays ->> height, weight, deep???
    keras.layers.Dense(10)
])

# compiling the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# training the model

# feeding the training data to the model
model.fit(train_images, train_labels, epochs=10)

# evaluating the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy: ', test_acc)
# the accuracy on the test dataset is little less than the accuracy on the training dataset
# test dataset accuracy < train dataset accuracy ->> 0.88 > 0.91
''' 
    this gap represents overfitting ->> model performs worse on new,
    previously unseen inputs than it does on the training data
    An overfitted model "memorizes" the noise and details in the training dataset to a point
    where it negatively impacts the performance of the model on the new data
'''

# making predictions
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
'''
    A prediction is an array of 10 numbers. 
    They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing.
'''

print('Confidence: ', np.argmax(predictions[0]))
# the model is confident that the image is an ankle boot ->> class_names[9]
print('Class name: ', test_labels[0])  # examining the test label shows that the classification is correct


def plot_images(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predidcted_label = np.argmax(predictions_array)
    if predidcted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% {}".format(class_names[predidcted_label],
                                       100*np.max(predictions_array),
                                       class_names[true_label]),
                                       color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# verifying predictions
# i = 0
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_images(i, predictions[i], test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions[i], test_labels)
# plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_images(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# using the trained model

# Grab an image from the test dataset.
img = test_images[1]
print('\nImage shape: ', img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))
print('adding the image to a batch:', img.shape)

predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

print(np.argmax(predictions_single[0]))






