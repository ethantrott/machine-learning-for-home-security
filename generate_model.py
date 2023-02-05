# Ethan Trott
# COS 470
# Semester Project

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import cv2
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

NUM_EPOCHS = 10
classes = ["non-fire", "fire"]

print(tf.__version__)

def load_dataset(datasetPath):
  imagePaths = list(paths.list_images(datasetPath))
  data = []

  print("Loading and resizing images...")
  for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (128, 128))
    data.append(image)
  return np.array(data, dtype="float32")

model = tf.keras.Sequential([
    tf.keras.layers.SeparableConv2D(16, (7, 7), padding="same", input_shape=(128,128,3)),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(2),
    tf.keras.layers.Activation("softmax")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

fireData = load_dataset("fire_images")
nonFireData = load_dataset("non_fire_images")

fireLabels = np.ones((fireData.shape[0],))
nonFireLabels = np.zeros((nonFireData.shape[0],))

data = np.vstack([fireData, nonFireData])
labels = np.hstack([fireLabels, nonFireLabels])
data /= 255

labels = tf.keras.utils.to_categorical(labels, num_classes=2)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.1)

aug = tf.keras.preprocessing.image.ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

print("Compiling model...")
opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9,
	decay=0.01 / NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("Fitting model...")
H = model.fit(
	x=trainX,
  y=trainY,
  batch_size=64,
	validation_data=(testX, testY),
	epochs=NUM_EPOCHS,
  verbose=1)

print("Testing accuracy...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=classes))

print("Saving model...")
model.save("saved_model")