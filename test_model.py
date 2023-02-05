# Ethan Trott
# COS 470
# Semester Project

import tensorflow as tf
import numpy as np
import cv2

from imutils import paths
from sklearn.metrics import classification_report

test_dir = ""           #specify directory of images here
expected_result = 1     #expected result for images in this directory

model = tf.keras.models.load_model("saved_model")
classes = ["non-fire", "fire"]

def load_dataset(datasetPath):
  imagePaths = list(paths.list_images(datasetPath))
  data = []

  print("Loading and resizing images...")
  for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (128, 128))
    data.append(image)
  return np.array(data, dtype="float32")

test_images = load_dataset(test_dir)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

correct = 0
for i in range(len(test_images)):
  print('%s => %s || %s (expected %s)' % (list(paths.list_images(test_dir))[i], predictions[i], np.argmax(predictions[i]), expected_result))
  if (np.argmax(predictions[i]) == expected_result):
    correct += 1

print('%d correct out of %d, Accuracy:%f'% (correct, len(test_images), correct/len(test_images)))