from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import WeightedRandomSampler
import tensorflow as tf
import numpy as np
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive

# Loading data from Google Drive
drive.mount("/content/gdrive")

# Load Dataset
data_dir = '/content/gdrive/My Drive/covid_dataset'
# non-COVID train dataset
train_non_covid_dir = os.path.join('/content/gdrive/My Drive/covid_dataset/train/class-0')
# COVID train dataset
train_covid_dir = os.path.join('/content/gdrive/My Drive/covid_dataset/train/class-1')
# non-COVID valid dataset
valid_non_covid_dir = os.path.join('/content/gdrive/My Drive/covid_dataset/valid/class-0')
# COVID valid dataset
valid_covid_dir = os.path.join('/content/gdrive/My Drive/covid_dataset/valid/class-1')


# rescale by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

# load pictures
test_generator = test_datagen.flow_from_directory(
        '/content/gdrive/My Drive/covid_dataset/test/', 
        classes = ['class-0', 'class-1'],
        target_size=(256, 256),
        batch_size=120,
        class_mode='binary')

train_generator = train_datagen.flow_from_directory(
        '/content/gdrive/My Drive/covid_dataset/train/', 
        classes = ['class-0', 'class-1'],
        target_size=(256, 256),
        batch_size=120,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        '/content/gdrive/My Drive/covid_dataset/valid/', 
        classes = ['class-0', 'class-1'],
        target_size=(256, 256),
        batch_size=19,
        class_mode='binary',
        shuffle=False)


# creating the model
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
# 64 neuron hidden layer
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(1, activation='sigmoid')])

model.summary() # model summary to visualize the model
model.compile(optimizer = tf.optimizers.Adam(), loss = 'binary_crossentropy', metrics=['accuracy'])

m = model.fit(train_generator,
      steps_per_epoch=5,  
      epochs=10,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=6)

model.evaluate(test_generator)