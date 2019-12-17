import os
import tensorflow as tf
import segmentation_models as sm
from PIL import Image
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras.models import load_model
import sklearn, time
import code, shutil # code.interact(local=dict(globals(), **locals()))

# own librabries
import model_factory, my_utils, my_callbacks


# DATA preprocessing
def load_and_preprocess_binary_data(config):
  positive_images = []
  negative_images = []
  #
  print('load images')
  for it, file in enumerate(os.listdir(config['dataset_dir'] + '/positive_samples')):
    img = Image.open(config['dataset_dir'] + '/positive_samples/' + file)
    img = img.resize([224,224])
    #img = img.resize([299,299])
    positive_images.append(img)

  for it, file in enumerate(os.listdir(config['dataset_dir'] + '/negative_samples')):
    img = Image.open(config['dataset_dir'] + '/negative_samples/' + file)
    img = img.resize([224,224])
    #img = img.resize([299,299])
    negative_images.append(img)
  #

  print('preprocess data')
  positive_train = positive_images[:-int(config['val_split'] * len(positive_images))]
  positive_test = positive_images[-int(config['val_split'] * len(positive_images)):]

  negative_train = negative_images[:-int(config['val_split'] * len(negative_images))]
  negative_test = negative_images[-int(config['val_split'] * len(negative_images)):]

  train_images = np.concatenate([np.stack(positive_train), np.stack(negative_train)])
  test_images = np.concatenate([np.stack(positive_test), np.stack(negative_test)])

  if len(positive_train) <= len(negative_train):
    train_labels = np.concatenate([np.ones(len(positive_train), dtype=np.int32), np.zeros(len(negative_train), dtype=np.int32)])
    test_labels = np.concatenate([np.ones(len(positive_test), dtype=np.int32), np.zeros(len(negative_test), dtype=np.int32)])
  else:
    train_labels = np.concatenate([np.zeros(len(positive_train), dtype=np.int32), np.ones(len(negative_train), dtype=np.int32)])
    test_labels = np.concatenate([np.zeros(len(positive_test), dtype=np.int32), np.ones(len(negative_test), dtype=np.int32)])

  train_images = preprocess_input(train_images)
  test_images = preprocess_input(test_images)
  #
  train_images = np.transpose(train_images, [0,3,1,2])
  test_images = np.transpose(test_images, [0,3,1,2])
  #
  return train_images, test_images, train_labels, test_labels



def extract_data_features(dataset_dir, save_features=True):
  print('extract features with pretrained inception net')
  #
  train_images, test_images, train_labels, test_labels = load_and_preprocess_binary_data(dataset_dir)
  # resnet = ResNet50(include_top=False, weights='imagenet')
  # resnet.predict(np.zeros([5,224,224,3]))
  # code.interact(local=dict(globals(), **locals()))
  train_images = InceptionResNetV2(include_top=False, weights='imagenet').predict(train_images)
  test_images = InceptionResNetV2(include_top=False, weights='imagenet').predict(test_images)
  # TODO store these + the groundtruth somewhere in order to avoid long computation times!!!
  # code.interact(local=dict(globals(), **locals()))
  if save_data:
    np.savez(open(dataset_dir + '/preprocessed.npz','wb'), train_images=train_images, test_images=test_images, train_labels=train_labels, test_labels=test_labels)
  #
  return train_images, test_images, train_labels, test_labels



def load_features(dataset_dir):
  print('load data!')
  data = np.load(open(dataset_dir + '/preprocessed.npz','rb'))
  train_images = data['train_images']
  test_images = data['test_images']
  train_labels = data['train_labels']
  test_labels = data['test_labels']
  return train_images, test_images, train_labels, test_labels
