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


# DATA loading
# method to load the isic data
def load_isic_data():
  x_train = []
  y_train = []
  for file in os.listdir('data/ISBI2016_ISIC_Part1_Training_Data'):
    x_img = Image.open('data/ISBI2016_ISIC_Part1_Training_Data/' + file)
    x_img = x_img.resize([224,224])
    #x_train.append(np.array(x_img) / 255.0)
    x_train.append(np.array(x_img))
    nr = file[len('ISIC_'):-len('.jpg')]
    y_img = Image.open('data/ISBI2016_ISIC_Part1_Training_GroundTruth/ISIC_' + nr + '_Segmentation.png')
    y_img = y_img.resize([224,224])
    #y_train.append(np.array(np.array(y_img) / 255, dtype=np.int8))
    y_train.append(np.expand_dims(np.array(y_img) / 255, axis=-1))
  return np.stack(x_train[:-100]), np.stack(y_train[:-100]), np.stack(x_train[-100:]), np.stack(y_train[-100:])


# DATA preprocessing
def preprocess_data():
  print('preprocess data!')
  x_train, y_train, x_val, y_val = load_isic_data()
  # preprocess input according to keras preprocessing for the BACKBONE
  x_train = preprocess_input(x_train) # TODO can be done in a far better way, that regards the backbone!
  x_val = preprocess_input(x_val)
  print('data preprocessed!')
  return x_train, y_train, x_val, y_val

