import os
import tensorflow as tf
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
import model_factory, dataloader, my_utils, my_callbacks
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
# from tensorflow.keras.applications.resnet import ResNet50
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np
import code, os
from PIL import Image
#code.interact(local=dict(globals(), **locals()))

# (ctrain_images, ctrain_labels), (ctest_images, ctst_labels) = cifar10.load_data()
FINETUNE_FEATURE_EXTRACTOR = True
DATASET_DIR = 'data/corrosion'
# DATASET_DIR = 'data/isic2019'

print('enter your command!')
code.interact(local=dict(globals(), **locals()))
# possible example run
#
train_images, test_images, train_labels, test_labels = dataloader.load_and_preprocess_data(DATASET_DIR)
#
model = model_factory.create_model(finetune_feature_extractor=FINETUNE_FEATURE_EXTRACTOR)
#
model_factory.train_model(model, train_images, train_labels, test_images, test_labels)
# some sort of evaluation would fit here