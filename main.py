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

corrosion_config = {
	'finetune_feature_extractor' : True,
	'num_classes' : 2,
	'is_probabilistic' : True,
	'dataset_dir' : 'data/corrosion',
	'validation_interval' : 1,
	'val_split' : 0.2,
	'optimizer_name' : 'rmsprop',
	'loss_name' : 'elbo',
	'num_channels' : 1536,
	'height' : 5,
	'width' : 5,
	'model_name' : None,
	'batch_size' : 250,
	'dropout_rate' : 0.5,
	'learning_rate' : 0.001,
	'num_epochs' : 100,
	'num_particles' : 10
}

multiclass_isic2019_config = {
	'finetune_feature_extractor' : True,
	'num_classes' : 8,
	'is_probabilistic' : True,
	'dataset_dir' : 'data/isic2019_multiclass',
	'validation_interval' : 1,
	'val_split' : 0.01,
	'optimizer_name' : 'adam',
	'loss_name' : 'elbo',
	'num_channels' : 1536,
	'height' : 5,
	'width' : 5,
	'model_name' : None,
	'batch_size' : 25,
	'dropout_rate' : 0.5,
	'learning_rate' : 0.001,
	'num_epochs' : 100,
	'num_particles' : 10,
	'scaling_factor' : 1014,
	'num_splits' : 10
}

config = multiclass_isic2019_config

print('enter your command!')
code.interact(local=dict(globals(), **locals()))
# possible example run
#
model = model_factory.create_model(config=config)
test_images, test_labels = dataloader.load_and_preprocess_multiclass_validation_data(config=config)
model_factory.train_model(model, test_images, test_labels, config=config)


#train_images, test_images, train_labels, test_labels = dataloader.load_and_preprocess_binary_data(config=config)
#
model = model_factory.create_model(config=config)
#
model_factory.train_model(model, train_images, train_labels, test_images, test_labels, config=config)
# some sort of evaluation would fit here
x = model_factory.calculate_flattened_predictions(model, test_images, test_labels, train_labels.mean())
results = (x[0] == x[2])