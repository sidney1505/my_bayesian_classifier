import os
import tensorflow as tf
import segmentation_models as sm
from PIL import Image
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
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
import dataloader, my_utils, my_callbacks

# own LOSSES

# create loss function for probabilistic model
negloglik = lambda labels_distribution, labels: -tf.reduce_mean(labels.log_prob(labels_distribution))



# MODEL definition
def create_bayesian_le_net():
  pass


# MODEL definition
def create_model(is_probabilistic=True, loss=negloglik, height=5, width=5, num_channels = 1536, num_classes = 2, backbone='inceptionresnetv2', finetune_feature_extractor=True):
  # the second probabilistic part of the model
  if is_probabilistic:
    dense_input  = Input(shape=(None, height, width, num_channels))
    dense_input_reshaped = tf.keras.layers.Reshape([-1, height * width * num_channels])(dense_input)
    # brings uncertainty into the model weights
    dense_output = tfp.layers.DenseFlipout(num_channels, activation=tf.nn.leaky_relu)(dense_input_reshaped)
    dense_output = tfp.layers.DenseFlipout(num_channels, activation=tf.nn.leaky_relu)(dense_output)
    dense_output = tfp.layers.DenseFlipout(num_classes)(dense_output)
    # brings uncertainty into the output
    output_distribution = tfp.layers.DistributionLambda(lambda t: tfd.Categorical(logits=t))(dense_output)
  else:
    dense_input  = Input(shape=(None, None, None, num_channels))
    dense_output = tf.keras.layers.Dense(num_channels, activation=tf.nn.relu)(dense_input)
    output_distribution = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(dense_input)
  #
  top_model = Model(inputs=dense_input, outputs=output_distribution, name='top_model')
  #
  if finetune_feature_extractor:
    #
    feature_extractor = InceptionResNetV2(include_top=False, weights='imagenet')
    # put both parts of the model together
    unet_output = feature_extractor.output
    full_output = top_model(unet_output)
    full_model = Model(inputs=feature_extractor.input, outputs=full_output)
  else:
    full_model = top_model
  # compiles the keras model
  print('compile model!')
  # elbo = lambda labels_distribution, labels: -tf.reduce_mean(labels.log_prob(labels_distribution)) + sum(full_model.losses) # TODO integrate this in the loss!
  full_model.compile( \
      'Adam', \
      loss=loss
  )
  # return the completed model
  return full_model



# MODEL training
def train_model(model, x_train, y_train, x_val, y_val, num_epochs=50, initial_epoch=0, target_field_mean=None):
  #
  values = [str(x) for x in time.gmtime()]
  model_name = '_'.join(values)
  os.makedirs("models/" + model_name)
  filepath="models/" + model_name + "/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5"
  checkpoint =  tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=num_epochs / 10)
  early_stopper = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
  lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss')
  # own callbacks
  if target_field_mean == None:
    target_field_mean = y_train.mean()
  f1andUncertaintiesCallback = my_callbacks.F1andUncertaintiesCallback(validation_data=(x_val, y_val), interval=1, target_field_mean=target_field_mean)
  #
  print('fit model!')
  history = model.fit( \
   x=x_train,\
   y=y_train,\
   batch_size=16,\
   epochs=num_epochs,\
   validation_data=(x_val, y_val),\
   callbacks=[early_stopper, checkpoint, lr_decay, f1andUncertaintiesCallback] \
  )
  #
  print('save model!')
  model.save_weights("models/" + model_name + "/final")
  # model.save('models/negative_loglikelihood_model.h5')



# 
def calculate_flattened_predictions(model, x, y, target_field_mean=0.5, num_particles=10):
  print('calculate predictions!')
  prediction_list = [model.predict(x) for i in range(num_particles)]
  #
  predictions_field = np.array(prediction_list)
  mean_field = np.mean(predictions_field, axis=0)
  var_field = np.var(predictions_field, axis=0)
  target_field = y
  #
  mean_field_flattened = mean_field.flatten()
  var_field_flattened = var_field.flatten()
  target_field_flattened = target_field.flatten()
  # TODO normalizing with the mean can't be the proper way to this!
  quantile = np.quantile(mean_field_flattened, 1 - target_field_mean)
  pred_field_flattened = np.array(list(map(lambda idx: mean_field_flattened[idx] >= quantile, range(mean_field_flattened.size))))
  #
  return pred_field_flattened, var_field_flattened, target_field_flattened
