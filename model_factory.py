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
def create_model(config):
  # the second probabilistic part of the model
  dense_input  = Input(shape=(None, config['height'], config['width'], config['num_channels']))
  dense_input_reshaped = tf.keras.layers.Reshape([-1, config['height'] * config['width'] * config['num_channels']])(dense_input)
  if config['is_probabilistic']:
    # brings uncertainty into the model weights
    dense_output = tfp.layers.DenseFlipout(config['num_channels'], activation=tf.nn.leaky_relu)(dense_input_reshaped)
    dense_output = tfp.layers.DenseFlipout(config['num_channels'], activation=tf.nn.leaky_relu)(dense_output)
    dense_output = tf.keras.layers.Dropout(config['dropout_rate'])(dense_output)
    dense_output = tfp.layers.DenseFlipout(config['num_classes'])(dense_output)
    # dense_output = tfp.layers.DenseFlipout(num_classes, activation=tf.nn.softmax)(dense_output)
    # brings uncertainty into the output
    output_distribution = tfp.layers.DistributionLambda(lambda t: tfd.Categorical(logits=t))(dense_output)
  else:
    #
    dense_output = tf.keras.layers.Dense(config['num_channels'], activation=tf.nn.relu)(dense_input_reshaped)
    dense_output = tf.keras.layers.Dense(config['num_channels'], activation=tf.nn.relu)(dense_output)
    dense_output = tf.keras.layers.Dropout(config['dropout_rate'])(dense_output, training=True)
    output_distribution = tf.keras.layers.Dense(config['num_classes'], activation=tf.nn.softmax)(dense_output)
  #
  top_model = Model(inputs=dense_input, outputs=output_distribution, name='top_model')
  #
  if config['finetune_feature_extractor']:
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
  if not config['is_probabilistic']:
    loss = lambda labels, logits: tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
  elif config['loss_name'] == 'negloglik':
    loss = lambda labels, labels_distribution: -tf.reduce_mean(labels_distribution.log_prob(labels))
  elif config['loss_name'] == 'elbo':
    loss = lambda labels, labels_distribution: -tf.reduce_mean(labels_distribution.log_prob(labels)) + sum(full_model.losses)
  #
  if config['optimizer_name'] == 'rmsprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=config['learning_rate'])
  elif config['optimizer_name'] == 'sgld':
    optimizer = tfp.optimizer.StochasticGradientLangevinDynamics(0.001)
  elif config['optimizer_name'] == 'vsgd':
    optimizer = tfp.optimizer.VariationalSGD(batch_size=25, total_num_examples=125, max_learning_rate=0.01)
  #
  full_model.compile( \
      optimizer=optimizer, \
      loss=loss
  )
  # return the completed model
  return full_model



# MODEL training
def train_model(model, x_train, y_train, x_val, y_val, config, target_field_mean=None):
  #
  datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
  # compute quantities required for featurewise normalization
  # (std, mean, and principal components if ZCA whitening is applied)
  datagen.fit(x_train)
  #
  # own callbacks
  callbacks = []
  if target_field_mean == None:
    target_field_mean = y_train.mean()
  if config['num_classes'] == 2:
    f1andUncertaintiesCallback = my_callbacks.F1andUncertaintiesCallback(validation_data=(x_val, y_val), interval=config['validation_interval'], target_field_mean=target_field_mean)
    callbacks.append(f1andUncertaintiesCallback)
  #
  if config['model_name'] == None:
    date_values = [str(x) for x in time.gmtime()]
    config['model_name'] = '_'.join(date_values)
    os.makedirs("models/" + config['model_name'])
  filepath="models/" + config['model_name'] + "/checkpoint-{epoch:02d}.hdf5"
  checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='f1_score', verbose=1, save_best_only=True, mode='max', period=config['validation_interval'])
  callbacks.append(checkpoint)
  #
  early_stopper = tf.keras.callbacks.EarlyStopping(monitor='f1_score', patience=config['num_epochs'] / 4, restore_best_weights=True, mode='max')
  callbacks.append(early_stopper)
  # lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss')
  #
  print('fit model!')
  history = model.fit_generator( \
   datagen.flow(x_train, y_train, batch_size=config['batch_size']), \
   epochs=config['num_epochs'],\
   validation_data=datagen.flow(x_val, y_val, batch_size=config['batch_size']),\
   callbacks=callbacks \
  )
  #
  print('save model!')
  model.save_weights("models/" + config['model_name'] + "/final")
  # model.save('models/negative_loglikelihood_model.h5')



# calculate_flattened_predictions(model, test_images, test_labelss, train_labels.mean())
def calculate_flattened_predictions(model, x, y, target_field_mean=0.5, num_particles=10):
  print('calculate predictions!')
  prediction_list = [model.predict(x) for i in range(num_particles)]
  #
  predictions_field = np.array(prediction_list)
  mean_field = np.mean(predictions_field, axis=0)
  var_field = np.var(predictions_field, axis=0)
  target_field = y
  if len(mean_field.shape) == 3:
     mean_field = np.squeeze(mean_field)
     mean_field = np.transpose(mean_field)
     mean_field = mean_field[0]
     mean_field = np.transpose(mean_field)
     var_field = np.squeeze(var_field)
     var_field = np.transpose(var_field)
     var_field = var_field[0]
     var_field = np.transpose(var_field)
  #
  mean_field_flattened = mean_field.flatten()
  var_field_flattened = var_field.flatten()
  target_field_flattened = target_field.flatten()
  # TODO normalizing with the mean can't be the proper way to this!
  quantile = np.quantile(mean_field_flattened, 1 - target_field_mean)
  pred_field_flattened = np.array(list(map(lambda idx: mean_field_flattened[idx] >= quantile, range(mean_field_flattened.size))))
  #
  return pred_field_flattened, var_field_flattened, target_field_flattened
