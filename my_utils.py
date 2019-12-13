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
import model_factory, dataloader, my_callbacks



# own version of calculating the F1 score
def calculate_f1_score(pred_field_flattened, target_field_flattened, verbose=True):
  assert pred_field_flattened.size == target_field_flattened.size, "sizes aren't equal in calculate_f1_score"
  true_positive = 0
  false_positive = 0
  false_negative = 0
  true_negative = 0
  for i in range(target_field_flattened.size):
      if pred_field_flattened[i] == 1 and target_field_flattened[i] == 1:
        true_positive += 1
      elif pred_field_flattened[i] == 1 and target_field_flattened[i] == 0:
        false_positive += 1
      elif pred_field_flattened[i] == 0 and target_field_flattened[i] == 1:
        false_negative += 1
      elif pred_field_flattened[i] == 0 and target_field_flattened[i] == 0:
        true_negative += 1
      else:
        print('error!')
  precision = true_positive / (true_positive + false_positive)
  recall = true_positive / (true_positive + false_negative)
  if precision + recall == 0:
    f1_score = 0.0
  else:
    f1_score = 2 * precision * recall / (precision + recall)
  #
  if verbose:
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('f1_score: ' + str(f1_score))
  return f1_score




#
def calculate_uncertainties(pred_field_flattened, var_field_flattened, target_field_flattened, verbose=True):
  assert pred_field_flattened.size == var_field_flattened.size == target_field_flattened.size, "sizes don't match in check_uncertainties"
  is_correct_field_flattened = np.array(pred_field_flattened == target_field_flattened, dtype=np.int32)
  num_correct = np.sum(is_correct_field_flattened)
  num_false = is_correct_field_flattened.size - num_correct
  ratio_correct = num_correct / is_correct_field_flattened.size
  idxs = np.argsort(var_field_flattened)[::-1]
  top_10_size = int(var_field_flattened.size / 10)
  # check top 10 error rates
  num_correct_in_top_10 = 0
  for idx in range(top_10_size):
    num_correct_in_top_10 += is_correct_field_flattened[idxs[idx]]
  num_false_in_top_10 = top_10_size - num_correct_in_top_10
  proportion_of_mistakes_in_top_10 = num_false_in_top_10 / num_false
  # check bottom 10 error rates
  num_correct_in_bottom_10 = 0
  for idx in range(top_10_size):
    num_correct_in_bottom_10 += is_correct_field_flattened[idxs[-(idx + 1)]]
  num_false_in_bottom_10 = top_10_size - num_correct_in_bottom_10
  proportion_of_mistakes_in_bottom_10 = num_false_in_bottom_10 / num_false
  #
  num_correct_in_middle = 0
  for idx in range(top_10_size,is_correct_field_flattened.size - top_10_size):
    num_correct_in_middle += is_correct_field_flattened[idxs[-(idx + 1)]]
  num_false_in_middle = (is_correct_field_flattened.size - 2 * top_10_size) - num_correct_in_middle
  proportion_of_mistakes_in_middle_part = num_false_in_middle / num_false
  #
  if verbose:
    print('ratio_correct: ' + str(ratio_correct))
    print('num_correct: ' + str(num_correct))
    print('num_false: ' + str(num_false))
    print('proportion_of_mistakes_in_top_10: ' + str(proportion_of_mistakes_in_top_10))
    print('proportion_of_mistakes_in_bottom_10: ' + str(proportion_of_mistakes_in_bottom_10))
    print('proportion_of_mistakes_in_middle_part: ' + str(proportion_of_mistakes_in_middle_part))
  return proportion_of_mistakes_in_top_10



# own version of the F1 score in tensorflow - unfortunately it slows down the training extremely
def tensorflow_f1_score(target_field, mean_pred_field):
  #
  target_field_flattened = tf.reshape(target_field, [-1])
  #
  mean_pred_field_flattened = tf.reshape(mean_pred_field, [-1])
  quantile = tfp.stats.percentile(mean_pred_field_flattened, 100 * (1 - train_positive_ratio))
  pred_field_flattened = tf.map_fn(lambda mean_pred: tf.cast(mean_pred >= quantile, tf.float32), mean_pred_field_flattened)
  pred_field_flattened_int = tf.cast(pred_field_flattened, tf.int32)
  target_field_flattened_int = tf.cast(target_field_flattened, tf.int32)
  '''#
  update_true_positive = lambda value, pred, target: value + tf.cast(pred == 1 and target == 1, tf.int32)
  update_false_positive = lambda value, pred, target: value + tf.cast(pred == 1 and target == 0, tf.int32)
  update_false_negative = lambda value, pred, target: value + tf.cast(pred == 0 and target == 1, tf.int32)
  update_true_negative = lambda value, pred, target: value + tf.cast(pred == 0 and target == 0, tf.int32)
  #
  update_values = lambda tp, fp, fn, tn, pred, target: \
    update_true_positive(tp, pred, target), update_false_positive(fp, pred, target), update_false_negative(fn, pred, target), update_true_negative(tn, pred, target)
  #
  tp, fp, fn, tn = tf.reduce(lambda it, akk: update_values(akk[0], akk[1], akk[2], akk[3], it[0], it[1]), zip(pred_field_flattened_int, target_field), [0,0,0,0])'''
  #
  true_positive_fn = lambda pred, target: tf.cast(tf.logical_and(tf.math.equal(pred, 1), tf.math.equal(target, 1)), tf.int32)
  false_positive_fn = lambda pred, target: tf.cast(tf.logical_and(tf.math.equal(pred, 1), tf.math.equal(target, 0)), tf.int32)
  false_negative_fn = lambda pred, target: tf.cast(tf.logical_and(tf.math.equal(pred, 0), tf.math.equal(target, 1)), tf.int32)
  true_negative_fn = lambda pred, target: tf.cast(tf.logical_and(tf.math.equal(pred, 0), tf.math.equal(target, 0)), tf.int32)
  #
  calc_value = lambda pred, target: tf.stack([true_positive_fn(pred, target), false_positive_fn(pred, target), false_negative_fn(pred, target), true_negative_fn(pred, target)], -1)
  #
  preds_targets_zipped = tf.stack([pred_field_flattened_int, target_field_flattened_int], -1)
  #
  value_field = tf.map_fn(lambda pred_target_pair: calc_value(pred_target_pair[0], pred_target_pair[1]), preds_targets_zipped)
  #
  value_sums = tf.reduce_sum(value_field, axis=0)
  #
  true_positive = value_sums[0]
  false_positive = value_sums[1]
  false_negative = value_sums[2]
  true_negative = value_sums[3]
  #
  precision = true_positive / (true_positive + false_positive)
  recall = true_positive / (true_positive + false_negative)
  f1_score = 2 * precision * recall / (precision + recall)
  #
  return f1_score




# visualization of the uncertainties
# visualize_uncertainty(0)
def visualize_uncertainty(model, image_nr, num_particles=10, epoch_str='', target_field_mean=0.5, print_metrics=True):
  shutil.rmtree('visualizations/' + epoch_str + '_' + str(image_nr), ignore_errors=True)
  os.makedirs('visualizations/' + epoch_str + '_' + str(image_nr))
  files = os.listdir('data/ISBI2016_ISIC_Part1_Training_Data')
  file = files[image_nr]
  x_img = Image.open('data/ISBI2016_ISIC_Part1_Training_Data/' + file)
  x_img = x_img.resize([224,224])
  x_img = np.expand_dims(np.array(x_img), 0)
  x_img = preprocess_input(x_img)
  #
  nr = file[len('ISIC_'):-len('.jpg')]
  y_img = Image.open('data/ISBI2016_ISIC_Part1_Training_GroundTruth/ISIC_' + nr + '_Segmentation.png')
  y_img = np.array(np.array(y_img.resize([224,224])) / 255, dtype=np.int32)
  #
  print('make predictions!')
  pred_field_flattened, var_field_flattened, target_field_flattened = model_factory.calculate_flattened_predictions(model, x_img, y_img, target_field_mean=target_field_mean, num_particles=num_particles)
  #
  var_field_normalized_flattened = (var_field_flattened - var_field_flattened.min()) / (var_field_flattened.max() - var_field_flattened.min())
  error_field_flattened = np.abs(pred_field_flattened - target_field_flattened)
  #
  if print_metrics:
    f1_score = calculate_f1_score(pred_field_flattened, target_field_flattened)
    proportion_of_mistakes_in_top_10 = calculate_uncertainties(pred_field_flattened, var_field_normalized_flattened, target_field_flattened)
  # create overview
  print('create overview!')
  overview = np.zeros([224 + 20 + 224, 224 + 20 + 224, 3], dtype=np.float32)
  #
  for i in range(overview.shape[0]):
    for j in range(overview.shape[1]):
      if i >= 224 and i < 264 or j >= 224 and j < 264:
        overview[i][j] = np.ones([3], dtype=np.float32)
  #
  for it in range(pred_field_flattened.size):
    row = int(it / 224)
    col = it - 224 * row
    overview[row][col][1] = float(pred_field_flattened[it])
  pred_field_flattened_floats = np.array(pred_field_flattened, dtype=np.float32)
  pred_field = np.array(255 * pred_field_flattened_floats.reshape([224,224]), dtype=np.uint8)
  img_pred_field = Image.fromarray(pred_field, mode='L')
  img_pred_field.save('visualizations/' + str(image_nr) + '/pred_visualization.png')
  #
  for it in range(target_field_flattened.size):
    row = int(it / 224)
    col = it - 224 * row
    overview[row][col + 224 + 20][2] = float(target_field_flattened[it])
  target_field_flattened_floats = np.array(target_field_flattened, dtype=np.float32)
  target_field = np.array(255 * target_field_flattened_floats.reshape([224,224]), dtype=np.uint8)
  img_target_field = Image.fromarray(target_field, mode='L')
  img_target_field.save('visualizations/' + epoch_str + '_' + str(image_nr) + '/target_visualization.png')
  #
  for it in range(var_field_normalized_flattened.size):
    row = int(it / 224)
    col = it - 224 * row
    overview[row + 224 + 20][col][0] = float(var_field_normalized_flattened[it])
  var_field_flattened_floats = np.array(var_field_flattened, dtype=np.float32)
  var_field = np.array(255 * var_field_flattened_floats.reshape([224,224]), dtype=np.uint8)
  img_var_field = Image.fromarray(var_field, mode='L')
  img_var_field.save('visualizations/' + epoch_str + '_' + str(image_nr) + '/var_visualization.png')
  #
  for it in range(error_field_flattened.size):
    row = int(it / 224)
    col = it - 224 * row
    overview[row + 224 + 20][col + 224 + 20][0] = float(error_field_flattened[it])
  error_field_flattened_floats = np.array(error_field_flattened, dtype=np.float32)
  error_field = np.array(255 * error_field_flattened_floats.reshape([224,224]), dtype=np.uint8)
  img_error_field = Image.fromarray(error_field, mode='L')
  img_error_field.save('visualizations/' + epoch_str + '_' + str(image_nr) + '/error_visualization.png')
  print('overview created!')