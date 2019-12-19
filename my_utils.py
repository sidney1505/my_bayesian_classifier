import tensorflow as tf
import numpy as np
import code # code.interact(local=dict(globals(), **locals()))



# own version of calculating the F1 score
def calculate_f1_score(pred_field_flattened, target_field_flattened, verbose=True):
  if not pred_field_flattened.size == target_field_flattened.size:
    print("sizes don't match in calculate_f1_score")
    code.interact(local=dict(globals(), **locals()))
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
  return precision, recall, f1_score



# own version of calculating the accuracy
def calculate_accuracy(pred_field_flattened, target_field_flattened, verbose=True):
  if not pred_field_flattened.size == target_field_flattened.size:
    print("sizes don't match in calculate_f1_score")
    code.interact(local=dict(globals(), **locals()))
  num_correct = np.sum(pred_field_flattened == target_field_flattened)
  accuracy = num_correct / pred_field_flattened.size
  return accuracy



# gives some intuition for the correlation of the errors and the uncertainties
def calculate_uncertainties(pred_field_flattened, var_field_flattened, target_field_flattened, verbose=True):
  if not pred_field_flattened.size == var_field_flattened.size == target_field_flattened.size:
    print("sizes don't match in check_uncertainties")
    code.interact(local=dict(globals(), **locals()))
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
    print('avg_proportion_of_mistakes_in_middle_part: ' + str(proportion_of_mistakes_in_middle_part / 8))
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