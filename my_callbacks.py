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
from tensorflow.keras.callbacks import Callback

# own librabries
import model_factory, dataloader, my_utils


# custom f1 score callback
class F1ScoreCallback(Callback):
    def __init__(self, validation_data=(), interval=1, target_field_mean=0.5):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.target_field_mean = target_field_mean

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            pred_field_flattened, var_field_flattened, target_field_flattened = model_factory.calculate_flattened_predictions(self.model, self.X_val, self.y_val, self.target_field_mean)
            score = my_utils.calculate_f1_score(pred_field_flattened, target_field_flattened)

# custom f1 score callback
class UncertaintiesCallback(Callback):
    def __init__(self, validation_data=(), interval=1, target_field_mean=0.5):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.target_field_mean = target_field_mean

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            pred_field_flattened, var_field_flattened, target_field_flattened = model_factory.calculate_flattened_predictions(self.model, self.X_val, self.y_val, self.target_field_mean)
            score = my_utils.calculate_uncertainties(pred_field_flattened, var_field_flattened, target_field_flattened)

class F1andUncertaintiesCallback(Callback):
    def __init__(self, validation_data=(), interval=10, target_field_mean=0.5):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        assert target_field_mean <= 0.5, "f1 score is not a good metric for more then 50% positive samples!!!"
        self.target_field_mean = target_field_mean

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            print('epoch: ' + str(epoch))
            pred_field_flattened, var_field_flattened, target_field_flattened = model_factory.calculate_flattened_predictions(self.model, self.X_val, self.y_val, self.target_field_mean)
            precision, recall, f1_score = my_utils.calculate_f1_score(pred_field_flattened, target_field_flattened)
            logs['precision'] = precision
            logs['recall'] = recall
            logs['f1_score'] = f1_score
            print('avg_target: ' + str(self.target_field_mean))
            print('avg_prediction: ' + str(np.mean(pred_field_flattened)))
            print('avg_uncertainty: ' + str(np.mean(var_field_flattened)))
            uncertainty_score = my_utils.calculate_uncertainties(pred_field_flattened, var_field_flattened, target_field_flattened)
            logs['uncertainty_score'] = uncertainty_score


# custom f1 score callback
class VisualizationCallback(Callback):
    def __init__(self, validation_data=(), interval=1, target_field_mean=0.5):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.target_field_mean = target_field_mean

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            my_utils.visualize_uncertainty(self.model, 850, num_particles=10, epoch_str=str(epoch))

