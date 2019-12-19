import code # code.interact(local=dict(globals(), **locals()))
from tensorflow.keras.callbacks import Callback
import numpy as np

# own librabries
import model_factory, my_utils



class F1andUncertaintiesCallback(Callback):
    def __init__(self, validation_data, config, target_field_mean=0.5):
        super(Callback, self).__init__()

        self.X_val, self.y_val = validation_data
        assert target_field_mean <= 0.5, "f1 score is not a good metric for more then 50% positive samples!!!"
        self.target_field_mean = target_field_mean
        self.config = config

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.config['validation_interval'] == 0:
            print('epoch: ' + str(epoch))
            pred_field_flattened, var_field_flattened, target_field_flattened = model_factory.calculate_flattened_predictions(self.model, self.X_val, self.y_val, self.config, self.target_field_mean)
            precision, recall, f1_score = my_utils.calculate_f1_score(pred_field_flattened, target_field_flattened)
            logs['precision'] = precision
            logs['recall'] = recall
            logs['f1_score'] = f1_score
            logs['chosen_acc'] = f1_score
            print('avg_target: ' + str(self.target_field_mean))
            print('avg_prediction: ' + str(np.mean(pred_field_flattened)))
            print('avg_uncertainty: ' + str(np.mean(var_field_flattened)))
            uncertainty_score = my_utils.calculate_uncertainties(pred_field_flattened, var_field_flattened, target_field_flattened)
            logs['uncertainty_score'] = uncertainty_score



class AccuracyAndUncertaintiesCallback(Callback):
    def __init__(self, validation_data, config):
        super(Callback, self).__init__()

        self.config = config
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.config['validation_interval'] == 0:
            print('epoch: ' + str(epoch))
            pred_field_flattened, var_field_flattened, target_field_flattened = model_factory.calculate_flattened_predictions(self.model, self.X_val, self.y_val, self.config)
            accuracy = my_utils.calculate_accuracy(pred_field_flattened, target_field_flattened)
            logs['accuracy'] = accuracy
            logs['chosen_acc'] = accuracy
            print('avg_uncertainty: ' + str(np.mean(var_field_flattened)))
            #
            prediction_class_numbers = list(map(lambda class_nr: np.sum(pred_field_flattened == class_nr), range(self.config['num_classes'])))
            prediction_class_distribution = np.array(prediction_class_numbers) / pred_field_flattened.size
            print('prediction_class_distribution: ' + str(prediction_class_distribution))
            #
            target_class_numbers = list(map(lambda class_nr: np.sum(target_field_flattened == class_nr), range(self.config['num_classes'])))
            target_class_distribution = np.array(target_class_numbers) / target_field_flattened.size
            print('target_class_distribution: ' + str(target_class_distribution))
            #
            uncertainty_score = my_utils.calculate_uncertainties(pred_field_flattened, var_field_flattened, target_field_flattened)
            logs['uncertainty_score'] = uncertainty_score

