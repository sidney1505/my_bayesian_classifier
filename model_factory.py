import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import time, scipy, os
from scipy import stats
import code # code.interact(local=dict(globals(), **locals()))

# own librabries
import my_callbacks

# MODEL definition
def create_model(config):
    # the second probabilistic part of the model
    dense_input    = Input(shape=(None, config['height'], config['width'], config['num_channels']))
    dense_input_reshaped = tf.keras.layers.Reshape([-1, config['height'] * config['width'] * config['num_channels']])(dense_input)
    if config['is_probabilistic']:
        # brings uncertainty into the model weights
        dense_output = tfp.layers.DenseFlipout(config['num_channels'], activation=tf.nn.leaky_relu)(dense_input_reshaped)
        dense_output = tfp.layers.DenseFlipout(config['num_channels'], activation=tf.nn.leaky_relu)(dense_output)
        dense_output = tf.keras.layers.Dropout(config['dropout_rate'])(dense_output)
        dense_output = tfp.layers.DenseFlipout(config['num_classes'])(dense_output)
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
        loss = lambda labels, labels_distribution: -tf.reduce_mean(labels_distribution.log_prob(labels)) + sum(full_model.losses) / config['scaling_factor']
    #
    if config['optimizer_name'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config['learning_rate'])
    if config['optimizer_name'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
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
def train_model(model, x_val, y_val, config, x_train=None, y_train=None, target_field_mean=None): 
    #
    if config['load_data_with_numpy']:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
        datagen.fit(x_train) # only possible, if x_train is    precomputed
        data_loader = datagen.flow(x_train, y_train, batch_size=config['batch_size'])
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            preprocessing_function=preprocess_input,
            data_format='channels_first')
        data_loader = datagen.flow_from_directory(config['dataset_dir'], batch_size=config['batch_size'], target_size=(224, 224)),
    # datagen.fit(x_train) # only possible, if x_train is    precomputed

    # own callbacks
    callbacks = []
    if config['num_classes'] == 2:
        if target_field_mean == None:
            if config['load_data_with_numpy']:
                target_field_mean = y_train.mean()
            else:
                target_field_mean = y_val.mean() # TODO this way is very dirty
        f1andUncertaintiesCallback = my_callbacks.F1andUncertaintiesCallback(validation_data=(x_val, y_val), config=config, target_field_mean=target_field_mean)
        callbacks.append(f1andUncertaintiesCallback)
    elif config['num_classes'] >= 2:
        accuracyAndUncertaintiesCallback = my_callbacks.AccuracyAndUncertaintiesCallback(validation_data=(x_val, y_val), config=config)
        callbacks.append(accuracyAndUncertaintiesCallback)

    #
    if config['model_name'] == None:
        date_values = [str(x) for x in time.gmtime()]
        config['model_name'] = '_'.join(date_values)
        os.makedirs("models/" + config['model_name'])
    filepath="models/" + config['model_name'] + "/checkpoint-{epoch:02d}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='chosen_acc', verbose=1, save_best_only=True, mode='max', period=config['validation_interval'])
    callbacks.append(checkpoint)
    #
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='chosen_acc', patience=config['num_epochs'] / 4, restore_best_weights=True, mode='max')
    callbacks.append(early_stopper)
    # lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss')
    #
    print('fit model!')
    history = model.fit_generator( \
        data_loader, \
        epochs=config['num_epochs'],\
        validation_data=datagen.flow(x_val, y_val, batch_size=config['batch_size']),\
        callbacks=callbacks \
    )
    #
    print('save model!')
    model.save_weights("models/" + config['model_name'] + "/final")
    # model.save('models/negative_loglikelihood_model.h5') # tfp not compatible until now as it seems



# calculate_flattened_predictions(model, test_images, test_labelss, train_labels.mean())
def calculate_flattened_predictions(model, x, y, config, target_field_mean=0.5, num_particles=10):
    print('calculate predictions!')
    pred_field_flattened_list, var_field_flattened_list, target_field_flattened_list = [], [], []
    for split in range(config['num_splits']):
        target_field = y[config['num_splits'] * split: config['num_splits'] * (split + 1)]
        x_input = x[config['num_splits'] * split: config['num_splits'] * (split + 1)]
        prediction_list = [model.predict(x_input) for i in range(config['num_particles'])]
        target_field_flattened = target_field.flatten()
        if config['num_classes'] > 2:
            # TODO doesn't work for segmentation in that form
            prediction_field = np.zeros([prediction_list[0].size, config['num_classes']])
            for prediction in prediction_list:
                prediction_flattened = prediction.flatten()
                for idx in range(prediction_flattened.size):
                    prediction_field[idx][prediction_flattened[idx]] += 1
            pred_field_flattened = np.argmax(prediction_field, axis=-1)
            prediction_field_transposed = np.transpose(prediction_field)
            var_field_flattened = stats.entropy(prediction_field_transposed)
        elif config['num_classes'] == 2:
            predictions_field = np.array(prediction_list)
            mean_field = np.mean(predictions_field, axis=0) # TODO will this cause balancing problems???
            var_field = np.var(predictions_field, axis=0)
            #
            if not config['is_probabilistic']: # TODO isn't there a more elegant way???
                 mean_field = np.squeeze(mean_field)
                 mean_field = np.transpose(mean_field)
                 mean_field = mean_field[0]
                 mean_field = np.transpose(mean_field)
                 var_field = np.squeeze(var_field)
                 var_field = np.transpose(var_field)
                 var_field = var_field[0]
                 var_field = np.transpose(var_field)
            #
            var_field_flattened = var_field.flatten()
            # TODO normalizing with the mean can't be the proper way to this!
            mean_field_flattened = mean_field.flatten()
            quantile = np.quantile(mean_field_flattened, 1 - target_field_mean)
            pred_field_flattened = np.array(list(map(lambda idx: mean_field_flattened[idx] >= quantile, range(mean_field_flattened.size))))        
        #
        pred_field_flattened_list.append(pred_field_flattened)
        var_field_flattened_list.append(var_field_flattened)
        target_field_flattened_list.append(target_field_flattened)
    pred_field_flattened = np.concatenate(pred_field_flattened_list, axis=0)
    var_field_flattened = np.concatenate(var_field_flattened_list, axis=0)
    target_field_flattened = np.concatenate(target_field_flattened_list, axis=0)
    return pred_field_flattened, var_field_flattened, target_field_flattened
