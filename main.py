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
PREPROZESS_DATA = True
DATASET_DIR = 'data'
num_hidden_units = 128

if PREPROZESS_DATA:
	positive_images = []
	negative_images = []
	#
	print('load images')
	for it, file in enumerate(os.listdir(DATASET_DIR + '/positive_samples')):
		img = Image.open(DATASET_DIR + '/positive_samples/' + file)
		img = img.resize([224,224])
		positive_images.append(img)

	for it, file in enumerate(os.listdir(DATASET_DIR + '/negative_samples')):
		img = Image.open(DATASET_DIR + '/negative_samples/' + file)
		img = img.resize([224,224])
		negative_images.append(img)
	#

	print('preprocess data')
	rust_train = positive_images[:-int(0.1 * len(positive_images))]
	rust_test = positive_images[-int(0.1 * len(positive_images)):]

	norust_train = negative_images[:-int(0.1 * len(negative_images))]
	norust_test = negative_images[-int(0.1 * len(negative_images)):]

	train_images = np.concatenate([np.stack(rust_train), np.stack(norust_train)])
	test_images = np.concatenate([np.stack(rust_test), np.stack(norust_test)])

	train_labels = np.concatenate([np.zeros(len(rust_train), dtype=np.int32), np.ones(len(norust_train), dtype=np.int32)])
	test_labels = np.concatenate([np.zeros(len(rust_test), dtype=np.int32), np.ones(len(norust_test), dtype=np.int32)])

	train_images = preprocess_input(train_images)
	test_images = preprocess_input(test_images)
	#
	print('extract features with pretrained inception net')
	# resnet = ResNet50(include_top=False, weights='imagenet')
	# resnet.predict(np.zeros([5,224,224,3]))
	# code.interact(local=dict(globals(), **locals()))
	train_images = np.transpose(train_images, [0,3,1,2])
	test_images = np.transpose(test_images, [0,3,1,2])
	train_images = InceptionResNetV2(include_top=False, weights='imagenet').predict(train_images)
	test_images = InceptionResNetV2(include_top=False, weights='imagenet').predict(test_images)
	# TODO store these + the groundtruth somewhere in order to avoid long computation times!!!
	# code.interact(local=dict(globals(), **locals()))
	np.savez(open(DATASET_DIR + '/preprocessed.npz','wb'), train_images=train_images, test_images=test_images, train_labels=train_labels, test_labels=test_labels)
else:
	print('load data!')
	data = np.load(open(DATASET_DIR + '/preprocessed.npz','rb'))
	train_images = data['train_images']
	test_images = data['test_images']
	train_labels = data['train_labels']
	test_labels = data['test_labels']

print('define keras model')
layers = [ \
	tf.keras.layers.Reshape((-1,5 * 5 * 1536)), \
	tfp.layers.DenseFlipout(2 * num_hidden_units, activation=tf.nn.leaky_relu), \
	tfp.layers.DenseFlipout(num_hidden_units, activation=tf.nn.leaky_relu), \
	tfp.layers.DenseFlipout(num_hidden_units, activation=tf.nn.leaky_relu), \
	tfp.layers.DenseFlipout(2, activation=tf.nn.leaky_relu), \
	tfp.layers.DistributionLambda(lambda t: tfd.Categorical(logits=t)) \
]
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.1)
model = tf.keras.Sequential(layers)
negloglik = lambda labels_distribution, labels: -tf.reduce_mean(labels.log_prob(labels_distribution))
# code.interact(local=dict(globals(), **locals()))

#
print('compile model')
model.compile(optimizer='adam', \
              loss=negloglik, \
              metrics=['accuracy'])


# define the callbacks used
num_epochs = 5000
values = [str(x) for x in time.gmtime()]
model_name = '_'.join(values)
os.makedirs("models/" + model_name)
filepath="models/" + model_name + "/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint =  tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=num_epochs / 10)
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss')
# own callbacks
target_field_mean = train_labels.mean()
#f1score_callback = my_callbacks.F1ScoreCallback(validation_data=(test_images, test_labels), interval=1, target_field_mean=target_field_mean)
#uncertainties_callback = my_callbacks.UncertaintiesCallback(validation_data=(test_images, test_labels), interval=1, target_field_mean=target_field_mean)
# visualization_callback = my_callbacks.VisualizationCallback(validation_data=(x_val, y_val_squeezed), interval=1, target_field_mean=target_field_mean)
f1andUncertaintiesCallback = my_callbacks.F1andUncertaintiesCallback(validation_data=(test_images, test_labels), interval=1, target_field_mean=target_field_mean)
#
print('fit model!')
history = model.fit( \
 x=train_images,\
 y=train_labels,\
 batch_size=16,\
 epochs=num_epochs,\
 validation_data=(test_images, test_labels),\
 #callbacks=[early_stopper, checkpoint, lr_decay, f1score_callback, uncertainties_callback, visualization_callback] \
 callbacks=[early_stopper, checkpoint, lr_decay, f1andUncertaintiesCallback] \
)
code.interact(local=dict(globals(), **locals()))
#
print('evaluate model')
model.evaluate(test_images, test_labels, verbose=2)

print('Enter commands!')
code.interact(local=dict(globals(), **locals()))
# example skript
x_train, y_train, x_val, y_val = dataloader.preprocess_data()
target_field_mean = y_train.mean()
# create model
model = model_factory.create_model()
# either load model
example_path = 'models/2019_12_10_16_39_7_1_344_0/checkpoint-06-244.14.hdf5'
model.load_weights(example_path)
# or train model (or both)
model_factory.train_model(model, x_train, y_train, x_val, y_val, num_epochs=500, initial_epoch=30, target_field_mean=target_field_mean)

my_utils.visualize_uncertainty(model, 850, target_field_mean=target_field_mean)

pred_field_flattened, var_field_flattened, target_field_flattened = model_factory.calculate_flattened_predictions(model, x_val, y_val, target_field_mean)
my_utils.calculate_uncertainties(pred_field_flattened, var_field_flattened, target_field_flattened)
my_utils.calculate_f1_score(pred_field_flattened, target_field_flattened)