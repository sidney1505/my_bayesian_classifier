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

rust_images = []
norust_images = []
#
print('load images')
for file in os.listdir('download/rust'):
	img = Image.open('download/rust/' + file)
	img = img.resize([224,224])
	rust_images.append(img)

for file in os.listdir('download/norust'):
	img = Image.open('download/norust/' + file)
	img = img.resize([224,224])
	norust_images.append(img)
#
print('preprocess data')
rust_train = rust_images[:-int(0.2 * len(rust_images))]
rust_test = rust_images[:-int(0.2 * len(rust_images))]

norust_train = norust_images[:-int(0.2 * len(norust_images))]
norust_test = norust_images[-int(0.2 * len(norust_images)):]

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
train_images = InceptionResNetV2(include_top=False, weights='imagenet').predict(train_images)
test_images = InceptionResNetV2(include_top=False, weights='imagenet').predict(test_images)

print('define keras model')
code.interact(local=dict(globals(), **locals()))
layers = [ \
	tf.keras.layers.Reshape((-1,5 * 5 * 1536)), \
	tf.keras.layers.Dense(1024, activation=tf.nn.relu), \
	tf.keras.layers.Dropout(0.5), \
	tfp.layers.DistributionLambda(lambda t: tfd.Categorical(logits=t)) \
]
model = tf.keras.Sequential(layers)
negloglik = lambda labels_distribution, labels: -tf.reduce_mean(labels.log_prob(labels_distribution))
# code.interact(local=dict(globals(), **locals()))

#
print('compile model')
model.compile(optimizer='adam', \
              loss=negloglik, \
              metrics=['accuracy'])

print('train model')
code.interact(local=dict(globals(), **locals()))
model.fit(train_images, train_labels, epochs=20)
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