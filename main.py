import argparse
import code # code.interact(local=dict(globals(), **locals()))

# own librabries
import model_factory, dataloader, my_utils, my_callbacks


parser = argparse.ArgumentParser(description='Process some integers.')

# default setting is for binary isic2019 classification
parser.add_argument('--finetune_feature_extractor', type=bool, default=True, help='Whether the feature extracting CNN should be frozen or not.')
parser.add_argument('--num_classes', type=int, default=2, help='The number of classes present in the data set to learn.')
parser.add_argument('--is_probabilistic', type=bool, default=True, help='Whether one wants to train the model in a probabilistic way or not.')
parser.add_argument('--dataset_dir', default='data/isic2019', help='The directory of the data set.')
parser.add_argument('--validation_interval', type=int, default=1, help='The frequency the model is validated (relevant for very small datasets to reduce overhead computation)')
parser.add_argument('--val_split', type=float, default=0.1, help='The proportion of the validation split while training.')
parser.add_argument('--optimizer_name', default='adam', help='The optimizer, that is used out of [adam, rmsprob]')
parser.add_argument('--loss_name', default='elbo', help='The loss, that is used out of [negloglik, elbo]')
parser.add_argument('--num_channels', type=int, default=1536, help='The number of channels, that are extracted by the CNN.')
parser.add_argument('--height', type=int, default=5, help='The height of the extracted feature map.')
parser.add_argument('--width', type=int, default=5, help='The width of the extracted feature map.')
parser.add_argument('--model_name', default=None, help='The name of the model - if none is given it will use the current date instead.')
parser.add_argument('--batch_size', type=int, default=25, help='The batch size of the mini batches, that is used. Choose as high as possible in order to stabilize variational training via flipout!')
parser.add_argument('--dropout_rate', type=float, default=0.8, help='The dropout rate used in the last layer before the logits-layer!')
parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning rate used for training.')
parser.add_argument('--num_epochs', type=int, default=True, help='The number of epochs the model is trained. Early stopping is implemented, so choosing a high number does not hurt the performance!')
parser.add_argument('--num_particles', type=int, default=10, help='The number of particles used for inference. The higher, the more stable the predcition and the better the uncertainty estimates, but its very time consuming!')
parser.add_argument('--steps_per_epoch', type=int, default=1014, help='Number of steps per epoch - used to scale second part of the ELBO loss correctly')
parser.add_argument('--num_splits', type=int, default=10, help='Number of splits used for validation in order to fit computation into GPU memory and RAM.')

args = parser.parse_args()

# the hyperparameters are tracked over the whole program via the config dict in order to keep them synchronized
config = {
	'finetune_feature_extractor' : args['finetune_feature_extractor'],
	'num_classes' : args['num_classes'],
	'is_probabilistic' : args['is_probabilistic'],
	'dataset_dir' : args['dataset_dir'],
	'validation_interval' : args['validation_interval'],
	'val_split' : args['val_split'].,
	'optimizer_name' : args['optimizer_name'],
	'loss_name' : args['loss_name'],
	'num_channels' : args['num_channels'],
	'height' : args['height'],
	'width' : args['width'],
	'model_name' : args['model_name'],
	'batch_size' : args['batch_size'],
	'dropout_rate' : args['dropout_rate0'].,
	'learning_rate' : args['learning_rate0'].,
	'num_epochs' : args['num_epochs'],
	'num_particles' : args['num_particles'],
	'scaling_factor' : args['scaling_factor'],
	'num_splits' : args['num_splits']
}

print('enter your command!')
code.interact(local=dict(globals(), **locals()))
# possible example run

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