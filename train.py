import argparse
import logging
import math
import os

import pandas as pd
import tensorflow as tf

from config import MEASUREMENTS_CSV_FILENAME
from data_augmentation import balanced_data_batch_generator
from neural_networks.neural_networks_common import get_empty_model, add_model_cmd_arg, load_model
from neural_networks import TrainValTensorBoardCallback

FORMAT = '%(asctime)-15s : %(message)s'


# run: python train.py -m nvidia -n 30 -b 50 -o true -r True -c 20

def train_model(model, args, train_data, valid_data, model_name, from_epoch=0):
	"""
	Train the model
	"""
	# Saves the model after every epoch.
	# quantity to monitor, verbosity i.e logging mode (0 or 1),
	# if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
	# mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
	# made based on either the maximization or the minimization of the monitored quantity. For val_acc,
	# this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
	# inferred from the name of the monitored quantity.
	model_trained_out_dir = 'trained_models\\{}'.format(model_name)
	model_path_base = 'trained_models\\{}\\{}-model-'.format(model_name, model_name)
	checkpoint_name = model_path_base + '{epoch:03d}.h5'
	checkpoint = tf.keras.callbacks.ModelCheckpoint(
		checkpoint_name,
		monitor='val_loss',
		verbose=0,
		save_best_only=args.save_best_only,
		mode='auto')

	tensorboard_callback = TrainValTensorBoardCallback.TrainValTensorBoard(
		log_dir=model_trained_out_dir + "\\logs",
		update_freq='epoch'
	)

	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate))

	# Fits the model on data generated batch-by-batch by a Python generator.

	# The generator is run in parallel to the model, for efficiency.
	# For instance, this allows you to do real-time data augmentation on images on CPU in
	# parallel to training your model on GPU.
	# so we reshape our data into their appropriate batches and train our model simulatenously
	model.fit_generator(
		generator=balanced_data_batch_generator(train_data, args.batch_size, True),
		steps_per_epoch=math.ceil(len(train_data) / (args.batch_size)),
		epochs=args.nb_epoch + from_epoch,
		max_queue_size=1,
		validation_data=balanced_data_batch_generator(valid_data, args.batch_size, False),
		validation_steps=len(valid_data),
		callbacks=[checkpoint, tensorboard_callback],
		verbose=2,
		initial_epoch=from_epoch
	)


# for command line args
def s2b(s):
	"""
	Converts a string to boolean value
	"""
	s = s.lower()
	return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def load_training_data():
	logging.info("Start loading data")
	train_data = get_data_frame(".\\out\\data")
	valid_data = get_data_frame(".\\out\\validation_data")

	# train_data, valid_data = train_test_split(frame, test_size=test_size, random_state=None)
	logging.info("Train data loaded (count: {})".format(len(train_data)))
	logging.info("Test data loaded (count: {})".format(len(valid_data)))
	return train_data, valid_data


def get_data_frame(directory):
	data = []
	for dir in os.walk(directory):
		if dir[0] == directory:
			continue

		measurements_file = os.path.join(dir[0], MEASUREMENTS_CSV_FILENAME)
		loaded_data = pd.read_csv(measurements_file, sep=',', decimal='.', usecols=[0, 1, 2, 3], header=None,
		                          names=['frame', 'steering', 'throttle', 'brake'])
		loaded_data['data_dir'] = dir[0]
		data.append(loaded_data)
	frame = pd.concat(data, axis=0, ignore_index=True)
	return frame


def main():
	parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
	parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=10)
	parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=150)
	parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
	parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
	parser.add_argument('-r', '--resume', help='resume training from checkpoint', type=s2b, default='false')
	parser.add_argument('-c', '--checkpoint', help='checkpoint number to resume training from', type=int, default=0)
	parser.add_argument('-f', '--fine-tuning', help='train networks with fine tuning', type=s2b, default='false')
	add_model_cmd_arg(parser)
	args = parser.parse_args()

	# print parameters
	print('-' * 30)
	print('Parameters')
	print('-' * 30)
	for key, value in vars(args).items():
		print('{:<20} := {}'.format(key, value))
	print('-' * 30)

	# load data
	data = load_training_data()
	# build model
	logging.info("Loading neural network model: {}".format(args.model))

	if not args.resume:
		model = get_empty_model(args.model, args.fine_tuning)
	else:
		model = load_model(args.model, args.checkpoint, args.fine_tuning)

	# train model on data, it saves as model.h5
	train_model(model, args, *data, args.model, args.checkpoint)


if __name__ == '__main__':
	logging.basicConfig(format=FORMAT)
	logging.getLogger().setLevel(logging.INFO)
	main()
