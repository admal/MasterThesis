import argparse
import tensorflow as tf
import os
import csv
import logging

from data_augmentation import batch_generator
from neural_networks.VGG16Model import VGG16Model
from config import MEASUREMENTS_CSV_FILENAME
from sklearn.model_selection import train_test_split #to split out training and testing data
FORMAT = '%(asctime)-15s : %(message)s'

def train_model(model, args, train_data, valid_data):
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
	checkpoint = tf.keras.callbacks.ModelCheckpoint('trained_models\\model-{epoch:03d}.h5',
	                                                monitor='val_loss',
	                                                verbose=0,
	                                                save_best_only=args.save_best_only,
	                                                mode='auto')

	# calculate the difference between expected steering angle and actual steering angle
	# square the difference
	# add up all those differences for as many data points as we have
	# divide by the number of them
	# that value is our mean squared error! this is what we want to minimize via
	# gradient descent
	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate))

	# Fits the model on data generated batch-by-batch by a Python generator.

	# The generator is run in parallel to the model, for efficiency.
	# For instance, this allows you to do real-time data augmentation on images on CPU in
	# parallel to training your model on GPU.
	# so we reshape our data into their appropriate batches and train our model simulatenously
	model.fit_generator(batch_generator(train_data, args.batch_size, True),
	                    args.samples_per_epoch,
	                    args.nb_epoch,
	                    max_queue_size=1,
	                    validation_data=batch_generator(valid_data, args.batch_size, False),
	                    # nb_val_samples=len(valid_data),
	                    validation_steps=len(valid_data),
	                    callbacks=[checkpoint],
	                    verbose=1)


# for command line args
def s2b(s):
	"""
	Converts a string to boolean value
	"""
	s = s.lower()
	return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def load_data():
	directory = ".\\out"
	logging.info("Start loading data")
	data = []
	for dir in os.walk(directory):
		if dir[0] == directory:
			continue
		measurements_file = os.path.join(dir[0], MEASUREMENTS_CSV_FILENAME)
		# print(measurements_file)

		with open(measurements_file) as csvfile:
			rows = csv.reader(csvfile)
			for row in rows:
				data.append((dir[0], int(row[0]), float(row[1]), float(row[2]), float(row[3])))

	train_data, valid_data= train_test_split(data, test_size=0.1, random_state=None)
	logging.info("Train data loaded (count: {})".format(len(train_data)))
	logging.info("Test data loaded (count: {})".format(len(valid_data)))
	return train_data, valid_data


def main():
	parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
	# parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='data')
	parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
	parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
	parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=10)
	parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=100)
	parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=40)
	parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
	parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
	args = parser.parse_args()

	# print parameters
	print('-' * 30)
	print('Parameters')
	print('-' * 30)
	for key, value in vars(args).items():
		print('{:<20} := {}'.format(key, value))
	print('-' * 30)

	# load data
	data = load_data()
	# build model
	model = VGG16Model().model()
	compiled_model = VGG16Model.compile_model(model)
	# train model on data, it saves as model.h5
	train_model(compiled_model, args, *data)

if __name__ == '__main__':
	logging.basicConfig(format=FORMAT)
	logging.getLogger().setLevel(logging.INFO)
	main()
