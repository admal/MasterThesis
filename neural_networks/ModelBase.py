import tensorflow as tf

from config import LEARNING_RATE


class ModelBase:
	# TODO: TO REMOVE I THINK
	@staticmethod
	def compile_model(model):
		adam = tf.keras.optimizers.Adam(LEARNING_RATE)
		model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=adam)
		return model

	@staticmethod
	def load_weights(model, filename):
		model.load_weights(filename)
		return model
