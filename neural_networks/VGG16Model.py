import tensorflow as tf
from config import *
from data_augmentation import INPUT_SHAPE
from neural_networks.ModelBase import ModelBase


class VGG16Model(ModelBase):
	def model(self):
		model_vgg16_conv = tf.keras.applications.VGG16(
			weights='imagenet',
			include_top=False,
			input_shape=INPUT_SHAPE
		)

		ret_model = tf.keras.models.Sequential()
		ret_model.add(model_vgg16_conv)
		ret_model.add(
			tf.keras.layers.Flatten()
		)
		ret_model.add(
			tf.keras.layers.Dense(
				1164,
				activation='relu'
			)
		)
		ret_model.add(
			tf.keras.layers.Dense(
				100,
				activation='relu'
			)
		)
		ret_model.add(
			tf.keras.layers.Dense(
				50,
				activation='relu'
			)
		)
		ret_model.add(
			tf.keras.layers.Dense(
				10,
				activation='relu'
			)
		)
		ret_model.add(
			tf.keras.layers.Dense(2)
		)

		ret_model.summary()
		return ret_model