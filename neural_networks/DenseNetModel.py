from data_augmentation import INPUT_SHAPE
from neural_networks.ModelBase import ModelBase
import tensorflow as tf


class DenseNetModel(ModelBase):
	def model(self):
		densenet = tf.keras.applications.densenet.DenseNet121(
			weights='imagenet',
			include_top=False,
			input_shape=INPUT_SHAPE
		)

		ret_model = tf.keras.models.Sequential()
		ret_model.add(densenet)
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
			tf.keras.layers.Dropout(0.5)
		)
		ret_model.add(
			tf.keras.layers.Dense(
				100,
				activation='relu'
			)
		)
		ret_model.add(
			tf.keras.layers.Dropout(0.5)
		)
		ret_model.add(
			tf.keras.layers.Dense(
				50,
				activation='relu'
			)
		)
		ret_model.add(
			tf.keras.layers.Dropout(0.5)
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
