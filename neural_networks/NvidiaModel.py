import tensorflow as tf

from data_augmentation import INPUT_SHAPE
from neural_networks.ModelBase import ModelBase


class NvidiaModel(ModelBase):
	def model(self, fine_tuning):
		model = tf.keras.Sequential()

		model.add(
			tf.keras.layers.Conv2D(
				input_shape=INPUT_SHAPE,
				filters=24,
				kernel_size=[5, 5],
				padding='same',
				strides=[2, 2],
				activation='relu')
		)
		model.add(
			tf.keras.layers.Conv2D(
				filters=36,
				kernel_size=[5, 5],
				padding='same',
				strides=[2, 2],
				activation='relu')
		)
		model.add(
			tf.keras.layers.Conv2D(
				filters=48,
				kernel_size=[5, 5],
				padding='same',
				strides=[2, 2],
				activation='relu')
		)
		model.add(
			tf.keras.layers.Conv2D(
				filters=64,
				kernel_size=[3, 3],
				padding='same',
				strides=[1, 1],
				activation='relu')
		)
		model.add(
			tf.keras.layers.Conv2D(
				filters=64,
				kernel_size=[3, 3],
				padding='same',
				strides=[1, 1],
				activation='relu')
		)

		model.add(
			tf.keras.layers.Flatten()
		)
		model.add(
			tf.keras.layers.Dense(
				1164,
				activation='relu'
			)
		)
		model.add(
			tf.keras.layers.Dense(
				100,
				activation='relu'
			)
		)
		model.add(
			tf.keras.layers.Dense(
				50,
				activation='relu'
			))
		model.add(
			tf.keras.layers.Dense(
				10,
				activation='relu'
			)
		)
		model.add(
			tf.keras.layers.Dense(2)
		)
		model.summary()
		return model
