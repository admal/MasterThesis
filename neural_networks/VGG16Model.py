import tensorflow as tf
from data_augmentation import INPUT_SHAPE
from neural_networks.ModelBase import ModelBase

class VGG16Model(ModelBase):
	def model(self, fine_tuning):
		model_vgg16_conv = tf.keras.applications.VGG16(
			weights='imagenet',
			include_top=False,
			input_shape=INPUT_SHAPE
		)
		model_vgg16_conv.trainable = True
		set_trainable = False
		for layer in model_vgg16_conv.layers:
			if layer.name in ['block5_conv1', 'block4_conv1']:
				set_trainable = True
			if set_trainable:
				layer.trainable = True
			else:
				layer.trainable = False

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
			tf.keras.layers.Dropout(0.2)
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