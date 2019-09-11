from data_augmentation import INPUT_SHAPE
from neural_networks.ModelBase import ModelBase
import tensorflow as tf


class ResNet50Model(ModelBase):
	def model(self, fine_tuning):
		resnet = tf.keras.applications.ResNet50(
			weights='imagenet',
			include_top=False,
			input_shape=INPUT_SHAPE,
			pooling='avg'
		)

		resnet.trainable = False
		if fine_tuning:
			half = int(len(resnet.layers) / 2)
			for layer in resnet.layers[-half:]:
				layer.trainable = True

		ret_model = tf.keras.models.Sequential()
		ret_model.add(resnet)
		ret_model.add(
			tf.keras.layers.Dense(2, activity_regularizer=tf.keras.regularizers.l1(0.01))
		)
		ret_model.summary()
		return ret_model
