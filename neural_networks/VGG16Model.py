import tensorflow as tf
from config import *
from data_augmentation import INPUT_SHAPE


class VGG16Model:
	def model(self):
		model_vgg16_conv = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
		model_vgg16_conv.summary()

		input = tf.keras.layers.Input(shape=INPUT_SHAPE, name='image_input')
		# Use the generated model
		output_vgg16_conv = model_vgg16_conv(input)

		# Add the fully-connected layers
		x = tf.keras.layers.Flatten(name='flatten')(output_vgg16_conv)
		x = tf.keras.layers.Dense(1164, activation='relu', name='fc1')(x)
		x = tf.keras.layers.Dense(100, activation='relu', name='fc2')(x)
		x = tf.keras.layers.Dense(50, activation='relu', name='fc3')(x)
		x = tf.keras.layers.Dense(10, activation='relu', name='fc4')(x)
		x = tf.keras.layers.Dense(2, activation='linear', name='output')(x)

		my_model = tf.keras.Model(inputs=input, outputs=x)

		for layer in my_model.layers[:15]:
			layer.trainable = False

		my_model.summary()
		return my_model

	@staticmethod
	def compile_model(model):
		adam = tf.keras.optimizers.Adam(LEARNING_RATE)
		model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=adam)
		return model

	@staticmethod
	def load_weights(model, filename):
		model.load_weights(filename)
		return model