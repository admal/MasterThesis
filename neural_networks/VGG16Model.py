import tensorflow as tf


class AlexModel:
	def tmp(self):
		inputs = tf.keras.Input(shape=(None, None, 3), name='image_input')
		vgg16_model = tf.keras.applications.VGG16(wights='imagenet', include_top=False)

		output_vgg16_conv = vgg16_model(inputs)

		x = tf.keras.layers.Flatten(name='flatten')(output_vgg16_conv)
		x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
		x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
		x = tf.keras.layers.Dense(2, activation='softmax', name='predictions')(x)
