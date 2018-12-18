import numpy as np
import cv2
import os

from config import CENTER_CAMERA_NAME, LEFT_CAMERA_NAME, RIGHT_CAMERA_NAME

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 64, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

RANDOM_IMAGE_FLIP_PROBABILITY = 0.5
RANDOM_IMAGE_NOISE_PROBABILITY = 0.2
IMAGE_NOISE_TYPE = 's&p'


# Source of the code is based on an excelent piece code from stackoverflow
# http://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv

def noise_generator(noise_type, image):
	"""
	Generate noise to a given Image based on required noise type

	Input parameters:
		image: ndarray (input image data. It will be converted to float)

		noise_type: string
			'gauss'        Gaussian-distrituion based noise
			'poission'     Poission-distribution based noise
			's&p'          Salt and Pepper noise, 0 or 1
			'speckle'      Multiplicative noise using out = image + n*image
						   where n is uniform noise with specified mean & variance
	"""
	row, col, ch = image.shape
	if noise_type == "gauss":
		mean = 0.0
		var = 0.01
		sigma = var ** 0.5
		gauss = np.array(image.shape)
		gauss = np.random.normal(mean, sigma, (row, col, ch))
		gauss = gauss.reshape(row, col, ch)
		noisy = image + gauss
		return noisy.astype('uint8')
	elif noise_type == "s&p":
		s_vs_p = 0.5
		amount = 0.004
		out = image
		# Generate Salt '1' noise
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt))
		          for i in image.shape]
		out[coords] = 255
		# Generate Pepper '0' noise
		num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper))
		          for i in image.shape]
		out[coords] = 0
		return out
	elif noise_type == "poisson":
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy
	elif noise_type == "speckle":
		gauss = np.random.randn(row, col, ch)
		gauss = gauss.reshape(row, col, ch)
		noisy = image + image * gauss
		return noisy
	else:
		return image


def crop(img):
	# bot 50
	# top 300
	return img[300:-50, :, :]


def resize(img):
	return cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def random_flip(img, angle):
	if np.random.rand() < RANDOM_IMAGE_FLIP_PROBABILITY:
		return cv2.flip(img, 1), -angle
	else:
		return img, angle


def random_noise(img):
	if np.random.rand() < RANDOM_IMAGE_NOISE_PROBABILITY:
		return noise_generator(IMAGE_NOISE_TYPE, img)
	else:
		return img


def random_brightness(img):
	"""
	Randomly adjust brightness of the image.
	Source code: https://github.com/llSourcell/How_to_simulate_a_self_driving_car
	"""
	# HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
	hsv[:, :, 2] = hsv[:, :, 2] * ratio
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def choose_image(center, left, right, angle):
	chosen = np.random.choice(3)
	if chosen == 0:
		return center, angle
	if chosen == 1:
		return left, angle - 0.2
	if chosen == 2:
		return right, angle + 0.2


def preprocess(img):
	processed = crop(img)
	processed = resize(processed)
	return processed


def augment(img, angle):
	augmented_img, augmented_angle = random_flip(img, angle)
	augmented_img = random_brightness(augmented_img)
	augmented_img = random_noise(augmented_img)

	return augmented_img, augmented_angle


def load_image(directory_path, file):
	img = cv2.imread(os.path.join(directory_path, file))
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_images(directory_path, frame):
	return load_image(directory_path, CENTER_CAMERA_NAME.format(frame)),\
	       load_image(directory_path, LEFT_CAMERA_NAME.format(frame)),\
	       load_image(directory_path, RIGHT_CAMERA_NAME.format(frame))


def get_acceleration(acceleration, braking):
	return acceleration - braking # we can not have both values different 0. So we get either acceleration either -breaking, range[-1, 1]

def normalize_colors(img):
	return img / 127.5 - 1.0

def batch_generator(data, batch_size, is_training):
	images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
	steers = np.empty([batch_size, 2])

	while True:
		i = 0
		for idx in np.random.permutation(len(data)):
			row = data[idx]
			data_dir, frame, steering_angle, acceleration, braking = row
			acceleration_brake_val = get_acceleration(acceleration, braking)

			# argumentation
			if is_training and np.random.rand() < 0.6:
				center, left, right = load_images(data_dir, frame)
				image, steering_angle = choose_image(center, left, right, steering_angle)
			else:
				image = load_image(data_dir, CENTER_CAMERA_NAME.format(frame))

			# add the image and steering angle to the batch
			image = preprocess(image)
			images[i] = normalize_colors(image)
			# steers[i] = steering_angle
			# accelerations[i] = acceleration_brake_val
			steers[i, 0] = steering_angle
			steers[i, 1] = acceleration_brake_val
			i += 1
			if i == batch_size:
				break
		yield images, steers


if __name__ == '__main__':
	org_img = cv2.imread("out\\20181113184752\\CameraRGB_000041.png")
	img, _ = augment(org_img, 0)
	img = preprocess(img)
	cv2.imshow("SSS", img)
	cv2.imshow("orig", org_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
