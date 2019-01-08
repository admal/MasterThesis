import argparse
import logging
import datetime
import random
import time
import numpy as np
import cv2

from data_augmentation import preprocess
from neural_networks.VGG16Model import VGG16Model

from carla.client import make_carla_client, VehicleControl
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line


def generate_settings(args):
	settings = CarlaSettings()
	settings.set(
		SynchronousMode=True,
		SendNonPlayerAgentsInfo=False,
		NumberOfVehicles=0,
		NumberOfPedestrians=0,
		WeatherId=1,
		QualityLevel=args.quality_level)
	settings.randomize_seeds()

	return settings


def drive(args, model):
	print("Connecting...")
	with make_carla_client(args.host, args.port) as client:
		print("Generating settings...")
		settings = generate_settings(args)
		camera_pos_x = 2
		camera_pos_y = 0
		camera_pos_z = 1

		camera = Camera("MainCamera")
		camera.set_image_size(800, 600)
		camera.set_position(camera_pos_x, camera_pos_y, camera_pos_z)
		settings.add_sensor(camera)

		scene = client.load_settings(settings)

		print("Starting episode...")
		# Choose one player start at random.
		number_of_player_starts = len(scene.player_start_spots)
		player_start = random.randint(0, max(0, number_of_player_starts - 1))
		client.start_episode(player_start)
		print("Start driving...")
		while True:
			measurements, sensor_data = client.read_data()

			steer = 0.0
			acceleration = 0.0
			for name, measurement in sensor_data.items():
				input = preprocess(measurement.data)
				input = np.array(input / 127.5 - 1, dtype=np.float32)
				input = np.expand_dims(input, axis=0)
				ret = model.predict(input)[0]
				print(ret)
				steer = ret[0]
				acceleration = 0.5 + ret[1] #TMP

			control = VehicleControl()
			control.steer = steer
			control.throttle = acceleration
			client.send_control(control)

def main(args):
	print("Loading model...")
	model = VGG16Model().model()
	model = VGG16Model.load_weights(model, "trained_models\\model-009.h5")

	while True:
		try:
			print("START")
			drive(args, model)
			print("\nFinished (at {})".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

		except TCPConnectionError as error:
			logging.error(error)
			time.sleep(1)


if __name__ == '__main__':
	argparser = argparse.ArgumentParser(description=__doc__)
	argparser.add_argument(
		'-v', '--verbose',
		action='store_true',
		dest='debug',
		help='print debug information')
	argparser.add_argument(
		'--host',
		metavar='H',
		default='localhost',
		help='IP of the host server (default: localhost)')
	argparser.add_argument(
		'-p', '--port',
		metavar='P',
		default=2000,
		type=int,
		help='TCP port to listen to (default: 2000)')
	argparser.add_argument(
		'-q', '--quality-level',
		choices=['Low', 'Epic'],
		type=lambda s: s.title(),
		default='Epic',
		help='graphics quality level, a lower level makes the simulation run considerably faster.')
	args = argparser.parse_args()
	main(args)