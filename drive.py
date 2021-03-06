#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Carla simulation started...
"""
import argparse
import logging
import time

from carla.sensor import Camera
from carla.settings import CarlaSettings
from data_augmentation import preprocess
from neural_networks.neural_networks_common import add_model_cmd_arg, load_model
from common import *
from tests.common import evaluate_run, save_run
try:
	import numpy as np
except ImportError:
	raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# from carla import image_converter
from carla.client import make_carla_client, VehicleControl
from carla.tcp import TCPConnectionError

from config import USE_SPEED_CONSTRAINTS, MINIMAL_SPEED, MAXIMAL_SPEED

# the pygame screen is created only to handle key input
WINDOW_WIDTH = 100
WINDOW_HEIGHT = 100

class Timer(object):
	def __init__(self):
		self.step = 0
		self._lap_step = 0
		self._lap_time = time.time()

	def tick(self):
		self.step += 1

	def lap(self):
		self._lap_step = self.step
		self._lap_time = time.time()

	def ticks_per_second(self):
		return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

	def elapsed_seconds_since_lap(self):
		return time.time() - self._lap_time

def gen_settings(args):
	settings = CarlaSettings()
	settings.set(
		SynchronousMode=True,
		SendNonPlayerAgentsInfo=False,
		NumberOfVehicles=0,
		NumberOfPedestrians=0,
		WeatherId=args.weather,
		QualityLevel=args.quality_level)
	settings.randomize_seeds()
	camera_pos_x = 2
	camera_pos_y = 0
	camera_pos_z = 1

	camera = Camera("MainCamera")
	camera.set_image_size(800, 600)
	camera.set_position(camera_pos_x, camera_pos_y, camera_pos_z)
	settings.add_sensor(camera)
	return settings

class CarlaGame(object):
	def __init__(self, carla_client, args, model):
		self.client = carla_client
		self._carla_settings = gen_settings(args)
		self._timer = None
		self._model = model
		self._model_name = args.model
		self._map_name = args.map_name
		self._checkpoint = args.checkpoint
		self._velocities = []
		self._run_start_time = None
		self._weather = args.weather

	def save_result(self):
		run_dir = save_run(self.points_x, self.points_y, self._model_name, self._map_name)
		evaluate_run(run_dir, self._map_name, self._velocities, self._model_name, self._checkpoint, self._weather)

	def execute(self):
		self._initialize_game()
		frame = 0
		try:
			while True:
				frame = frame + 1
				is_ok = self._on_loop(frame)
				if not is_ok:
					self.save_result()
					break
		except KeyboardInterrupt:
			self.save_result()

	def _initialize_game(self):
		self._on_new_episode()

		logging.debug('pygame started')

	def _on_new_episode(self):
		self._carla_settings.randomize_seeds()
		scene = self.client.load_settings(self._carla_settings)
		number_of_player_starts = len(scene.player_start_spots)
		player_start = np.random.randint(number_of_player_starts)
		print('Starting new episode...')
		self.client.start_episode(player_start)
		self._timer = Timer()
		self._run_start_time = time.time()
		self.line_points = []
		self.points_x = []
		self.points_y = []

	def _on_loop(self, frame):
		self._timer.tick()

		skip_frames = 40
		measurements, sensor_data = self.client.read_data()
		current_position = vec3tovec2(measurements.player_measurements.transform.location)
		self._velocities.append(measurements.player_measurements.forward_speed * 3.6) # convert to km/h

		steer = 0.0
		acceleration = 0.0
		for name, measurement in sensor_data.items():
			model_input = preprocess(measurement.data)
			model_input = np.array(model_input / 127.5 - 1, dtype=np.float32)
			model_input = np.expand_dims(model_input, axis=0)
			ret = self._model.predict(model_input)[0]
			steer = ret[0]
			acceleration = ret[1]

		if USE_SPEED_CONSTRAINTS:
			if measurements.player_measurements.forward_speed * 3.6 < MINIMAL_SPEED:
				acceleration = 0.7
			elif measurements.player_measurements.forward_speed * 3.6 > MAXIMAL_SPEED:
				acceleration = 0

		control = VehicleControl()
		control.steer = steer
		control.throttle = acceleration
		self.client.send_control(control)
		if frame < skip_frames:
			logging.info("Skipping first {} frames...".format(skip_frames))
			return True

		self.line_points.append(current_position)
		self.points_x.append(current_position[0])
		self.points_y.append(current_position[1])

		if len(self.line_points) > 1:
			dist_from_start = distance(self.line_points[0], current_position)
		else:
			dist_from_start = 10000

		if dist_from_start < 1 and frame > skip_frames + 100:
			logging.info("Position: {} is already logged".format(current_position))
			return False

		if self._timer.elapsed_seconds_since_lap() > 0.5:
			self._print_player_measurements(control)
			logging.info("Add point: [{:.4f},{:.4f}], points count: {:0>4d}, distance from start: {:.4f}".format(
				current_position[0],
				current_position[1],
				len(self.line_points),
				dist_from_start))
			self._timer.lap()

		return True

	def _print_player_measurements(self, control):
		msg = "steer: {:.2f}, throttle: {:.2f}, "
		msg += "avg speed: {:.2f}km/h "
		logging.info(msg
			.format(
			control.steer,
			control.throttle,
			np.average(self._velocities)
		))

def main():
	argparser = argparse.ArgumentParser(
		description='AI Driving')
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
		help='graphics quality level, a lower level makes the simulation run considerably faster')
	argparser.add_argument(
		'-w', '--weather',
		type=int,
		default=0,
		help='weather preset'
	)
	add_model_cmd_arg(argparser)
	argparser.add_argument(
		'-c',
		'--checkpoint',
		dest='checkpoint',
		type=int,
		default=1,
		help='number of model\'s checkpoint to load'
	)
	argparser.add_argument(
		'-t',
		'--map_name',
		default='TestTown',
		choices=['TestTown', 'Town03', 'Town04', 'TestTown02', 'TestTown03']
	)
	args = argparser.parse_args()

	log_level = logging.DEBUG if args.debug else logging.INFO
	logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

	logging.info('listening to server %s:%s', args.host, args.port)

	print(__doc__)

	model = load_model(args.model, args.checkpoint, False)

	while True:
		try:
			with make_carla_client(args.host, args.port) as client:
				game = CarlaGame(client, args, model)
				game.execute()
				break

		except TCPConnectionError as error:
			logging.error(error)
			time.sleep(1)


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')
