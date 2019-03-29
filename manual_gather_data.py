#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.
#running carla simulator
# CarlaUE4.exe /Game/Maps/Town04 -carla-settings=CarlaSettings.ini -windowed -ResX=800 -ResY=600 -carla-server

"""
Welcome to CARLA manual control.

Use joystick for driving
LEFT TRIGGER : BRAKING
RIGHT TRIGGER : THROTTLE
LEFT AXIS : STEERING

STARTING in a moment...
"""


import argparse
import csv
import logging
import random
import time
import datetime

import os

from config import MEASUREMENTS_CSV_FILENAME

try:
	import pygame
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
	import numpy as np
except ImportError:
	raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla.sensor import Camera


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
X_AXIS_DEADZONE = (-0.2, 0.2)

#moje

weather_presets = {
	0: 'Default',
	1: 'ClearNoon',
	2: 'CloudyNoon',
	3: 'WetNoon',
	4: 'WetCloudyNoon',
	5: 'MidRainyNoon',
	6: 'HardRainNoon',
	7: 'SoftRainNoon',
	8: 'ClearSunset',
	9: 'CloudySunset',
	10: 'WetSunset',
	11: 'WetCloudySunset',
	12: 'MidRainSunset',
	13: 'HardRainSunset',
	14: 'SoftRainSunset',
}


def save_run_config(directory, args):
	f = open(directory + '\\config.txt', 'w')
	f.write(str(args))


def create_out_directory():
	out_directory = '.\\out\\{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
	os.makedirs(out_directory)
	return out_directory



class CameraSettings:
	x = 0
	y = 0
	z = 0

	width = 800
	height = 600
	name = ''
	postprocessing = "SceneFinal"

	def __init__(self, x, y, z, name, postprocessing):
		self.x = x
		self.y = y
		self.z = z
		self.name = name
		self.postprocessing = postprocessing


def create_camera(settings, camera_settings):
	camera = Camera(camera_settings.name)
	camera.set_image_size(camera_settings.width, camera_settings.height)
	camera.set_position(camera_settings.x, camera_settings.y, camera_settings.z)
	settings.add_sensor(camera)


def add_cameras(settings):
	camera_pos_x = 2
	camera_pos_y = 0
	camera_pos_z = 1

	camera_settings = CameraSettings(camera_pos_x, camera_pos_y, camera_pos_z, "CameraRGB_C", "SceneFinal")
	create_camera(settings, camera_settings)

	camera_settings = CameraSettings(camera_pos_x, camera_pos_y + 0.3, camera_pos_z, "CameraRGB_R", "SceneFinal")
	create_camera(settings, camera_settings)

	camera_settings = CameraSettings(camera_pos_x, camera_pos_y - 0.3, camera_pos_z, "CameraRGB_L", "SceneFinal")
	create_camera(settings, camera_settings)


def generate_settings(args):
	settings = CarlaSettings()
	settings.set(
		SynchronousMode=True,
		SendNonPlayerAgentsInfo=False,
		NumberOfVehicles=0,
		NumberOfPedestrians=0,
		WeatherId=args.weather,
		QualityLevel=args.quality_level)
	settings.randomize_seeds()

	return settings


def get_settings_for_scene(args):
	if args.settings_filepath is None:
		settings = generate_settings(args)
		add_cameras(settings)
	else:
		with open(args.settings_filepath, 'r') as fp:
			settings = fp.read()
	return settings


#-end moje










def make_carla_settings(args):
	settings = get_settings_for_scene(args)
	return settings


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


class CarlaGame(object):
	def __init__(self, carla_client, args):
		self.client = carla_client
		self._carla_settings = make_carla_settings(args)
		self._timer = None
		self._display = None
		self._main_image = None
		# self._map_view = None
		self._is_on_reverse = False
		# self._display_map = args.map
		self._city_name = None
		self._map = None
		# self._map_shape = None
		# self._map_view = None
		self._position = None
		self._agent_positions = None

	def execute(self, out_directory, skip_frames, frames):
		"""Launch the PyGame."""
		pygame.init()
		pygame.joystick.init()
		self._initialize_game()
		self._init_joystick()
		self._out_directory = out_directory
		with open(out_directory + '\\' + MEASUREMENTS_CSV_FILENAME, 'w', newline='') as csvfile:
			measurements_file = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			try:
				frame = 0
				while True:
					for event in pygame.event.get():
						if event.type == pygame.QUIT:
							return
					self._on_loop(measurements_file, frames, frame, skip_frames )
					self._on_render()
					frame = frame + 1
					if self._saved_frames >= frames:
						logging.info("Saved: {} frames. Closing...".format(self._saved_frames))
						break
			finally:
				logging.info("Quiting pygame...")
				pygame.quit()

	_current_joystick = None
	def _init_joystick(self):
		# get the first joystick
		self._current_joystick = pygame.joystick.Joystick(0)
		self._current_joystick.init()

	def _initialize_game(self):
		self._on_new_episode()

		if self._city_name is not None:
			self._map = CarlaMap(self._city_name, 0.1643, 50.0)
			# self._map_shape = self._map.map_image.shape
			# self._map_view = self._map.get_map(WINDOW_HEIGHT)

			# extra_width = int((WINDOW_HEIGHT/float(self._map_shape[0]))*self._map_shape[1])
			# self._display = pygame.display.set_mode(
				# (WINDOW_WIDTH + extra_width, WINDOW_HEIGHT),
				# pygame.HWSURFACE | pygame.DOUBLEBUF)
			self._display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
		else:
			self._display = pygame.display.set_mode(
				(WINDOW_WIDTH, WINDOW_HEIGHT),
				pygame.HWSURFACE | pygame.DOUBLEBUF)

		logging.debug('pygame started')

	def _on_new_episode(self):
		# self._carla_settings.randomize_seeds()
		# self._carla_settings.randomize_weather()
		scene = self.client.load_settings(self._carla_settings)
		# if self._display_map:
		self._city_name = scene.map_name
		number_of_player_starts = len(scene.player_start_spots)
		player_start = np.random.randint(number_of_player_starts)
		print('Starting new episode...')
		self.client.start_episode(player_start)
		self._timer = Timer()
		self._is_on_reverse = False


	def _write_measurements_to_csv(self, measurements_file, frame, autopilot_measurements):
		measurements_file.writerow(
			[frame, autopilot_measurements.steer, autopilot_measurements.throttle, autopilot_measurements.brake])

	_saved_frames = 0
	_out_directory = ""

	def _on_loop(self, measurements_file, frames, frame, skip_frames):
		self._timer.tick()

		measurements, sensor_data = self.client.read_data()

		self._main_image = sensor_data.get('CameraRGB_C', None)
		control = self._get_joystick_control()
		# Set the player position
		if self._city_name is not None:
			self._position = self._map.convert_to_pixel([
				measurements.player_measurements.transform.location.x,
				measurements.player_measurements.transform.location.y,
				measurements.player_measurements.transform.location.z])
			self._agent_positions = measurements.non_player_agents


		if frame % skip_frames == 0:
			self._saved_frames = self._saved_frames + 1
			logging.info("[SAVE] {}/{}: steering: {}, acc: {}, brake: {} ".format(
				self._saved_frames,
				frames,
				control.steer,
				control.throttle,
				control.brake))

			self._write_measurements_to_csv(measurements_file, frame, control)

			for name, measurement in sensor_data.items():
				filename = self._out_directory + '\\{}_{:0>6d}'.format(name, int(frame))
				measurement.save_to_disk(filename)


		if control is None:
			self._on_new_episode()
		else:
			self.client.send_control(control)


	def _get_joystick_control(self):
		#define ids of important axes on joystick
		TRIGGERS = 2
		X_AXIS = 0

		control = VehicleControl()
		control.throttle = 0
		control.brake = 0
 		# dont know why but axes in PYGAME are inverted, LEFT_TRIGGGER is >0 and RIGHT_TRIGGER <0
		# LEFT_TRIGGER is for braking
		trigger_value = self._current_joystick.get_axis(TRIGGERS)
		if trigger_value > 0:
			control.brake = trigger_value
		# RIGHT_TRIGGER is for throttle
		elif trigger_value< 0:
			control.throttle = -trigger_value
		
		control.steer  = self._current_joystick.get_axis(X_AXIS)
		if control.steer > X_AXIS_DEADZONE[0] and control.steer < X_AXIS_DEADZONE[1]:
			control.steer = 0
		
		return control

	def _on_render(self):
		if self._main_image is not None:
			array = image_converter.to_rgb_array(self._main_image)
			surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
			self._display.blit(surface, (0, 0))

		pygame.display.flip()


def main():
	argparser = argparse.ArgumentParser(
		description='CARLA Manual Control Client')
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
		'-s', '--skip_frames',
		type=int,
		default=10,
		help='save screen every skip_frames'
	)
	argparser.add_argument(
		'-w', '--weather',
		type=int,
		default=0,
		help='weather preset'
	)
	argparser.add_argument(
		'-f', '--frames',
		type=int,
		default=None,
		help='number of frames to be saved (default: 1000)'
	)
	argparser.add_argument(
		'-c', '--carla-settings',
		metavar='PATH',
		dest='settings_filepath',
		default=None,
		help='Path to a "CarlaSettings.ini" file')
	args = argparser.parse_args()

	log_level = logging.DEBUG if args.debug else logging.INFO
	logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

	logging.info('listening to server %s:%s', args.host, args.port)

	print(__doc__)

	out_directory = create_out_directory()
	save_run_config(out_directory, args)

	while True:
		try:

			with make_carla_client(args.host, args.port) as client:
				game = CarlaGame(client, args)
				game.execute(out_directory, args.skip_frames, args.frames)
				break

		except TCPConnectionError as error:
			logging.error(error)
			time.sleep(1)


if __name__ == '__main__':

	try:
		main()
	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')
