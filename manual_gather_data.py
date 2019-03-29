# Modified file: Carla 0.8.4 manual_control.py
# Changes: adjust script for my data collection, joystick controller

# Running carla simulator
# CarlaUE4.exe /Game/Maps/Town04 -carla-settings=CarlaSettings.ini -windowed -ResX=800 -ResY=600 -carla-server

"""
Welcome to CARLA manual control.

Use joystick for driving
LEFT TRIGGER : BRAKING
RIGHT TRIGGER : THROTTLE
LEFT AXIS : STEERING

STARTING in a moment...
"""

import csv
import logging
import time
try:
	import pygame
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')
try:
	import numpy as np
except ImportError:
	raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla.client import make_carla_client, VehicleControl
from carla.tcp import TCPConnectionError

from gathering_data_common import *
from config import MEASUREMENTS_CSV_FILENAME

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
X_AXIS_DEADZONE = (-0.2, 0.2)


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
		self._carla_settings = get_settings_for_scene(args, sync_mode=False)
		self._timer = None
		self._display = None
		self._main_image = None
		self._is_on_reverse = False
		self._city_name = None
		self._map = None
		self._position = None
		self._agent_positions = None
		self._saved_frames = 0
		self._out_directory = ""
		self._current_joystick = None

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

	def _init_joystick(self):
		# get the first joystick
		self._current_joystick = pygame.joystick.Joystick(0)
		self._current_joystick.init()

	def _initialize_game(self):
		self._on_new_episode()
		self._display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT),pygame.HWSURFACE | pygame.DOUBLEBUF)
		logging.debug('pygame started')

	def _on_new_episode(self):
		scene = self.client.load_settings(self._carla_settings)
		self._city_name = scene.map_name
		number_of_player_starts = len(scene.player_start_spots)
		player_start = np.random.randint(number_of_player_starts)
		print('Starting new episode...')
		self.client.start_episode(player_start)
		self._timer = Timer()
		self._is_on_reverse = False

	def _on_loop(self, measurements_file, frames, frame, skip_frames):
		self._timer.tick()
		_, sensor_data = self.client.read_data()
		self._main_image = sensor_data.get('CameraRGB_C', None)
		control = self._get_joystick_control()

		# save every skip_frames and start saving after 20 first (car is not in proper place and not on the ground)
		if frame > 20 and frame % skip_frames == 0:
			self._saved_frames = self._saved_frames + 1
			logging.info("[SAVE] {}/{}: steering: {}, acc: {}, brake: {} ".format(
				self._saved_frames,
				frames,
				control.steer,
				control.throttle,
				control.brake))

			write_measurements_to_csv(measurements_file, frame, control)

			for name, sensor_data in sensor_data.items():
				save_frame_image(self._out_directory, frame, sensor_data, name)

		if control is None:
			self._on_new_episode()
		else:
			self.client.send_control(control)

	def _get_joystick_control(self):
		# define ids of important axes on joystick
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
		elif trigger_value < 0:
			control.throttle = -trigger_value
		
		control.steer = self._current_joystick.get_axis(X_AXIS)
		if X_AXIS_DEADZONE[0] < control.steer < X_AXIS_DEADZONE[1]:
			control.steer = 0
		
		return control

	def _on_render(self):
		if self._main_image is not None:
			array = image_converter.to_rgb_array(self._main_image)
			surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
			self._display.blit(surface, (0, 0))

		pygame.display.flip()


def main():
	argparser = generate_run_arguments()
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