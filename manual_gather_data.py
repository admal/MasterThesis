# Modified file: Carla 0.8.4 manual_control.py

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

from config import MEASUREMENTS_CSV_FILENAME
from gathering_data_common import generate_run_arguments, create_out_directory, save_run_config, \
	write_measurements_to_csv, save_frame_image, get_settings_for_scene

try:
	import pygame
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
	import numpy as np
except ImportError:
	raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla.client import make_carla_client, VehicleControl
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

X_AXIS_DEADZONE = (-0.15, 0.15)

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
		self._current_joystick = None
		self._out_directory = None
		self._saved_frames = 0
		self._max_saved_frames = 0

	def execute(self, out_directory, frames, every_second):
		"""Launch the PyGame."""
		pygame.init()
		pygame.joystick.init()
		self._initialize_game()
		self._init_joystick()
		self._max_saved_frames = frames

		self._out_directory = out_directory
		logging.info("Ot directory: {}".format(self._out_directory))
		with open(out_directory + '\\' + MEASUREMENTS_CSV_FILENAME, 'w', newline='') as csvfile:
			measurements_file = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			try:
				while True:
					for event in pygame.event.get():
						if event.type == pygame.QUIT:
							return
					self._on_loop(measurements_file, every_second)
					self._on_render()
					if self._saved_frames >= self._max_saved_frames:
						logging.info("All data saved. Quit...")
						break
			finally:
				pygame.quit()

	def _initialize_game(self):
		self._on_new_episode()

		logging.debug('pygame started')

	def _init_joystick(self):
		# get the first joystick
		self._current_joystick = pygame.joystick.Joystick(0)
		self._current_joystick.init()

	def _on_new_episode(self):
		scene = self.client.load_settings(self._carla_settings)
		print("Settings\n{}".format(self._carla_settings))
		number_of_player_starts = len(scene.player_start_spots)
		player_start = np.random.randint(number_of_player_starts)
		print('Starting new episode...')
		self.client.start_episode(player_start)
		self._timer = Timer()

	def _on_loop(self, measurements_file, every_second):
		self._timer.tick()

		control = self._get_joystick_control()
		if self._timer.elapsed_seconds_since_lap() > every_second:
			self._save_data_frame(control, measurements_file)

		if control is None:
			self._on_new_episode()
		else:
			self.client.send_control(control)

	def _save_data_frame(self, control, measurements_file):
		_, sensor_data = self.client.read_data()
		if len(sensor_data) < 3:
			logging.warning("{}/{}: missing camera shots from Carla, skip saving (#shots: {})".format(
				self._saved_frames + 1,
				self._max_saved_frames,
				len(sensor_data)
			))
			return

		write_measurements_to_csv(measurements_file, self._saved_frames, control)
		for name, sensor_data in sensor_data.items():
			save_frame_image(self._out_directory, self._saved_frames, sensor_data, name)
		logging.info("[SAVE] {}/{}: steering: {}, acc: {}, brake: {} ".format(
			self._saved_frames + 1,
			self._max_saved_frames,
			control.steer,
			control.throttle,
			control.brake))
		self._saved_frames = self._saved_frames + 1
		self._timer.lap()

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

	def _print_player_measurements(self, player_measurements):
		message = 'Step {step} ({fps:.1f} FPS): '
		message += '{speed:.2f} km/h, '
		message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
		message = message.format(
			step=self._timer.step,
			fps=self._timer.ticks_per_second(),
			speed=player_measurements.forward_speed * 3.6,
			other_lane=100 * player_measurements.intersection_otherlane,
			offroad=100 * player_measurements.intersection_offroad)
		print_over_same_line(message)

	def _on_render(self):
		pass

def main():
	argparser = generate_run_arguments()
	argparser.add_argument(
		'-e', '--every-second',
		default=0.2,
		type=float,
		help='Save measurements every -e seconds'
	)
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
				game.execute(out_directory, args.frames, args.every_second)
				break

		except TCPConnectionError as error:
			logging.error(error)
			time.sleep(1)


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')
