import configparser
import cv2
import numpy as np
import argparse
import csv
import logging
import os
import datetime
import random
import time
from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

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


def print_measurements(frame, measurements):
	number_of_agents = len(measurements.non_player_agents)
	player_measurements = measurements.player_measurements
	message = '' + str(frame) + ' '
	message += 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
	message += '{speed:.0f} km/h, '
	message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
	message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
	message += '({agents_num:d} non-player agents in the scene)'
	message = message.format(
		pos_x=player_measurements.transform.location.x,
		pos_y=player_measurements.transform.location.y,
		speed=player_measurements.forward_speed * 3.6,  # m/s -> km/h
		col_cars=player_measurements.collision_vehicles,
		col_ped=player_measurements.collision_pedestrians,
		col_other=player_measurements.collision_other,
		other_lane=100 * player_measurements.intersection_otherlane,
		offroad=100 * player_measurements.intersection_offroad,
		agents_num=number_of_agents)
	print_over_same_line(message)


def write_measurements_to_csv(measurements_file, frame, autopilot_measurements):
	measurements_file.writerow(
		[frame, autopilot_measurements.steer, autopilot_measurements.throttle, autopilot_measurements.brake])


def start_gathering_data(args, out_directory):
	with make_carla_client(args.host, args.port) as client:
		settings = get_settings_for_scene(args)
		scene = client.load_settings(settings)

		# Choose one player start at random.
		number_of_player_starts = len(scene.player_start_spots)
		player_start = random.randint(0, max(0, number_of_player_starts - 1))
		client.start_episode(player_start)

		skip_frames = args.skip_frames  # make screen every skip_frames

		with open(out_directory + '\\measurements.csv', 'w', newline='') as csvfile:
			measurements_file = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

			# let skip first 20 frames (car is in the air)
			for skip_frame in range(20):
				measurements, sensor_data = client.read_data()
				logging.info("Skipping frames...")
				control = measurements.player_measurements.autopilot_control
				client.send_control(control)

			for frame in range(args.frames * skip_frames):
				measurements, sensor_data = client.read_data()
				print_measurements(frame, measurements)

				if frame % skip_frames == 0:
					logging.info("{}: Save".format(frame))
					write_measurements_to_csv(measurements_file, frame,
					                          measurements.player_measurements.autopilot_control)
					for name, measurement in sensor_data.items():
						filename = out_directory + '\\{}_{:0>6d}'.format(name, int(frame / skip_frames))
						measurement.save_to_disk(filename)

				control = measurements.player_measurements.autopilot_control
				client.send_control(control)

	pass


def parse_arguments():
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
	argparser.add_argument(
		'-c', '--carla-settings',
		metavar='PATH',
		dest='settings_filepath',
		default=None,
		help='Path to a "CarlaSettings.ini" file')
	argparser.add_argument(
		'-f', '--frames',
		type=int,
		default=None,
		help='number of frames to be generated (default: 1000)'
	)
	argparser.add_argument(
		'-w', '--weather',
		type=int,
		default=0,
		help='weather preset'
	)
	argparser.add_argument(
		'-s', '--skip_frames',
		type=int,
		default=20,
		help='save screen every skip_frames'
	)

	args = argparser.parse_args()
	print(str(args))
	return args


def main():
	args = parse_arguments()

	log_level = logging.DEBUG if args.debug else logging.INFO
	logging.basicConfig(level=log_level)

	out_directory = create_out_directory()
	save_run_config(out_directory, args)

	logging.info('STARTING GATHERING DATA (at {})'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

	while True:
		try:
			start_gathering_data(args, out_directory)
			print("\nFinished (at {})".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
			return

		except TCPConnectionError as error:
			logging.error(error)
			time.sleep(1)


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print('\nCanceled by user...')
