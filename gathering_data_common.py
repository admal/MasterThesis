import datetime
import os
import argparse
import cv2

from carla.sensor import Camera
from carla.settings import CarlaSettings
from data_augmentation import preprocess

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


def generate_settings(args, sync_mode=True):
	settings = CarlaSettings()
	settings.set(
		SynchronousMode=sync_mode,
		SendNonPlayerAgentsInfo=False,
		NumberOfVehicles=0,
		NumberOfPedestrians=0,
		WeatherId=args.weather,
		QualityLevel=args.quality_level)
	settings.randomize_seeds()

	return settings


def get_settings_for_scene(args, sync_mode=True):
	if args.settings_filepath is None:
		settings = generate_settings(args, sync_mode)
		add_cameras(settings)
	else:
		with open(args.settings_filepath, 'r') as fp:
			settings = fp.read()
	return settings


def write_measurements_to_csv(measurements_file, frame, control):
	measurements_file.writerow([frame, control.steer, control.throttle, control.brake])



def generate_run_arguments():
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
		default=1000,
		help='number of frames to be saved (default: 1000)'
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
		default=10,
		help='save screen every skip_frames'
	)
	return argparser


def save_frame_image(out_directory, frame, sensor_data, name):
	filename = out_directory + '\\{}_{:0>6d}.png'.format(name, int(frame))
	data = sensor_data.data
	data = preprocess(data)
	data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
	cv2.imwrite(filename, data)