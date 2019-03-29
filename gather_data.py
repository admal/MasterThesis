import csv
import logging
import random
import time

from carla.client import make_carla_client
from carla.tcp import TCPConnectionError
from config import MEASUREMENTS_CSV_FILENAME
from gathering_data_common import *


def start_gathering_data(args, out_directory):
	with make_carla_client(args.host, args.port) as client:
		settings = get_settings_for_scene(args)
		scene = client.load_settings(settings)

		# Choose one player start at random.
		number_of_player_starts = len(scene.player_start_spots)
		player_start = random.randint(0, max(0, number_of_player_starts - 1))
		client.start_episode(player_start)

		skip_frames = args.skip_frames  # make screen every skip_frames

		with open(out_directory + '\\' + MEASUREMENTS_CSV_FILENAME, 'w', newline='') as csvfile:
			measurements_file = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

			# let skip first 20 frames (car is in the air)
			for _ in range(20):
				measurements, sensor_data = client.read_data()
				logging.info("Skipping frames...")
				control = measurements.player_measurements.autopilot_control
				client.send_control(control)

			frame = 0
			saved_frames = 0
			last_log_skip_frame = 0
			while True:
				measurements, sensor_data = client.read_data()

				# if car is standing than data is ignored
				if measurements.player_measurements.forward_speed <= 0.1 and measurements.player_measurements.autopilot_control.throttle == 0:
					if last_log_skip_frame != frame:
						logging.info("[{}] Car is stopped, skip saving screen...".format(frame))
						last_log_skip_frame = frame
					control = measurements.player_measurements.autopilot_control
					client.send_control(control)
					continue

				if frame % skip_frames == 0:
					saved_frames = saved_frames + 1
					logging.info("[SAVE] {}/{}: steering: {}, acc: {}, brake: {} ".format(
						saved_frames,
						args.frames,
						measurements.player_measurements.autopilot_control.steer,
						measurements.player_measurements.autopilot_control.throttle,
						measurements.player_measurements.autopilot_control.brake))

					write_measurements_to_csv(measurements_file, frame, measurements.player_measurements.autopilot_control)

					for name, measurement in sensor_data.items():
						save_frame_image(out_directory, frame, measurement, name)

				frame = frame + 1
				control = measurements.player_measurements.autopilot_control
				client.send_control(control)

				if saved_frames >= args.frames:
					logging.info("Saved frames: {}; STOP".format(saved_frames))
					break


def main():
	argparser = generate_run_arguments()
	args = argparser.parse_args()
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
