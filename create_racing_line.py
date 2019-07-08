import logging
from carla.client import make_carla_client
import matplotlib.pyplot as plt
from gathering_data_common import generate_run_arguments, get_settings_for_scene
import pandas as pd
from common import *

points_x = []
points_y = []


def start(args):
	line_points = []
	skip_frames = 40
	with make_carla_client(args.host, args.port) as client:
		settings = get_settings_for_scene(args)
		scene = client.load_settings(settings)
		client.start_episode(0)

		written_info_stop = False
		frame = 0
		while True:
			frame = frame + 1
			measurements, sensor_data = client.read_data()

			if frame < skip_frames:
				logging.info("Skipping first {} frames...".format(skip_frames))
				control = measurements.player_measurements.autopilot_control
				client.send_control(control)
				continue

			# if car is standing than data is ignored
			if measurements.player_measurements.forward_speed <= 0.1 and measurements.player_measurements.autopilot_control.throttle == 0:
				if not written_info_stop:
					written_info_stop = True
					logging.info("Car is stopped, stop gathering data...")
				control = measurements.player_measurements.autopilot_control
				client.send_control(control)
				continue

			written_info_stop = False
			current_position = vec3tovec2(measurements.player_measurements.transform.location)
			if len(line_points) > 1:
				dist_from_start = distance(line_points[0], current_position)
			else:
				dist_from_start = 10000

			if dist_from_start < 0.5:
				logging.info("Position: {} is already logged".format(current_position))
				break



			line_points.append(current_position)
			points_x.append(current_position[0])
			points_y.append(current_position[1])
			logging.info("Add point: [{:.4f},{:.4f}], points count: {:0>4d}, distance from start: {:.4f}".format(
				current_position[0],
				current_position[1],
				len(line_points),
				dist_from_start))

			control = measurements.player_measurements.autopilot_control
			client.send_control(control)


def main():
	argparser = generate_run_arguments()
	argparser.add_argument(
		'-m', '--map',
		choices=['Town01', 'Town02', 'Town03', 'Town04', 'TestTown', 'TestTown02', 'TestTown03'],
		default='Town01',
		help='The name of town for which race line is created (filename will be created with that name)'
	)
	args = argparser.parse_args()
	log_level = logging.DEBUG if args.debug else logging.INFO
	logging.basicConfig(level=log_level)
	start(args)
	df = pd.DataFrame({'x': points_x, 'y': points_y})
	df.to_csv('out\\map_points\\' + args.map+'.csv', index=False)


if __name__ == '__main__':
	try:
		main()


		plt.plot(points_x, points_y, 'ro')
		plt.scatter(points_x[0], points_y[0], s=1000, c='green')
		plt.scatter(points_x[len(points_x) - 1], points_y[len(points_y) - 1], s=1000, c='blue')
		plt.show()
	except KeyboardInterrupt:
		print('\nCanceled by user...')
