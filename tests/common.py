import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import logging
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

out_run_folder = 'out\\map_points\\autonomous_runs\\'
reference_dir = 'out\\map_points\\reference\\'
run_csv_name = 'run.csv'
run_image_name = '{}.png'
result_name = 'result.txt'


def evaluate_run(run_dir, map_name, model_name='', checkpoint=0):
	reference_csv = reference_dir + '{}\\{}.csv'.format(map_name, map_name)
	reference_points = pd.read_csv(reference_csv, dtype={'x': float, 'y': float})

	run_csv = os.path.join(run_dir, run_csv_name)
	run_points = pd.read_csv(run_csv, dtype={'x': float, 'y': float})

	stacked_reference = np.column_stack((reference_points['x'], reference_points['y']))
	stacked_run = np.column_stack((run_points['x'], run_points['y']))

	dist = compare(stacked_reference, stacked_run)
	f = open(os.path.join(run_dir, result_name), 'w')
	f.write('Run results\n==============\nmodel: {} epoch: {}\nResult:{}'.format(model_name, checkpoint, dist))
	f.close()

	if model_name == '' or checkpoint == 0:
		logging.warning('Not all information about run has been provided! Provide them by hand into result.txt')


def compare(line1, line2):
	logging.info('Start DTW algorithm...')
	distance, _ = fastdtw(line1, line2, dist=euclidean)
	logging.info('DTW algorithm finished. Final score: {}'.format(distance))
	return distance


def save_run(line_x, line_y, model_name, map_name):
	logging.info("Start saving run for {} on {}...".format(model_name, map_name))
	dir_path = out_run_folder + "{}\\{}\\".format(model_name, map_name)

	run_dir = create_run_dir(dir_path)
	save_plot(line_x, line_y, map_name, run_dir)
	save_to_csv(line_x, line_y, run_dir)
	logging.info('Saved in {}'.format(run_dir))
	return run_dir


def create_run_dir(dir_path):
	logging.info("Start creating run directory...")
	timestamp = datetime.timestamp(datetime.now())
	run_dir = dir_path + 'run-{}'.format(timestamp)
	os.makedirs(run_dir)
	return run_dir


def save_to_csv(line_x, line_y, run_dir):
	logging.info("Start saving points...")
	df = pd.DataFrame({'x': line_x, 'y': line_y})
	df.to_csv(os.path.join(run_dir, 'run.csv'), index=False)


def save_plot(line_x, line_y, map_name, run_dir):
	logging.info("Start saving plot...")
	plt.plot(line_x, line_y, 'ro')
	plt.scatter(line_x[0], line_y[0], s=500, c='green')
	plt.scatter(line_x[len(line_x) - 1], line_y[len(line_y) - 1], s=500, c='blue')
	plt.savefig(os.path.join(run_dir, run_image_name.format(map_name)))
