import csv
import logging
import pandas as pd
import numpy as np
import os

from config import MEASUREMENTS_CSV_FILENAME


def describe():
	directory = ".\\out"
	logging.info("Start loading data")
	steerings = []
	throttles = []
	dir_count = 0
	last_automatic_dir_name = ".\\out\\20190402210820"
	for dir in os.walk(directory):
		if dir[0] == directory:
			continue
		measurements_file = os.path.join(dir[0], MEASUREMENTS_CSV_FILENAME)
		dir_count = dir_count + 1
		local_steerings = []
		with open(measurements_file) as csvfile:
			rows = csv.reader(csvfile)
			for row in rows:
				steerings.append(float(row[1]))
				local_steerings.append(float(row[1]))

				throttles.append(float(row[2]) - float(row[3]))

		ss = pd.Series(local_steerings)
		l_right = ss[ss >= 0.05]
		l_left = ss[ss <= -0.05]
		l_center = ss[np.logical_and(ss > -0.05, ss < 0.05)]

		print("{}; Right: {}; Center: {}; Left: {}; ".format(dir[0], len(l_right), len(l_center), len(l_left)))
		if dir[0] == last_automatic_dir_name:
			print("======================MANUAL DATA======================")

	print("Directories count: {}".format(dir_count))
	print("Data count: {}".format(len(steerings)))

	s = pd.Series(steerings)
	print(s.describe(include='all'))
	right = s[s >= 0.05]
	left = s[s <= -0.05]
	center = s[np.logical_and(s > -0.05, s < 0.05)]

	print("   All: {}; Right: {}; Center: {}; Left: {}; ".format(len(s), len(right), len(center), len(left)))
	print("  LEFT: {:.02f}% {}/{}".format(len(left) / len(s) * 100.0, len(left), len(s)))
	print(" RIGHT: {:.02f}% {}/{}".format(len(right) / len(s) * 100.0, len(right), len(s)))
	print("CENTER: {:.02f}% {}/{}".format(len(center) / len(s) * 100.0, len(center), len(s)))

if __name__ == '__main__':
	describe()
