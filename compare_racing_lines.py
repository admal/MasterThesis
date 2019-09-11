import argparse
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def compare(line1, line2):
	distance, path = fastdtw(line1, line2, dist=euclidean, radius=10)
	print(distance)
	pass


if __name__ == '__main__':
	argparser = argparse.ArgumentParser(description=__doc__)
	argparser.add_argument(
		'-l1', '--line1'
	)
	argparser.add_argument(
		'-l2', '--line2'
	)
	args = argparser.parse_args()

	line1 = pd.read_csv(args.line1, dtype={'x': np.float, 'y': np.float})
	line2 = pd.read_csv(args.line2, dtype={'x': np.float, 'y': np.float})

	compare(np.column_stack((line1['x'], line1['y'])), np.column_stack((line2['x'], line2['y'])))
