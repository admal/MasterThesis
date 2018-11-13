import cv2
import numpy as np


def main():
	img = cv2.imread(
		"C:\\Users\\Adam\\Documents\\Materialy_pw\\PW\\Magisterka\\AutonomousCarProject\\out\\20181113185136\\CameraDepth_000063.png")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	converted = np.zeros(img.shape, dtype=np.uint8)

	far = 1.0
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			R = img[i, j, 0]
			G = img[i, j, 1]
			B = img[i, j, 2]
			val = R + G * 256 + B * 256 * 256
			val = val / (256 * 256 * 256 - 1)
			val = val * far * 255

			converted[i, j] = [val, val, val]

	converted = cv2.cvtColor(converted, cv2.COLOR_RGB2BGR)
	cv2.imwrite("depth_converted.jpg", converted)


if __name__ == '__main__':
	main()
