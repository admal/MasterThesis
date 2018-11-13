import cv2
import numpy as np

classes = {
	0: [0, 0, 0],  # None
	1: [70, 70, 70],  # Buildings
	2: [190, 153, 153],  # Fences
	3: [72, 0, 90],  # Other
	4: [220, 20, 60],  # Pedestrians
	5: [153, 153, 153],  # Poles
	6: [157, 234, 50],  # RoadLines
	7: [128, 64, 128],  # Roads
	8: [244, 35, 232],  # Sidewalks
	9: [107, 142, 35],  # Vegetation
	10: [0, 0, 255],  # Vehicles
	11: [102, 102, 156],  # Walls
	12: [220, 220, 0]  # TrafficSigns
}


def main():
	img = cv2.imread("C:\\Users\\Adam\\Documents\\Materialy_pw\\PW\\Magisterka\\AutonomousCarProject\\out\\20181113185136\\CameraSS_000063.png")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	converted = np.zeros(img.shape, dtype=np.uint8)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			converted[i,j] = classes[img[i, j, 0]]

	converted = cv2.cvtColor(converted, cv2.COLOR_RGB2BGR)
	cv2.imwrite("ss-converted.jpg", converted)



if __name__ == '__main__':
	main()