import math


def vec3tovec2(vec3):
	return [vec3.x, vec3.y]


def distance(p1, p2):
	return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
