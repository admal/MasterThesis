# Script is designed to run gathering data for every map and every weather preset
import subprocess
import psutil
import logging
import os
from datetime import datetime

logging.basicConfig(level=10, filename="gathering_data.log")

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

autopilot_maps = [
	'/Game/Maps/Town01',
	'/Game/Maps/Town02'
	# '/Game/Maps/Town04'
]


def kill_carla_processes():
	for proc in psutil.process_iter():
		try:
			pinfo = proc.as_dict(attrs=['pid', 'name'])
		except psutil.NoSuchProcess:
			pass
		else:
			if (pinfo['name'] == "CarlaUE4.exe"):
				print(pinfo)
				proc.terminate()


def log(msg):
	logging.debug("[{}] {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


if __name__ == '__main__':
	log("Start gathering data...")

	frames_count = 1000
	config_file = 'C:\\Carla\\builds\\build1\\WindowsNoEditor\\Example.CarlaSettings.ini'
	carla_server_file = 'C:\\Carla\\builds\\build1\\WindowsNoEditor\\CarlaUE4.exe'
	gather_data_file = "C:\\Users\\Adam\\Documents\\Materialy_pw\\PW\\Magisterka\\AutonomousCarProject\\gather_data.py"

	log("frames_count: {}; config file: {}; carla file: {};".format(frames_count, config_file, carla_server_file))

	for autopilot_map in autopilot_maps:
		log("Start gathering data map: {}".format(autopilot_map))
		log("Create carla process...")
		carla_pid = subprocess.Popen(
			[carla_server_file, autopilot_map, "-carla-settings=" + config_file, "-ResX=800", "-ResY=600", "-windowed", "-carla-server"])

		for weather in range(14):
			log("Start gathering data map: {}; weather: {}".format(autopilot_map, weather_presets[weather]))
			# parameters = "-f {} -w {} -v".format(frames_count, weather)
			data_pid = subprocess.Popen(
				["python", gather_data_file, "-f", str(frames_count), "-w", str(weather), "-v", "-s", "10"])
			# os.system("gather_data.py {}".format(parameters))
			data_pid.wait()
			log("Finished gathering data...")

		log("Kill carla processes...")
		kill_carla_processes()
