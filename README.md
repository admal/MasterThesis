# MasterThesis
Master thesis repository

# Requirements
Python 3.6 is used.  
To run project there are several libraries needed:
* numpy
* tensorflow
* pandas
* sckitlearn
* fastdtw
* argparse
* matplotlib
* scipy
* pygame (+working joystick for data gathering)
* Carla simulator 0.8.4

# Important scripts
* Data gathering - `gather_data.py`
* Running autonomous drive - `drive.py`
* Data summary - `data_summary.py`
* Gather ideal driving line - `create_racing_line.py`


# How to run carla for collecting data and autonomous drive
```
CarlaUE4.exe {MAP} -carla-settings=CarlaSettings.ini -windowed -ResX=800 -ResY=600 -carla-server
```
Available maps:
* /Game/Maps/Town01
* /Game/Maps/Town02
# Run test
```
python drive.py -v -w {WEATHER_PRESET} -m {MODEL_NAME} -c {MODEL_EPOCH} -t {MAP}
```