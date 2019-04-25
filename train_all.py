import subprocess
import logging
import time

FORMAT = '%(asctime)-15s : %(message)s'
nets = ["nvidia", "densenet", "resnet", "vgg"]
batches = {
	"nvidia": 150,
	"densenet": 50,
	"resnet": 30,
	"vgg": 90
}

if __name__ == '__main__':
	logging.basicConfig(filename="training.log", format=FORMAT)
	logging.getLogger().setLevel(logging.DEBUG)

	logging.info("Start training...")
	train_script = "train.py"
	epochs_count = 100
	for net in nets:
		logging.info("Start training {} net (epochs: {})".format(net, epochs_count))
		pid = subprocess.Popen(
			[
				train_script,
				"-n " + str(epochs_count),
				"-b " + str(batches[net]),
				"-o true"]
		)
		pid.wait()
		logging.info("Finished training {}".format(net))

	logging.info("Finished all training, wait 30 seconds to computer shutdown")
	time.sleep(30)
	logging.info("Computer shutdown")
	subprocess.call(["shutdown", "-f", "-s", "-t", "60"])