from neural_networks.DenseNetModel import DenseNetModel
from neural_networks.ModelBase import ModelBase
from neural_networks.NvidiaModel import NvidiaModel
from neural_networks.ResNet50Model import ResNet50Model
from neural_networks.VGG16Model import VGG16Model


def get_empty_model(model_name):
	if model_name == 'nvidia':
		return NvidiaModel().model()
	elif model_name == 'vgg':
		return VGG16Model().model()
	elif model_name == 'resnet':
		return ResNet50Model().model()
	elif model_name == 'densenet':
		return DenseNetModel().model()
	else:
		return None


def add_model_cmd_arg(parser):
	parser.add_argument('-m',
	                    help='choose model to train (vgg, nvidia, resnet, densenet)',
	                    dest='model',
	                    choices=['vgg', 'nvidia', 'resnet', 'densenet'],
	                    default='nvidia')


def load_model(model_name, checkpoint):
	print("Loading model ({})...".format(model_name))

	model = get_empty_model(model_name)
	model = ModelBase.load_weights(
		model,
		"trained_models\\{}\\{}-model-{:03d}.h5"
			.format(model_name, model_name, checkpoint))
	return model