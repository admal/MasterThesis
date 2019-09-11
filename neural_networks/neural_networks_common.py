from neural_networks.DenseNetModel import DenseNetModel
from neural_networks.ModelBase import ModelBase
from neural_networks.NvidiaModel import NvidiaModel
from neural_networks.ResNet50Model import ResNet50Model
from neural_networks.VGG16Model import VGG16Model


def get_empty_model(model_name, fine_tuning):
	if model_name == 'nvidia':
		return NvidiaModel().model(fine_tuning)
	elif model_name == 'vgg':
		return VGG16Model().model(fine_tuning)
	elif model_name == 'resnet':
		return ResNet50Model().model(fine_tuning)
	elif model_name == 'densenet':
		return DenseNetModel().model(fine_tuning)
	else:
		return None


def add_model_cmd_arg(parser):
	parser.add_argument('-m',
	                    help='choose model to train (vgg, nvidia, resnet, densenet)',
	                    dest='model',
	                    choices=['vgg', 'nvidia', 'resnet', 'densenet'],
	                    default='nvidia')


def load_model(model_name, checkpoint, fine_tuning):
	print("Loading model ({})...".format(model_name))

	model = get_empty_model(model_name, fine_tuning)
	model = ModelBase.load_weights(
		model,
		"trained_models\\{}\\{}-model-{:03d}.h5"
			.format(model_name, model_name, checkpoint))
	return model