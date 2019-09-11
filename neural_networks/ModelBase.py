class ModelBase:
	@staticmethod
	def load_weights(model, filename):
		model.load_weights(filename)
		return model
