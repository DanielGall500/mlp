import numpy as np
from src.layer import InputLayer, HiddenLayer

class Network:
	def __init__(self, input_size, \
		num_hidden_layers, num_hidden_units, \
		num_output_units, hidden_activation, \
		output_activation, weight_init='uniform',
		bias_init='ones'):
		self.input_size = input_size
		self.num_hidden_layers = num_hidden_layers
		self.num_hidden_units = num_hidden_units
		self.num_output_units = num_output_units
		self.cache = []

		self.weight_init = weight_init
		self.bias_init = bias_init

		self.il = self._init_input_layer(input_size)
		self.hl = self._init_hidden_layers(num_hidden_layers, \
			num_hidden_units, input_size, hidden_activation, weight_init,bias_init)
		self.ol = self._init_output_layer(num_output_units, input_size, \
			output_activation, weight_init, bias_init)

	def feed(self, I):
		if self._valid_input(I):
			#Empty cache
			self.cache = []

			#Feed Input Layer
			input_layer_output = self.il.feed(I)

			#Feed Hidden Layers
			layer_activations = input_layer_output
			for i, layer in enumerate(self.hl):
				#Pass through layer
				layer_outputs, layer_activations = layer.feed(layer_activations)

				#Store in the cache for backprop
				hidden_layer_info = self.get_layer_info(layer)
				self.cache.append(hidden_layer_info)

			#Feed Output Layer & Store Info in Cache
			self.ol.feed(layer_activations)
			output_layer_info = self.get_layer_info(self.ol)
			self.cache.append(output_layer_info)
		else:
			raise Exception("Network: Invalid Input {}".format(I))
		return self.cache

	def backward(self, dy):
		#Performs a backward pass of the network

		#Last layer: no activation
		num_layers = len(self.cache)
		output_layer = self.cache[num_layers-1]
		print(output_layer)

	def affine_backward(self, dout, I, w, b):
		dx, dw, db = None, None, None
		dw = np.reshape(x, (x.shape[0], -1)).T.dot(dout)
		dw = np.reshape(dw, w.shape)

		return dw

	def get_layer_info(self, layer):
		w, b = layer.get_weights_and_biases()
		l_outputs, l_activations = layer.get_output_and_activations()
		l_outputs = l_outputs.tolist()
		l_activations = l_activations.tolist()
		return {'outputs': l_outputs,\
				'activations': l_activations, \
				'weights': w, \
				'biases': b }

	def print(self):
		print("--Input Layer--")
		print("Size: ", self.input_size)
		print("Input: ", self.il.get_input())

		for i, layer in enumerate(self.hl):
			print("\n--Hidden Layer {}--".format(i+1))
			print("Input: ", layer.get_input())

			w,b = layer.get_weights_and_biases()
			print("Weights: ", w)
			print("Biases: ", b)

			print("Output: ", layer.get_output())

		print("\n--Output Layer--")
		w,b = self.ol.get_weights_and_biases()
		print("Input: ", self.ol.get_input())
		print("Weights: ", w)
		print("Biases: ", b)
		print("Output: ", self.ol.get_output())
		print("----\n")

	def get_cache(self):
		return self.cache

	def _valid_input(self, I):
		return len(I) == self.input_size

	def _init_input_layer(self, input_size):
		return InputLayer(input_size)

	def _init_hidden_layers(self, num_layers, num_units, \
		input_size, activation, weight_init='uniform', bias_init='zeros'):
		hidden_layers = []
		for i in range(num_layers):
			if i == 0:
				#Input length = Number of data samples
				layer = HiddenLayer(num_units, input_size, activation,\
				 weight_init, bias_init)
			else:
				#Input length = Number of units in previous layer
				layer = HiddenLayer(num_units, num_units, activation, \
					weight_init, bias_init)
			hidden_layers.append(layer)
		return hidden_layers

	def _init_output_layer(self, num_units, input_size, activation, weight_init, bias_init):
		return HiddenLayer(num_units, self.num_hidden_units, activation,\
			weight_init, bias_init)


