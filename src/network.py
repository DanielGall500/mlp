import numpy as np
from src.layer import InputLayer, HiddenLayer

class Network:
	def __init__(self, input_size, \
		num_hidden_layers, num_hidden_units, \
		num_output_units, hidden_activation, \
		output_activation):
		self.input_size = input_size
		self.num_hidden_layers = num_hidden_layers
		self.num_hidden_units = num_hidden_units
		self.num_output_units = num_output_units

		self.il = self._init_input_layer(input_size)
		self.hl = self._init_hidden_layers(num_hidden_layers, \
			num_hidden_units, input_size, hidden_activation)
		self.ol = self._init_output_layer(num_output_units, input_size, \
			output_activation)

	def feed(self, I):
		if self._valid_input(I):
			#Feed Input Layer
			input_layer_output = self.il.feed(I)

			#Feed Hidden Layers
			layer_output = input_layer_output
			for i, layer in enumerate(self.hl):
				layer_output = layer.feed(layer_output)

			#Feed Output Layer
			layer_output = self.ol.feed(layer_output)
		else:
			raise Exception("Network: Invalid Input")
		return layer_output

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

	def _valid_input(self, I):
		return len(I) == self.input_size

	def _init_input_layer(self, input_size):
		return InputLayer(input_size)

	def _init_hidden_layers(self, num_layers, num_units, \
		input_size, activation):
		hl = HiddenLayer(num_units, input_size, activation)
		hidden_layers = []
		for i in range(num_layers):
			if i == 0:
				#Input length = Number of data samples
				layer = HiddenLayer(num_units, input_size, activation)
			else:
				#Input length = Number of units in previous layer
				layer = HiddenLayer(num_units, num_units, activation)
			hidden_layers.append(layer)
		return hidden_layers

	def _init_output_layer(self, num_units, input_size, activation):
		return HiddenLayer(num_units, self.num_hidden_units, activation)


