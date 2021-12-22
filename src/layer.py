from src.unit.perceptron import Perceptron
from src.unit.perceptron import Unit, InputUnit
from src.unit.activation import FunctionType
import numpy as np


class Layer:
	def __init__(self, num_units):
		self.number_of_units = num_units

	def feed_input(self):
		pass

	def get_input(self):
		pass

	def get_output(self):
		pass

	def size(self):
		return self.number_of_units

	def get_units(self):
		pass

class InputLayer(Layer):
	def __init__(self, num_units):
		super().__init__(num_units)

		self.units = []
		for i in range(num_units):
			self.units.append(InputUnit())

	def feed(self, I: np.array) -> bool:
		if self._valid_input(I):
			for i, x in enumerate(I):
				unit = self.units[i]
				self.units[i].store(x)
		return self.get_output()

	def get_input(self):
		layer_input = []
		for unit in self.get_units():
			layer_input.append(unit.get_output())
		return np.array(layer_input)

	def get_output(self):
		return self.get_input()

	def size(self):
		return self.number_of_units

	def get_units(self):
		return self.units

	def _valid_input(self, inp):
		return len(inp) == self.size()

class HiddenLayer(Layer):
	def __init__(self, num_units, inputs_per_unit, \
		activation: FunctionType, weight_init='uniform', bias_init='zeros'):
		super(HiddenLayer, self).__init__(num_units)
		self.inputs_per_unit = inputs_per_unit
		self.weight_init = weight_init
		self.input = None
		self.units = []

		for i in range(num_units):
			percep = Perceptron(inputs_per_unit, activation, \
				weight_init=weight_init, bias_init=bias_init)
			self.units.append(percep)

	def feed(self, I):
		if self._valid_input(I):
			self.input = I
			for perc in self.units:
				perc.feed(I)
		else:
			raise Exception("Hidden Layer: Invalid Input {}".format(I))
		return self.get_output_and_activations()

	def get_input(self) -> np.array:
		return self.input

	def get_output_and_activations(self) -> np.array:
		outputs = []
		activations = []
		for perc in self.units:
			output, activation = perc.get_output_and_activation()
			outputs.append(output)
			activations.append(activation)
		return np.array(outputs), np.array(activations)

	def get_weights_and_biases(self):
		weights = []
		biases = []
		for unit in self.units:
			w = unit.get_weights()
			b = unit.get_bias()
			weights.append(w)
			biases.append(b)
		return weights, biases

	def get_units(self):
		return self.units

	def _valid_input(self, I):
		print("----")
		print(I)
		print(self.inputs_per_unit)
		print("----")
		return len(I) == self.inputs_per_unit
	

class OutputLayer(Layer):
	def __init__(self):
		return None

	def get_units(self):
		pass

	def add_unit(self, unit):
		pass

	def delete_unit(self, unit):
		pass
