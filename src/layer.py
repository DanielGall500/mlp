from perceptron.perceptron import Perceptron
from perceptron.perceptron import Unit, InputUnit
from perceptron.perceptron import activation
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
		pass

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
			return True
		return False

	def get_input(self):
		layer_input = []
		for unit in self.get_units():
			layer_input.append(unit.get_output())
		return layer_input

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
		activation: activation.FunctionType):
		super(HiddenLayer, self).__init__(num_units)
		self.inputs_per_unit = inputs_per_unit
		self.input = None
		self.units = []

		for i in range(num_units):
			percep = Perceptron(inputs_per_unit, activation)
			self.units.append(percep)

	def feed(self, I) -> bool:
		if self._valid_input(I):
			self.input = I
			for perc in self.units:
				perc.feed(I)
			return True
		return False

	def get_input(self) -> np.array:
		return self.input

	def get_output(self) -> np.array:
		output = []
		for perc in self.units:
			output.append(perc.get_output())
		return np.array(output)

	def size(self):
		pass

	def get_units(self):
		return self.units

	def _valid_input(self, I):
		return I.shape[0] == self.inputs_per_unit
	

class OutputLayer(Layer):
	def __init__(self):
		return None

	def get_units(self):
		pass

	def add_unit(self, unit):
		pass

	def delete_unit(self, unit):
		pass