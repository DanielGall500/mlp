
import numpy as np
from enum import Enum
import activation
from activation import FunctionType

"""
--Perceptron Class--
Takes input X and calculates Sum(wX + b)
"""
class Perceptron:
	def __init__(self, num_inputs, num_outputs, \
		activation = FunctionType.NONE):
		self.number_of_inputs = num_inputs
		self.number_of_outputs = num_outputs
		self.weights = self._init_weights(num_inputs)
		self.biases = self._init_weights(num_outputs)
		self.activation = activation
		return None

	def compute(self, inputs: np.array, with_activation=True) -> np.array:
		if self._valid_input(inputs):
			#Calculate Sum[wX + b]
			multiply_weights = np.multiply(self.weights, inputs)
			add_bias = np.add(multiply_weights, self.biases)
			sum_together = np.sum(add_bias)

			#Either apply activation function or not
			if with_activation:
				return self._activate(sum_together)
			else:
				return sum_together

	def _valid_input(self, inputs) -> bool:
		return (len(inputs) == self.number_of_inputs) \
		and (inputs.ndim == 1)

	def _activate(self, x):
		a = self.activation
		return activation.apply_activation(x, a)

	def _init_weights(self, num_inputs) -> np.array:
		return np.ones(num_inputs)

	def _init_biases(self, num_inputs) -> np.array:
		return np.ones(num_inputs)
