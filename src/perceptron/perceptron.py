
import numpy as np
from enum import Enum
from activation import Activation

"""
--Perceptron Class--
Takes input X and calcualtes Sum(wX + b)
"""
class Perceptron:
	def __init__(self, num_inputs, num_outputs, \
		activation: Activation = Activation.SIGMOID):
		self.number_of_inputs = num_inputs
		self.number_of_outputs = num_outputs
		self.weights = self._init_weights(num_inputs)
		self.biases = self._init_weights(num_outputs)
		self.activation = activation
		return None

	def forward_step(self, inputs, with_activation=True) -> np.array:
		if _valid_input(inputs):
			#Calculate Sum[wX + b]
			multiply_weights = np.multiply(self.wseights,inputs)
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

		if a == Activation.SIGMOID:
			return Activation.sigmoid(x)
		elif a == Activation.RELU:
			return Activation.relu(x)
		elif a == Activation.NONE:
			return x

	def _init_weights(self, num_inputs) -> np.array:
		return np.zeros(num_inputs)

	def _init_biases(self, num_inputs) -> np.array:
		return np.zeros(num_inputs)
