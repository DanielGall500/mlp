
import numpy as np

"""
Perceptron:

Variables
- I inputs
- w Weights
- b Biases
- O Outputs
- Activation Function

Function:
Return Sum(wI + b)
"""

class Perceptron:
	def __init__(self, num_inputs, num_outputs):
		self.number_of_inputs = num_inputs
		self.number_of_outputs = num_outputs
		self.weights = self._init_weights(num_inputs)
		self.biases = self._init_weights(num_outputs)
		return None

	def _init_weights(self, num_inputs) -> np.array:
		return np.zeros(num_inputs)

	def _init_biases(self, num_inputs) -> np.array:
		return np.zeros(num_inputs)
