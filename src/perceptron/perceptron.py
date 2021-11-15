import numpy as np
from enum import Enum
from perceptron.activation import FunctionType
import perceptron.activation

"""
--Perceptron Class--
Takes input X and calculates Sum(wX + b)
"""
class Unit:
	def __init__(self, num_inputs):
		self.number_of_inputs = num_inputs
		self.input = []
		self.output = []

	def __str__(self):
		return self.input

class InputUnit(Unit):
	def __init__(self):
		self.input = None

	def store(self, I):
		self.input = I

	def get_output(self):
		return self.input
	

class Perceptron(Unit):
	def __init__(self, num_inputs, \
		activation = FunctionType.NONE):

		#Basic Unit Parent
		super(Perceptron,self).__init__(num_inputs)
		self.input = None

		#Weights & Biases
		self.w = self._init_weights(num_inputs)
		self.dW = None
		self.b = self._init_biases(num_inputs)

		#Result of forward pass through perceptron
		self.activation = activation
		self.output = None
		return None

	def feed(self, I: np.array) -> np.array:
		#Ensure input is of the right size and dimension
		if self._valid_input(I):
			#Store input
			self.input = I

			#Calculate Sum[wI + b]
			wI = np.multiply(self.w, I)
			plus_b = np.add(wI, self.b)
			sum_together = np.sum(plus_b)

			#Calculating Activation(Perceptron Output)
			self.output = self._activate(sum_together)
		else:
			raise Exception("Perceptron: Invalid Input {}".format(I))

		return self.output

	def apply_weight_changes(self, dW: np.array):
		self.dW = dW
		self.w = np.add(self.w, self.dW)
		return self.w

	def get_weights(self):
		return self.w

	def get_weight_changes(self):
		return self.dW

	def get_biases(self):
		return self.b

	def get_input(self):
		return self.input

	def get_output(self):
		return self.output

	def _init_weights(self, num_inputs) -> np.array:
		return np.ones(num_inputs)

	def _init_biases(self, num_inputs) -> np.array:
		return np.ones(num_inputs)

	def _activate(self, x):
		a = self.activation
		return perceptron.activation.apply_activation(x, a)

	def _valid_input(self, inputs) -> bool:
		return (len(inputs) == self.number_of_inputs)
