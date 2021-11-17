import numpy as np
from enum import Enum
import src.unit.activation as activation
from src.unit.activation import FunctionType, WeightsCreator

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
		activation = FunctionType.NONE, \
		weight_init='uniform', bias_init='ones'):

		#Basic Unit Parent
		super(Perceptron,self).__init__(num_inputs)
		self.input = None

		#Weights & Biases
		self.weight_init = weight_init
		self.bias_init = bias_init

		self.w = self._init_weights(num_inputs, weight_init)
		self.dW = None
		self.b = self._init_biases(num_inputs, bias_init)

		#Result of forward pass through perceptron
		self.activation = activation
		self.output = None
		return None

	def feed(self, I: np.array) -> np.array:
		#Ensure input is of the right size and dimension
		if self._valid_input(I):
			#Store input
			self.input = I

			unit_calculations = []

			#Calculate Sum[wI + b]
			wI = np.multiply(self.w, I)
			sum_together = np.sum(wI)
			plus_b = np.add(sum_together, self.b)

			#Calculating Activation(Perceptron Output)
			self.output = self._activate(plus_b)
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

	def get_bias(self):
		return self.b

	def get_input(self):
		return self.input

	def get_output(self):
		return self.output

	def get_derivative(self):
		if self.output != None:
			if self.activation == FunctionType.SIGMOID:
				return activation.sigmoid_derivative(self.output)
			else:
				raise Exception("Invalid Activation Function")
		else:
			raise Exception("Perceptron Has No Previous Output")


	def _init_weights(self, num_weights, init_type) -> np.array:
		w_initialiser = WeightsCreator(num_weights)
		return w_initialiser.get(init_type)

	def _init_biases(self, num_inputs, init_type) -> np.array:
		if init_type == 'ones':
			return 1
			#return np.ones(num_inputs)
		elif init_type == 'zeros':
			return 0
			#return np.zeros(num_inputs)
		else:
			raise Exception("Invalid Bias Initialisation: {}"\
				.format(init_type))

	def _activate(self, x):
		a = self.activation
		return activation.apply_activation(x, a)

	def _valid_input(self, inputs) -> bool:
		return (len(inputs) == self.number_of_inputs)
