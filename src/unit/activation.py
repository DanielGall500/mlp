import numpy as np
from enum import Enum
from random import randint
import math

class FunctionType(Enum):
	NONE = 0
	SIGMOID = 1
	RELU = 2

def apply_activation(x, a: FunctionType):
	if a == FunctionType.SIGMOID:
		return sigmoid(x)
	elif a == FunctionType.RELU:
		return relu(x)
	elif a == FunctionType.NONE:
		return x

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
	return x * (1.0 - x)

def relu(x):
	return x if x > 0 else 0

#Weights & Biases Initialisation
class WeightsCreator:
	def __init__(self, num_weights):
		self.num_weights = num_weights

	def get(self, init_type):
		if init_type == 'uniform':
			return self.get_uniform()
		elif init_type == 'random':
			return self.get_random()
		else:
			raise Exception("Invalid Weight Initialisation: {}"\
				.format(init_type))

	def get_uniform(self):
		uniform = lambda x: np.ones(x)
		return uniform(self.num_weights)

	def get_random(self):
		#Retrieves random value between 0 and 1
		random = lambda a,b: round(randint(a,b) / float(b), 4)
		return [random(0,1000) for i in range(self.num_weights)]









