import numpy as np
from enum import Enum
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

def relu(x):
	return x if x > 0 else 0

