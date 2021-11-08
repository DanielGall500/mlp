import numpy as np
from enum import Enum
import math

class Activation(Enum):
	NONE = 0
	SIGMOID = 1
	RELU = 2

def sigmoid(x):
	return 1 / (1 + exp(-x))

def relu(x):
	return x if x > 0 else 0

