import numpy as np

class Network:
	def __init__(self, num_input, num_hidden, num_output, input_size):
		self.num_input_units = num_input
		self.num_hidden_units = num_hidden
		self.num_output_units = num_output
		self.input_size = input_size

