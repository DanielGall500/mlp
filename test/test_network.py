from src.unit.activation import FunctionType
from src.network import Network
import numpy as np
import unittest

class NetworkTest(unittest.TestCase):
	def test_init(self):
		n = Network(input_size=2, num_hidden_layers=2, \
			num_hidden_units=2, num_output_units=2, \
			hidden_activation=FunctionType.SIGMOID, \
			output_activation=FunctionType.NONE)

		self.assertEqual(n.input_size, 2)
		self.assertEqual(n.num_hidden_layers, 2)
		self.assertEqual(n.num_hidden_units, 2)
		self.assertEqual(n.num_output_units, 2)

	def test_forward_pass(self):
		n = Network(input_size=1, num_hidden_layers=2, \
			num_hidden_units=2, num_output_units=1, \
			hidden_activation=FunctionType.SIGMOID, \
			output_activation=FunctionType.NONE)

		I = np.array([5])
		n.feed(I)
		n.print()


if __name__ == "__main__":
	unittest.main()