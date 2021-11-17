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
		cache = n.get_cache()
		hl0 = [round(x,4) for x in cache[0]['activations']]
		hl1 = [round(x,4) for x in cache[1]['activations']]
		hl2 = [round(x,4) for x in cache[2]['activations']]

		self.assertEqual(hl0, [0.9975, 0.9975])
		self.assertEqual(hl1, [0.9524, 0.9524])
		self.assertEqual(hl2, [2.9047])

	def test_network_run(self):
		n = Network(input_size=1, num_hidden_layers=2, \
			num_hidden_units=2, num_output_units=1, \
			hidden_activation=FunctionType.SIGMOID, \
			output_activation=FunctionType.NONE, \
			weight_init='random', bias_init='zeros')

		I = np.array([5])
		n.feed(I)
		cache = n.get_cache()
		"""
		for i,layer in enumerate(cache):
			print("Layer {}".format(i))
			print(layer)
		"""

