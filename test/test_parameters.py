import unittest
from src.unit.activation import WeightsCreator

class ParameterTest(unittest.TestCase):
	def test_weights(self):
		w = WeightsCreator(num_weights=4)
		weights_uni = w.get_uniform()
		weights_rand = w.get_random()
		self.assertEqual(len(weights_uni), 4)
		self.assertEqual(len(weights_rand), 4)