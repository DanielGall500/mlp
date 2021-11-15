import unittest
import numpy as np
from perceptron import Perceptron
from activation import FunctionType
import activation

class TestPerceptron(unittest.TestCase):
	def test_input_initialisation(self):
		p = Perceptron(2)
		self.assertEqual(p.number_of_inputs, 2)

	def test_weight_initialisation(self):
		p = Perceptron(2)
		self.assertEqual(p.get_weights().size, 2)

	def test_bias_initialisation(self):
		p = Perceptron(2)
		self.assertEqual(p.get_biases().size, 2)

	def test_feed_without_activation(self):
		p = Perceptron(2)
		inputs = np.array([5,10])
		compute_without_activation = p.feed(inputs)
		self.assertEqual(compute_without_activation, 17)

	def test_feed_with_activation(self):
		p = Perceptron(2, FunctionType.SIGMOID)
		inputs = np.array([5,10])
		compute_with_activation = p.feed(inputs)
		self.assertEqual(compute_with_activation, activation.sigmoid(17))

	def test_weight_changes(self):
		p = Perceptron(2, FunctionType.SIGMOID)
		w = p.get_weights()

		weight_updates = np.array([-1,5])
		p.apply_weight_changes(weight_updates)
		new_weights = p.get_weights()

		self.assertCountEqual(new_weights, np.array([0,6]))


if __name__ == "__main__":
	unittest.main()
