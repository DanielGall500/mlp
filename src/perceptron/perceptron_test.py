import unittest
import activation
import numpy as np
from perceptron import Perceptron
from activation import FunctionType

class TestPerceptron(unittest.TestCase):
	def test_initialisation(self):
		p = Perceptron(2,2)
		self.assertEqual(p.number_of_inputs, 2)
		self.assertEqual(p.number_of_outputs, 2)

	def test_weight_initialisation(self):
		p = Perceptron(2,2)
		self.assertEqual(p.weights.size, 2)

	def test_bias_initialisation(self):
		p = Perceptron(2,2)
		self.assertEqual(p.biases.size, 2)

	def test_compute_without_activation(self):
		p = Perceptron(2,2)
		inputs = np.array([5,10])
		compute_without_activation = p.compute(inputs, \
			with_activation=False)

		self.assertEqual(compute_without_activation, 17)

	def test_compute_with_activation(self):
		p = Perceptron(2,2, FunctionType.SIGMOID)
		inputs = np.array([5,10])
		compute_with_activation = p.compute(inputs, \
			with_activation=True)

		self.assertEqual(compute_with_activation, activation.sigmoid(17))


if __name__ == "__main__":
	unittest.main()
