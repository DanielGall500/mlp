import unittest
import numpy as np
from src.unit.perceptron import Perceptron
from src.unit.activation import FunctionType
import src.unit.activation as activation

class TestPerceptron(unittest.TestCase):
	def test_input_initialisation(self):
		p = Perceptron(2)
		self.assertEqual(p.number_of_inputs, 2)

	def test_weight_initialisation(self):
		p = Perceptron(2)
		self.assertEqual(p.get_weights().size, 2)

	def test_bias_initialisation(self):
		p = Perceptron(2)
		self.assertEqual(p.get_bias(), 1)

	def test_feed_without_activation(self):
		p = Perceptron(2)
		inputs = np.array([5,10])
		output, activation = p.feed(inputs)
		self.assertEqual(activation, 16)

	def test_feed_with_activation(self):
		p = Perceptron(2, FunctionType.SIGMOID)
		inputs = np.array([5,10])
		output, activations = p.feed(inputs)
		self.assertEqual(activations, activation.sigmoid(16))

	def test_weight_changes(self):
		p = Perceptron(2, FunctionType.SIGMOID)
		w = p.get_weights()

		weight_updates = np.array([-1,5])
		p.apply_weight_changes(weight_updates)
		new_weights = p.get_weights()

		self.assertCountEqual(new_weights, np.array([0,6]))

	def test_output_derivative(self):
		p = Perceptron(2, FunctionType.SIGMOID)
		inputs = np.array([5,10])
		compute_with_activation = p.feed(inputs)
		print("\nOutput Derivative: {}".format(p.get_derivative()))

