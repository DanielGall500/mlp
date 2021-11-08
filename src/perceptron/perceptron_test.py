import unittest
from perceptron import Perceptron

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

	def test_activation(self):
		pass

	def test_forward_step(self):
		pass

if __name__ == "__main__":
	unittest.main()
