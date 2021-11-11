import unittest
from layer import InputLayer, HiddenLayer
from perceptron.activation import FunctionType
import numpy as np

class TestInputLayer(unittest.TestCase):
	def test_init(self):
		il = InputLayer(4)
		pass

	def test_get_units(self):
		il = InputLayer(4)
		units = il.get_units()
		self.assertEqual(len(units),4)

	def test_feed_input(self):
		example_input = [5,10,15,20]
		il = InputLayer(4)
		il.feed(example_input)
		layer_inp = il.get_input()
		self.assertEqual(layer_inp, example_input)

class TestHiddenLayer(unittest.TestCase):
	def test_feed(self):
		hl = HiddenLayer(2,2, FunctionType.SIGMOID)
		I = np.array([5,10])
		hl.feed(I)

		for perc in hl.units:
			self.assertCountEqual(perc.get_input(), I)

	def test_input(self):
		hl = HiddenLayer(2,2, FunctionType.SIGMOID)
		I = np.array([5,10])
		hl.feed(I)

		self.assertCountEqual(hl.get_input(), I)

	def test_output(self):
		hl = HiddenLayer(2,2, FunctionType.NONE)
		I = np.array([5,10])
		hl.feed(I)
		output = hl.get_output()

		for i, perc in enumerate(hl.units):
			self.assertEqual(perc.get_output(), output[i])


if __name__ == "__main__":
	unittest.main()