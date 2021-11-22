import numpy as np
import matplotlib.pyplot as plt

def main():
	NI = 2 #Number of inputs
	NH = 3 #Number of hidden units
	NO = 1 #Number of outputs

	#Weights first layer (NI,NH)
	W1 =  np.array([[0.1, 0.3, 0.6], \
				    [0.2, 0.4, 0.7]])

	print("W1 Shape: ", W1.shape)
	print("(NI, NH): ({}, {})\n".format(NI,NH))

	#Weights second layer (NH,NO)
	W2 = np.array([[0.3],\
				   [0.4],\
				   [0.5]])

	print("W2 Shape: ", W2.shape)
	print("(NH,NO): ({},{})\n".format(NH,NO))

	#Biases first layer (,NH)
	B1 = np.array([0, 0, 0])

	print("B1 Shape: ", B1.shape)
	print("(NH,): ({},)\n".format(NH))

	#Biases second layer (,NO)
	B2 = np.array([0])

	print("B2 Shape: ", B2.shape)
	print("(NO,): ({},)\n".format(NO))

	dW1 = np.empty(W1.shape) #Weight *changes* to be applied to W1
	dW2 = np.empty(W2.shape) #Weight *changes to be applied to W2

	dB1 = np.empty(B1.shape)
	dB2 = np.empty(B2.shape)

	Z1 = np.empty((1,NH)) #Activations first layer (,NH)
	Z2 = [] #Activations second layer (,NO)

	H = np.empty((1,NH)) #Where values of hidden neurons are stored (,NH)
	O = np.empty((1,NO)) #Where outputs are stored (,NO)

	X = np.array([[0,0],\
				  [0,1], \
				  [1,0], \
				  [1,1]])

	y_true = np.array([[0],\
			      [1],\
				  [1],\
				  [0]])

	print("X Shape: ", X.shape)

	errors = []
	weight_changes = []
	for i in range(20000):
		#Forward Pass
		H, cache_H = affine_forward(X, W1, B1)
		#print("H: ", H.shape)
		#print("(Number samples, {})\n".format(NH))

		Z1, cache_Z1 = Sigmoid().forward(H)
		#print("Z1: ", Z1.shape)
		#print("(Number samples, {})\n".format(NH))

		O, cache_O = affine_forward(Z1, W2, B2)
		#print("O: ", O.shape)
		#print("(Number samples, {})\n".format(NO))

		Z2, cache_Z2 = Sigmoid().forward(O)
		#print("Z2: ", Z2.shape)
		#print("(Number samples, {})".format(NO))

		error = L1().forward(Z2, y_true)
		errors.append(error)
		#print("Error: {}\n".format(round(error,4)))

		#Backward pass
		dout = L1().backward(Z2, y_true)
		#print("dout: {}".format(dout.shape))
		#print("(Number samples, 1)\n")

		dZ2 = Sigmoid().backward(dout, cache_Z2)
		#print("dZ2: {}".format(dZ2.shape))
		#print("{}\n".format(cache_Z2.shape))

		dO2, dW2, dB2 = affine_backward(dZ2, cache_O)
		#print("dO: {}\ndW2: {}\ndB2: {}\n".format(dO2.shape,\
		#	dW2.shape,dB2.shape))

		dZ1 = Sigmoid().backward(dO2, cache_Z1)
		#print("dZ1: {}\n".format(dZ1.shape))

		dO1, dW1, dB1 = affine_backward(dZ1, cache_H)
		#print("dO: {}\ndW2: {}\ndB2: {}\n".format(dO1.shape,\
		#	dW1.shape,dB1.shape))

		#Step
		LR = 0.1
		W1 = step(W1, dW1)
		W2 = step(W2, dW2)

		B1 = step(B1, dB1)
		B2 = step(B2, dB2)

	plt.plot(errors)
	plt.ylim([0,1])
	plt.show()

def step(w, dw, lr=0.01):
	return w - lr*dw

def affine_forward(x, w, b):
	out = None
	x_reshaped = np.reshape(x, (x.shape[0], -1))

	#Output of shape (N, M) - (4 Samples, 3 Hidden Units)
	out = x.dot(w) + b
	cache = (x, w, b)
	return out, cache

def affine_backward(dout, cache):
	x, w, b = cache
	dx, dw, db = None, None, None

	"""
	Derivative dw
	x [0 0]
	dout [5]
	x.T . dout =
	[ 0 ]
	[ 0 ]
	 """
	dw = np.reshape(x, (x.shape[0],-1)).T.dot(dout)
	dw = np.reshape(dw, w.shape)

	db = np.sum(dout, axis=0, keepdims=False)

	dx = dout.dot(w.T)
	dx = np.reshape(dx, x.shape)
	return dx, dw, db


class Sigmoid:
	def __init__(self):
		pass

	def forward(self, x):
		outputs = 1 / (1 + np.exp(-x))
		cache = outputs
		return outputs, cache

	def backward(self, dout, cache):
		dx = None
		dx = dout * cache * (1 - cache)
		return dx

class L1:
	def __init__(self):
		pass

	def forward(self, y_out, y_true):
		result = np.abs(y_out - y_true)
		result = result.mean()
		return result

	def backward(self, y_out, y_true):
		gradient = y_out - y_true

		#Index where 0
		zero_loc = np.where(gradient == 0)

		#Index where negative
		negative_loc = np.where(gradient < 0)

		#Index where positive
		positive_loc = np.where(gradient > 0)

		gradient[zero_loc] = 0
		gradient[positive_loc] = 1
		gradient[negative_loc] = -1
		return gradient

if __name__ == "__main__":
	main()

