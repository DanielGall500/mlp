import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from math import sqrt
import pandas as pd
import math

def experiment_XOR_function():
	print("MLP Training & Testing: Experiment 1")
	print("XOR Function")
	print("------------")

	#--Parameters--
	NUM_ITERATIONS = 1000
	LEARNING_RATE = 0.2
	NI = 2
	NH = 4
	NO = 1
	activation = Sigmoid
	loss = MSE

	#XOR data
	X = np.array([[0,0], \
				  [0,1], \
				  [1,0], \
				  [1,1]] )

	y = np.array([[0],\
			      [1],\
				  [1],\
				  [0]])

	#Train MLP
	mlp = MLP(NI=NI, NH=NH, NO=NO, activation=activation, loss=loss)
	mlp.train(X, y, NUM_ITERATIONS, LEARNING_RATE)

	#Test model on XOR data
	print("Time to test it out!")
	for sample in X:
		prediction = mlp.predict(sample)
		print("Sample: {}".format(sample))
		print("Prediction: {}\n".format(prediction))

	plot_error(mlp.error_info, "MLP: XOR Training Error (LR=0.2)")

def experiment_sine_function():
	print("MLP Training & Testing: Experiment 2")
	print("Sine Function")
	print("-------------")

	#--Parameters--
	NUM_EPOCHS = 30
	LEARNING_RATE = 0.01
	NI = 4 
	NH = 5
	NO = 1
	activation = Tanh
	loss = MSE

	#Construct dataset of uniform random integers in range (-1,+1)
	X = np.random.uniform(low=-1, high=1, size=(500,4))
	y = np.array([math.sin(x[0]-x[1]+x[2]-x[3]) for x in X])

	#Split into training & testing data
	training_split = 400
	test_split = X.shape[0]-training_split
	train_X, train_y, test_X, test_y = train_test_split(X,y,training_split)
	train_y = np.reshape(train_y, (training_split, 1))
	test_y = np.reshape(test_y, (test_split, 1))

	mlp = MLP(NI=NI, NH=NH, NO=NO, activation=activation, loss=loss)
	mlp.train_stochastic(train_X, train_y, NUM_EPOCHS, LEARNING_RATE)

	#Ensure sine function is modelled correctly
	print("Time to test it out!")
	y_pred = []
	for i,y in enumerate(test_y):
		pred = mlp.predict_no_threshold(test_X[i])
		y_pred.append(pred)

	accuracy = L1().forward(y_pred, test_y)
	print("MSE Error on Test Set: {}".format(accuracy))
	plot_error(mlp.error_info, "Q3: Modelling sin(x1-x2+x3-x4)")

	test_X_summed = [x[0]-x[1]+x[2]-x[3] for x in test_X]

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.scatter(test_X_summed, test_y, s=10, c='b')
	ax1.scatter(test_X_summed, y_pred, s=10, c='r')
	plt.show()

def experiment_handwritten_letters():
	print("MLP Training & Testing: Experiment 3")
	print("UCI Handwritten Letter Recognition")
	print("----------------------------------")

	#--Parameters--
	NUM_ITERATIONS = 1000
	LEARNING_RATE = 3
	NI = 16 #Number of inputs
	NH = 14 #Number of hidden units
	NO = 26 #Number of outputs
	activation=Sigmoid
	loss=MCCE

	#Import handwritten letter dataset
	df = pd.read_csv('data/letter-recognition.data', header=None)
	X = df.iloc[:, 1:].to_numpy()
	y = df.iloc[:, 0]

	#Encode letters as numerical values
	y_codes = y.astype('category').cat.codes
	y_dict = dict(enumerate(y.astype('category').cat.categories))

	#Split the model into training & testing data
	train_X, train_y, test_X, test_y = train_test_split(X, y_codes, 16000)

	#Train and test the MLP model
	mlp = MLP(NI, NH, NO, activation=activation, loss=loss)
	mlp.train(train_X, train_y, NUM_ITERATIONS, LEARNING_RATE)
	results = mlp.predict_img(test_X, test_y, y_dict)

	#The target variable test set
	true_letters = [y_dict[x] for x in test_y]

	#Count how many were correctly classified
	counter = 0
	for x,y in zip(results, true_letters):
		counter += int(x == y)

	print("Accuracy: {}%".format((counter / len(true_letters))*100))

	plot_error(mlp.error_info, "MLP: UCI Letter Recognition Training Error")

def plot_error(error_info, title):
	plt.plot(error_info)
	plt.title(title)
	plt.xlabel('Epoch', fontsize=12)
	plt.ylabel('Error', fontsize=12)
	plt.show()

def train_test_split(X, y, num_train):
	num_test = len(X) - num_train
	train_X, train_y= X[:num_train, :], y[:num_train]
	test_X, test_y = X[-num_test:, :], y[-num_test:]
	return train_X, train_y, test_X, test_y

"""
--Multi-Layer Perceptron Class--
Creates an MLP with an input layer, hidden layer, and output layer.
MLP can train on various datasets and model the data to make predictions.

NI => Number of Inputs
NH => Number of Hidden Units
NO => Number of Outputs
Activation => Sigmoid or Tanh
Loss => L1, Mean-Squared Error (MSE) or Multi-Class Cross Entropy (MCCE)
"""
class MLP:
	def __init__(self, NI, NH, NO, activation, loss):
		#Number of inputs, hidden units & outputs
		self.NI = NI
		self.NH = NH
		self.NO = NO
		self.activation = activation()
		self.output_activation = Tanh()
		self.loss = loss()

		#Weights 1st Layer (NI, NH)
		self.W1 = self._random_weights(-1, +1, (NI,NH))

		#Weights 2nd Layer (NH,NO)
		self.W2 = self._random_weights(-1, +1, (NH,NO))

		#Biases 1st Layer (,NH)
		self.B1 = np.zeros((1,NH))

		#Biases 2nd Layer (,NO)
		self.B2 = np.zeros((1,NO))

		#Weight *changes* to be applied to W1 & W2
		self.dW1 = np.empty(self.W1.shape)
		self.dW2 = np.empty(self.W2.shape)

		#Bias changed to be applied to B1 & B2
		self.dB1 = np.empty(self.B1.shape)
		self.dB2 = np.empty(self.B2.shape)

		#Activations 1st Layer (,NH)
		self.Z1 = np.empty((1,NH))

		#Activations 2nd Layer (,NO)
		self.Z1 = np.empty((1,NO))

		#Where values of hidden neurons are stored (,NH)
		self.H = np.empty((1,NH))

		#Where outputs are stored (,NO)
		self.O = np.empty((1,NO))

		self.error = None

		self.weights_info = []

	#Stochastic Gradient Descent
	def train_stochastic(self, X, y, num_epochs, lr=0.2):
		self.error_info = []
		num_samples = X.shape[0]
		for i in range(num_epochs):
			epoch_error = []
			for j in range(num_samples):
				#Get our single training sample for iteration
				sample_x = X[j]
				sample_y = y[j]

				#Reshape to fit model, necessary due to
				#single sample taken at a time
				sample_x = np.reshape(sample_x, (1,-1))

				#Complete one iteration of the model and store the error
				e = self.iteration(sample_x,sample_y,lr)
				epoch_error.append(e)

			#Compute, store & print the mean error from this epoch
			mean_error = np.array(epoch_error).mean()
			print("Epoch {} Error: {}".format(i, mean_error))
			self.error_info.append(mean_error)
		return self.error_info

	#Batch Gradient Descent
	def train(self, X, y, num_iterations, lr=0.2):
		self.error_info = []
		for i in range(num_iterations):
			e = self.iteration(X, y, lr)
			print("Iteration {} Error: {}".format(i,e))
			self.error_info.append(e)
		return self.error_info

	def iteration(self, X, y, lr):

		#Forward pass
		self.forward(X)

		#Calculate Error
		self.error = self.loss.forward(self.Z2, y)

		#--BACKWARD PASS--
		#Gradient: Error w.r.t activation output
		self.dout = self.loss.backward(self.Z2, y)

		#Gradient: Error w.r.t 2nd weights & biases
		self.dZ2 = self.output_activation.backward(self.dout, self.cache_Z2)

		#Gradient: Error w.r.t weights & biases
		self.dO2, self.dW2, self.dB2 = self._affine_backward(self.dZ2, self.cache_O)

		#Gradient: Error w.r.t 1st layer activation
		self.dZ1 = self.activation.backward(self.dO2, self.cache_Z1)

		#Gradient: Error w.r.t 1st weights & biases
		self.dO1, self.dW1, self.dB1 = self._affine_backward(self.dZ1, self.cache_H)

		#ISSUE: MEAN of dZ1 is too small (4.540476118476609e-42)
		#because dW2 is so high!
		self.W1 = self.step(self.W1, self.dW1, lr)

		self.W2 = self.step(self.W2, self.dW2, lr)

		self.B1 = self.step(self.B1, self.dB1, lr)
		self.B2 = self.step(self.B2, self.dB2, lr)

		self.weights_info.append(self.dW1.mean())
		#self.error_info.append(self.error)
		
		return self.error

	def step(self, w, dw, lr=0.01):
		return w - lr*dw

	#Don't apply thresholding
	def predict_no_threshold(self, X):
		return self.forward(X)

	#Apply threshold of 0.5
	def predict(self, X):
		feedforward = self.forward(X)
		if feedforward > 0.5:
			return 1
		else:
			return 0

	#Predict handwritten digits
	def predict_img(self, X, y, codes):
		feedforward = self.forward(X)

		results = []
		for sample in feedforward:
			index = np.where(sample == np.amax(sample))[0][0]
			results.append(codes[index])
		return results
		
	def forward(self, X):
		#--FORWARD PASS--
		#Input => First Layer
		self.H, self.cache_H = self._affine_forward(X, self.W1, self.B1)

		#Layer 1 => Activation 1
		self.Z1, self.cache_Z1 = self.activation.forward(self.H)

		#Activation 1 => Output Layer
		self.O, self.cache_O = self._affine_forward(self.Z1, self.W2, self.B2)

		#Output Layer => Output Activation
		self.Z2, self.cache_Z2 = self.output_activation.forward(self.O)

		return self.Z2

	def _affine_forward(self, x, w, b):
		out = None
		x_reshaped = np.reshape(x, (x.shape[0], -1))

		#Output of shape (N, M) - (4 Samples, 3 Hidden Units)
		out = x.dot(w) + b
		cache = (x, w, b)

		return out, cache

	def _affine_backward(self, dout, cache):
		x, w, b = cache
		dx, dw, db = None, None, None

		#Warning: Dot product can cause exploding gradients!
		dw = np.reshape(x, (x.shape[0],-1)).T.dot(dout)
		dw = np.reshape(dw, w.shape)
		db = np.sum(dout, axis=0, keepdims=False)

		dx = dout.dot(w.T)
		dx = np.reshape(dx, x.shape)
		return dx, dw, db

	#Initialise random weights in a uniform distribution
	def _random_weights(self, lo, hi, shape):
		return np.random.uniform(low=lo, high=hi, size=shape)

#Activation Function: Sigmoid
class Sigmoid:
	def forward(self, x):
		outputs = 1 / (1 + np.exp(-x))
		cache = outputs
		return outputs, cache

	def backward(self, dout, cache):
		dx = None
		dx = dout * cache * (1 - cache)
		return dx

#Activation Function: Tanh
class Tanh:
	def __init__(self):
		self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
		self.tanh = lambda x: 2 * self.sigmoid(2*x) - 1

	def forward(self,x):
		outputs = None
		cache = None
		outputs = self.tanh(x)
		cache = x
		return outputs, cache

	def backward(self, dout, cache):
		dx = None
		dx = (1 - np.power(self.tanh(cache),2))*dout
		return dx

#Loss Functions
class L1:
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

#Loss Function: Mean-Squared Error
class MSE:
	def forward(self, y_out, y_true):
		result = (y_out-y_true)**2
		result = result.mean()
		return result

	def backward(self, y_out, y_truth):
		gradient = 2 * (y_out - y_truth)
		return gradient

#Loss Function: Multi-Class Cross Entropy
class MCCE:
	def __init__(self):
		self.cache = {}

	def forward(self, y_out, y_true):
		N, C = y_out.shape
		y_true_one_hot = np.zeros_like(y_out)
		y_true_one_hot[np.arange(N), y_true] = 1

		#Transform logits into dist
		y_out_exp = np.exp(y_out - np.max(y_out, axis=1, keepdims=True))
		y_out_probs = y_out_exp / np.sum(y_out_exp, axis=1, keepdims=True)

		#Compute MCCE loss
		loss = -y_true_one_hot * np.log(y_out_probs)
		loss = loss.sum(axis=1).mean()
		self.cache['probs'] = y_out_probs
		return loss

	def backward(self, y_out, y_true):
		N, C = y_out.shape
		gradient = self.cache['probs']
		gradient[np.arange(N), y_true] -= 1
		gradient /= N
		return gradient


if __name__ == "__main__":
	#Question 1 & 2
	experiment_XOR_function()

	#Question 3 & 4
	experiment_sine_function()

	#Special Experiment
	experiment_handwritten_letters()