# The code below is written from the following youtube tutorial
# https://www.youtube.com/playlist?list=PLRyu4ecIE9tibdzuhJr94uQeKnOFkkbq6


# References for the variables used

# shape/ layerSize is architecture of the neural network. ex: (4,3,2,1)  means 4 units at input, 3 units at hidden layer 1, 2 units at hidden layer 2, 1 unit at output

# layerCount is number of layers in the network excluding input layer (as it is just the buffer to hold input)
# for shape = (4,3,2,1), layerCount is 3.

# _layerInput is the List containing preactivation of each layer. (excluding input layer and including the output layer)
# _layerInput[0] contains the preactivation of the hidden layer 1
# _layerInput[layerCount - 1] contains the preactivation of the output layer
# each _layerInput[i] is a matrix, and
# j'th column of _layerInput[i] contains preactivation of (i+1)'th hidden layer for j'th training example. 
 
# _layerOutput is the List containing activation of each layer. (excluding input layer and including the output layer)
# _layerOutput[0] contains the activation of the hidden layer 1
# _layerOutput[layerCount - 1] contains the activation of the output layer i.e. this contains the output

# outputDelta difference at the output layer for all training examples. j'th column contains the cordinate wise difference at the output for j'th training example
# outputDelta is also the gradient at output if we take the least square error function

# error sum of the squared error at the output layer for all the training examples. This is a number.

# delta is the List of gradient at preactivation starting from preactivation at output to preactivation at the 1st hidden layer
# delta[0] is the matrix of gradient at preactivation of the output layer
# delta[layerCount-1] is the matrix of gradient at preactivation of 1'st hidden layer
# delta[i] is the matrix of gradient at preactivation of (layerCount-i)'th hidden layer
# j'th column of delta[i] contains the gradient at preactivation of (layerCount-i)'th hidden layer for j'th training example.

# deltaPullback is the matrix of the gradient at the activation of a layer 
# Note: for gradient at the activation of i'th hidden layer, we need Wi matrix and the gradient of preactivation at (i+1) hidden layer
# deltaPullback is the matrix where j'th column shows the gradient of the activation for j'th training example.

import numpy as np

class backPropagationNetwork:
	layerCount = 0
	shape = None
	weights = []

	# constructor function for the class
	def __init__(self, layerSize):
		# layerSize is the Architecture of the NN as (4,3,1)
		self.layerCount = len(layerSize)-1 # input layers is just a buffer
		self.shape = layerSize

		# for the forward pass maybe.
		self._layerInput = []
		self._layerOutput = []
		self._previousWeightDelta = []

		# for a (4,3,2) layers the weight matrix will be of size
		# 3X5 (w1), 2X4 (w2)
		# zip just makes the list of tupples cordinate wise
		for(l1,l2) in zip(layerSize[:-1],layerSize[1:]):
			self.weights.append(np.random.normal(scale=0.1,size=(l2,l1+1)))

	# for sigmoid units return the function value at x or the value of the derivative at x
	# dependeing upon derivative flag
	def sgm(self,x,derivative=False):
		if not derivative:
			return 1/(1+np.exp(-x))
		else:
			out = self.sgm(x)
			return out*(1-out)


	# forward run/pass
	def run(self,X):
		
		# no of training examples
		m = X.shape[0]

		# initialize/ clear out the input and output list from previous run
		self._layerInput = []
		self._layerOutput = []

		# Forward pass
		for i in range(self.layerCount):
			if i == 0:
				layerInput = self.weights[0].dot(np.vstack([X.T, np.ones([1,m])]))            # vstack(a,b) stacks matrix/vector b below matrix/vector a
			else:
				layerInput = self.weights[i].dot(np.vstack([self._layerOutput[-1], np.ones([1,m])]))
			
			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sgm(layerInput))

		return self._layerOutput[-1].T


	def trainEpoch(self,X,Y,trainingRate = 0.2):
		# trains the network for one epoch
		delta = []
		m = X.shape[0]

		# forward pass before we can compute the gradient by back propagation
		self.run(X)

		# Computing the deltas for the preactivation at each layer including the output
		# for the sqaure error and the sigmoid activation units
		# delta(for preactivation) = delta(for activation)*(sigm(preactivation)(1-sigm(preactivation)))
		for i in reversed(range(self.layerCount)):                                             # reverse as the backpropogation work in reverse order
			
			if i == self.layerCount-1:                                                         # if this is for the preactivation at the output								
				outputDelta = self._layerOutput[i] - Y.T                                      # this is also the gradient at output if we take the least square error function
				error = np.sum(outputDelta**2)                                                # sum of all the elements along all dimensions
				delta.append(outputDelta*self.sgm(self._layerInput[i],True))                  # '*' operator is for coordinate wise multiplication
			else:
				deltaPullback = self.weights[i+1].T.dot(delta[-1]) 							  # this is the gradient at the activation of the hidden layer (i+1), note that i = 0
																							  # is for hidden layer 1.
				delta.append(deltaPullback[:-1,:]*self.sgm(self._layerInput[i],True))         # this is the gradient at the preactivation at hidden layer (i+1)
				# deltaPullback is the matrix where j'th column shows the gradient of the activation for j'th training example.
				# Now the last row of the deltaPullback matrix is for the gradients of the activation for bias units we add to each hidden layer
				# We know to compute the gradient at the preactivation from the gradient at the activation, we need preactivation first
				# and for bias units we dun have preactivation (i.e we dun have value of preactivation of bias units in self._layerInput[i])
				# We should remove this row as while calculating the delta of the layer (gradient at preactivation)

		# compute the weight delta (gradient)
		# If a weight matirx connects layer i to layer i+1, for computing gradient w.r.t weights we need activation at layer i and preactivation at layer i+1.
		# here delta will have preactivation and layerOutput will have the activation for the required layers
		for i in range(self.layerCount):
			deltaIndex = self.layerCount - 1 - i 												# delta[0] is preactivation at output and so on in backward direction
			if i == 0:
				layerOutput = np.vstack([X.T,np.ones([1,m])])									# for W0 the delta (preactivation) is input layers
			else:
				layerOutput = np.vstack([self._layerOutput[i-1],np.ones([1,self._layerOutput[i-1].shape[1]])])		# _layerOutput[0] contains the activation of the hidden layer 1 and so for Wi we need _layerOutput[i-1] 

			weightDelta = np.sum(layerOutput[None,:,:].transpose(2,0,1)*delta[deltaIndex][None,:,:].transpose(2,1,0),axis=0)
			# layerOutput[None,:,:].transpose(2,0,1) -> each column Cj in layerOutput (which is an output/activation matrix for some layer) will be transposed in a different layers where each
			# layer has this column Cj as one row.
			# Check (1 X m X n).transpose(2,0,1) = (n X 1 X m)

			# delta[deltaIndex].transpose(2,1,0) -> each column Cj in delta[deltaIndex] (which is an input/preactivation matrix for some layer) will be transposed in a different layers where each
			# layer has this column Cj as one column.
			# Check (1 X m X n).transpose(2,1,0) = (n X m X 1)

			# weightDelta will have the gradients for the weight matrix for the current iteration. Each layer will have the gradient weight matrix for a training example for current layer
			trainingRate = 0.5
			self.weights[i] -= trainingRate * weightDelta
		return error # incase useful



# if this module is run as a script, run the if and create a object of class defined above
if __name__ == "__main__":
	bpn = backPropagationNetwork((2,2,1)) # calling __init__ constructor for the bpn object
	print(bpn.shape)
	print(bpn.weights)

	# X = np.array([[0,0],[1,0],[0.5,3]])
	# Y = bpn.run(X)

	# print(X)
	# print(Y)

	X = np.array([[0,0],[1,1],[0,1],[1,0]])
	Y = np.array([[0.05],[0.05],[0.95],[0.95]])

	maxIteration = 100000
	minError = 1e-5
	for i in range(maxIteration+1):
		err = bpn.trainEpoch(X,Y)
		if i%2500 == 0:
			print "iteration {0}\t error: {1:0.6f}".format(i, err)
		if err <= minError:
			print "Minimum error reached as iteration {0}".format(i)
			break

	Ycomputed = bpn.run(X)
	print "Input: {0}\n Output: {1}".format(X,Ycomputed)
