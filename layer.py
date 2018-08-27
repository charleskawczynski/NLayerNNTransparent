import numpy as np

def sigmoid(x):
  return 1.0/(1+np.exp(-x))

def sigmoidGradient(z):
  sg = sigmoid(z)
  return sg*(1.0-sg)

def hypothesis(x):
  return 1.0/(1+np.exp(-x))

def uniform_dist(S):
  epsilon_init = 0.1
  temp = np.random.rand(S[0], S[1])*epsilon_init
  return temp - np.mean(temp)

class layer():
  weights = []
  activation = []

  def __init__(self,
    input_size,
    output_size,
    activation = lambda x: sigmoid(x),
    weight_dist = lambda x: uniform_dist(x),
    add_bias = True
    ):
    self.activation = activation
    self.activation_grad = lambda x : sigmoidGradient(x)
    self.weight_dist = weight_dist
    self.add_bias = add_bias
    self.input_size = input_size
    self.output_size = output_size
    self.weights_shape = (self.output_size, (self.input_size + 1))
    # Order is important:
    self.init_weights()
    self.define_flat_weights_shape()
    self.roll_weights()

  def set_weights_slice(self, weights_slice):
    self.weights_slice = weights_slice

  def init_weights(self):
    self.weights = self.weight_dist((self.output_size, self.input_size+1)) # +1 for bias

  def define_flat_weights_shape(self):
    self.weights_flat_shape = (np.prod(self.weights.shape), 1)

  def roll_weights(self):
    self.weights_flat = np.squeeze(np.reshape(self.weights, self.weights_flat_shape))

  def unroll_weights(self, weights_flat):
    self.weights = np.reshape(weights_flat[self.weights_slice], self.weights_shape)

  def activation_layer(self, input_layer):
    self.a = input_layer
    if self.add_bias: self.a = np.vstack((np.ones((1, self.a.shape[1])), self.a))
    self.z = np.matmul(self.weights, self.a)
    self.h = self.activation(self.z)
    return self.h, self.z, self.a

  def __str__(self):
    s ='------------------------\n'
    s+='input_size     = {}\n'.format(self.input_size)
    s+='output_size    = {}\n'.format(self.output_size)
    s+='weights_shape  = {}\n'.format(self.weights_shape)
    s+='weights_slice  = {}\n'.format(self.weights_slice)
    return s
