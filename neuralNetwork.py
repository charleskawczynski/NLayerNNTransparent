import sys
import layer as lay
import numpy as np
class neuralNetwork():

  def __init__(self, layers, regularization, batch_size):
    self.layers = layers
    self.regularization = regularization
    self.batch_size = batch_size
    self.coeff = 1/batch_size
    self.coeff_reg_grad = self.coeff*self.regularization
    self.coeff_reg = self.coeff*self.regularization*0.5
    self.N = len(self.layers)
    self.set_weights_slices()
    self.roll_weights()

  def set_weights_slices(self):
    i_stop = 0
    for i, L in enumerate(self.layers):
      i_start = i_stop
      i_stop += L.output_size * (L.input_size+1)
      weights_slice = range(i_start, i_stop)
      self.layers[i].set_weights_slice(weights_slice)

  def roll_weights(self):
    for i, L in enumerate(self.layers):
      self.layers[i].roll_weights()
    weights_flat = [self.layers[i].weights_flat for i, L in enumerate(self.layers)]
    self.weights_flat = np.hstack((x for x in weights_flat))

  def unroll_weights(self, weights_flat):
    for i, L in enumerate(self.layers):
      self.layers[i].unroll_weights(weights_flat)

  def hypothesis(self, input_layer):
    import copy
    h = copy.deepcopy(np.transpose(input_layer))
    H, Z, A = [h],[],[]
    for i, L in enumerate(self.layers):
      h, z, a = self.layers[i].activation_layer(h)
      H+=[copy.deepcopy(h)]
      Z+=[copy.deepcopy(z)]
      A+=[copy.deepcopy(a)]
    return H, Z, A

  def costFunction(self, X, Y):

    for i, L in enumerate(self.layers):
      self.layers[i].unroll_weights(self.weights_flat)

    H, Z, A = self.hypothesis(X)
    h = H[-1] # prediction is at last layer

    K = self.layers[-1].output_size
    Y_mat = np.zeros((K, self.batch_size))
    Y_squeeze = np.squeeze(Y)
    for k in range(0, K):
      Y_mat[k,:] = Y_squeeze==k+1
    S = -np.multiply(Y_mat, np.log(h)) - np.multiply((1-Y_mat), np.log(1-h))
    J = self.coeff*np.sum(S)

    ## Regularization for cost
    S_reg = 0
    for i, L in enumerate(self.layers):
      S_reg += np.sum(np.sum(self.layers[i].weights[:,1:]**2.0))
    J_reg = self.coeff_reg*S_reg
    J += J_reg

    # Back propagation
    self.layers[self.N-1].delta = h - Y_mat
    for i in reversed(range(self.N-1)):
      temp = np.matmul(np.transpose(self.layers[i+1].weights), self.layers[i+1].delta)
      sg = self.layers[i].activation_grad(Z[i])
      self.layers[i].delta = temp[1:,:]*sg

    # Need a matrix per training example:
    for i, L in enumerate(self.layers):
      self.layers[i].d_J_d_Theta = np.zeros(L.weights.shape)

    for t in range(0, self.batch_size):
      for i, L in enumerate(self.layers):
        self.layers[i].d_J_d_Theta += np.outer(self.layers[i].delta[:,t], np.transpose(A[i][:,t]))

    for i, L in enumerate(self.layers):
      self.layers[i].d_J_d_Theta *= self.coeff

    ## Regularization for grad
    for i, L in enumerate(self.layers):
      self.layers[i].d_J_d_Theta[:,1:] += self.coeff_reg_grad*self.layers[i].weights[:,1:]

    # Unroll gradients
    d_J_d_Theta_flat = [L.d_J_d_Theta.flatten() for i, L in enumerate(self.layers)]
    d_J_d_Theta_flat = np.hstack((x for x in d_J_d_Theta_flat))
    self.J = J
    self.grad_J = d_J_d_Theta_flat

  def predict(self, X):
    H, Z, A = self.hypothesis(X)
    p = np.argmax(H[-1], axis=0)
    p = [x+1 for x in p] # since indexing from 0, specific to MNIST
    return p

  def __str__(self,):
    s = ''.join([str(L) for i, L in enumerate(self.layers)])
    s+='------------------------\n'
    return s

def visualize_search(sol_history):
  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot(range(0,len(sol_history)), sol_history, 'b-o')
  plt.xlabel('Iteration')
  plt.ylabel('Error')
  plt.title('Error per iteration')
  plt.show()

def gradient_decent_adam_new(nn, X, Y):
  i, sol_history = 0, []
  nn.costFunction(X, Y)
  max_iter, tol = 200, 1e-6
  M, V, t = 0, 0, 0.0
  learning_rate, beta_1, beta_2, eps = 0.5, 0.9, 0.999, 1e-8
  L = logical_linspace(max_iter, N_true = 10)
  while nn.J > tol:
    t += 1.0
    G = nn.grad_J
    M = beta_1*M + (1-beta_1)*G
    V = beta_2*V + (1-beta_2)*G**2.0
    Mhat = M/(1-beta_1**t)
    Vhat = V/(1-beta_2**t)
    nn.weights_flat -= learning_rate*Mhat/(np.sqrt(Vhat)+eps)

    sol_history += [nn.J]
    nn.costFunction(X, Y)
    i += 1
    if i > max_iter:
      print('Solution not converged \nJ = {}'.format(nn.J))
      break
    if L[i]:
      print('Iter = {} | Loss = {}'.format(i, nn.J))
      sys.stdout.flush()
  sol_history += [nn.J]
  visualize_search(sol_history)
  return nn

def logical_linspace(max_iter, N_true=1000):
  """Returns a logical array, with max_iter
  elements, of N_true evenly spaced (roughly)
  true values.
  """
  arr = np.linspace(0, max_iter, N_true).astype(int)
  i_arr = range(0, max_iter+1)
  LL = []
  j = 0
  for i in i_arr:
    if i_arr[i] == arr[j]:
      LL += [True]
      j += 1
    else:
      LL += [False]
  check = LL.count(True)
  if check != N_true:
    print(LL)
    print(check)
    raise ValueError('Incorrect True count in logical_linspace')
  print('len(LL) = {}'.format(len(LL)))
  return np.array(LL)
