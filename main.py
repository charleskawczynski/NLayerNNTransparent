import os
import numpy as np
import pandas as pd
clear = lambda: os.system('cls')
clear()
import sys
import layer as L
import neuralNetwork as NN

def get_data():
  data_dir = '..'+os.sep+'data'+os.sep+'StanfordClass'+os.sep
  X = pd.read_csv(data_dir+'X_data.csv', header=None)
  Y = pd.read_csv(data_dir+'y_data.csv', header=None)
  X = np.array(X.values)
  Y = np.array(Y.values)
  Y = [t[0] for t in Y]
  m = X.shape[0]
  return X, Y, m

def main():
  X, Y, m = get_data()
  m = X.shape[0]
  layer_sizes       = [400, 25, 10]
  N = len(layer_sizes)
  print('Initializing Neural Network Parameters')
  layers = [L.layer(layer_sizes[i], layer_sizes[i+1]) for i in range(0, N-1)]
  nn = NN.neuralNetwork(layers, regularization = 1, batch_size = m)
  print(nn)
  print('Training Neural Network ')
  sys.stdout.flush()
  nn = NN.gradient_decent_adam_new(nn, X, Y)
  pred = nn.predict(X)
  array_correct = [1 if x==p else 0 for x,p in zip(pred, Y)]
  array_incorrect = [1 if x!=p else 0 for x,p in zip(pred, Y)]
  print('Training Set Correct:   {}'.format(np.sum(array_correct)))
  print('Training Set Incorrect: {}'.format(np.sum(array_incorrect)))
  print('Training Set Accuracy:  {}'.format(np.mean(array_correct) * 100))

main()
print('Done')
