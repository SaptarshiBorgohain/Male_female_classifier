import numpy as np

def sigmoid(x):
  #sigmoid function
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  # Sigmoid derivative.  
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  #   
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  
  def __init__(self):
    # Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()
    
    
  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    
    learn_rate = 0.1
    epochs = 1000 # number of times to loop through the entire dataset

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Do a feedforward (we'll need these values later)
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- Calculate partial derivatives.
        # --- Naming: p_L_p_w1 stands for "partial L partial w1"
        
        p_L_p_ypred = -2 * (y_true - y_pred)

        # Neuron o1
        p_ypred_p_w5 = h1 * deriv_sigmoid(sum_o1)
        p_ypred_p_w6 = h2 * deriv_sigmoid(sum_o1)
        p_ypred_p_b3 = deriv_sigmoid(sum_o1)

        p_ypred_p_h1 = self.w5 * deriv_sigmoid(sum_o1)
        p_ypred_p_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Neuron h1
        p_h1_p_w1 = x[0] * deriv_sigmoid(sum_h1)
        p_h1_p_w2 = x[1] * deriv_sigmoid(sum_h1)
        p_h1_p_b1 = deriv_sigmoid(sum_h1)

        # Neuron h2
        p_h2_p_w3 = x[0] * deriv_sigmoid(sum_h2)
        p_h2_p_w4 = x[1] * deriv_sigmoid(sum_h2)
        p_h2_p_b2 = deriv_sigmoid(sum_h2)

        #Updating weights and biases
        # Neuron h1
        
        self.w1 -= learn_rate * p_L_p_ypred * p_ypred_p_h1 * p_h1_p_w1
        self.w2 -= learn_rate * p_L_p_ypred * p_ypred_p_h1 * p_h1_p_w2
        self.b1 -= learn_rate * p_L_p_ypred * p_ypred_p_h1 * p_h1_p_b1

        # Neuron h2
        
        self.w3 -= learn_rate * p_L_p_ypred * p_ypred_p_h2 * p_h2_p_w3
        self.w4 -= learn_rate * p_L_p_ypred * p_ypred_p_h2 * p_h2_p_w4
        self.b2 -= learn_rate * p_L_p_ypred * p_ypred_p_h2 * p_h2_p_b2

        # Neuron o1
        
        self.w5 -= learn_rate * p_L_p_ypred * p_ypred_p_w5
        self.w6 -= learn_rate * p_L_p_ypred * p_ypred_p_w6
        self.w6 -= learn_rate * p_L_p_ypred * p_ypred_p_w6

      #Calculating total loss at the end of each epoch
    
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))

# Defining dataset
#The weights are in lbs. And the given value is 135-(total weight).This is to make things easier.
#The heights are in inches.And the given value is 66-(total height).To make things easier.
data = np.array([
  [-2, -1],  # Alice[weight, height]
  [25, 6],   # Bob[weight, height]
  [17, 4],   # Charlie[weight, height]
  [-15, -6], # Diana[weight, height]
])
#For girls the value is set to 1 and 0 for boys.
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Training neural network!!!!!!!!!!
network = OurNeuralNetwork()
network.train(data, all_y_trues)

emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily))
print("Frank: %.3f" % network.feedforward(frank))
#the values will be printed close to 0 and 1 with 3 decimal points.So if its close to 1 its a girl and close to 0 will be a boy.