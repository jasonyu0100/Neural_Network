import random
import numpy as np
random.seed(1)

def relu(x,derivative=False):
  if derivative == False:
    return max(x,0)
  else:
    if x >= 0: return 1
    else: return 0

def linear(x,derivative=False):
  if derivative == False: return x
  else: return 1

def sigmoid(x,derivative=False):
  if derivative == False: return 1 / float(1 + np.exp(-x)) 
  else: return x * (1-x)

class Network:
  def __init__(self,layers,output_mapping=False,network_type='classification'):
    self.layers = layers
    self.network_size = len(self.layers)
    self.input_size = layers[0].input_size
    self.output_size = layers[-1].size
    self.output_mapping = output_mapping
    self.network_type = network_type
    self.init_layers()

  def init_layers(self):
    for layer_index,current_layer in enumerate(self.layers):
      if layer_index == 0: previous_layer_size = self.input_size
      else: previous_layer_size = self.layers[layer_index-1].size
      current_layer.init_neurons(previous_layer_size)

  def forward_propogation(self,input_features):
    layer_output = input_features
    for layer_index in range(self.network_size):
      layer_output = self.layers[layer_index].layer_output(layer_output)
    return layer_output
  
  def predict(self,final_output_layer):
    prediction_index = max(range(len(final_output_layer)),key=lambda index:final_output_layer[index])
    return prediction_index
  
  def predict_verbose(self,final_output_layer):
    return self.output_mapping[self.predict(final_output_layer)]

  def cost_function(self,output,actual,derivative=False):
    if derivative == False: return (output - actual) ** 2
    else: return 2 * (output - actual)

  def cost_derivative(self,layer_index,current_neuron_index):
    next_layer = self.layers[layer_index + 1]
    cost_derivative = 0
    for neuron in next_layer.neurons:
      output_neuron_cost_derivative = neuron.cost_derivative * neuron.activation_derivative * neuron.weights[current_neuron_index]
      cost_derivative += output_neuron_cost_derivative
    return cost_derivative

  def total_cost(self,final_output_layer,label):
    total_cost = 0
    for output_index,output_value in enumerate(final_output_layer):
      if label == output_index: total_cost += self.cost_function(output_value,1)
      else: total_cost += self.cost_function(output_value,0)
    return total_cost

  def back_propogation(self,label,input_features,final_output_layer):
    total_cost = self.total_cost(final_output_layer,label)
    for layer_index,layer in reversed(list(enumerate(self.layers))):
      for neuron_index,neuron in enumerate(layer.neurons):
        for weight_index,weight in enumerate(neuron.weights):
          if layer_index == (self.network_size - 1):
            neuron.cost_derivative = self.cost_function(neuron.neuron_output,label[neuron_index],derivative=True)
          else:
            neuron.cost_derivative = self.cost_derivative(layer_index,neuron_index)
          neuron.activation_derivative = neuron.activation_function(neuron.neuron_input,derivative=True)
          neuron.weight_derivative = weight
          derivative = neuron.cost_derivative * neuron.activation_derivative * neuron.weight_derivative
          delta = -derivative
          neuron.stored_weight_deltas[weight_index] += delta
    return total_cost

  def train_network(self,training_set,valid_set,learning_rate,epochs,batch_size):
    counter = 0
    training_set_length = len(training_set)
    for epoch_count in range(epochs):
      total_cost = 0
      total_correct = 0
      for label,input_features in training_set:
        final_output_layer = self.forward_propogation(input_features)
        prediction = self.predict(final_output_layer)
        actual_label = self.predict(label)
        if prediction == actual_label: total_correct += 1
        total_cost += self.back_propogation(label,input_features,final_output_layer)
        counter += 1
        if counter % batch_size == 0:
          counter = 0
          self.update_network(learning_rate)
      self.update_network(learning_rate)
      accuracy = float(total_correct) / float(training_set_length)
      print("Epoch:{} Cost: {} Accuracy: {}".format(epoch_count,total_cost/len(training_set),accuracy))
      
  def evaluate_network(self,testing_set):
    if self.network_type == 'classification':
      total_correct = 0
      confusion_matrix = [[0 for j in range(self.output_size)] for i in range(self.output_size)]
      for label,input_features in testing_set:
        final_output_layer = self.forward_propogation(input_features)
        prediction = self.predict(final_output_layer)
        actual_label = self.predict(label)
        if prediction == actual_label: total_correct += 1
        confusion_matrix[actual_label][prediction] += 1
      accuracy = float(total_correct) / float(len(testing_set))
      print("Accuracy: {}%".format(100*accuracy))
      print("Confusion Matrix: ")
      for line in confusion_matrix:
        print(line)
    else:
      total_cost = 0
      for label,input_features in testing_set:
        final_output_layer = self.forward_propogation(input_features)
        total_cost += self.total_cost(final_output_layer,label)
      average_cost = total_cost / len(testing_set)
      print("Average cost: {}".format(average_cost))

  def show_weights(self):
    for layer_index,layer in enumerate(self.layers):
      weights = [[round(weight,2) for weight in neuron.weights] for neuron in layer.neurons]
      bias = [round(neuron.bias,2) for neuron in layer.neurons]
      print("Layer {}: Weights={} Bias={}".format(layer_index,weights,bias))

  def update_network(self,learning_rate):
    for layer in self.layers:
      layer.update_layer(learning_rate)

  @staticmethod
  def encode_one_hot_vectors(non_encoded_labels,length):
    encoded_vectors = []
    for index in non_encoded_labels:
      vector = [0 for i in range(length)]
      vector[index] = 1
      encoded_vectors.append(vector)
    return encoded_vectors


class Layer:
  def __init__(self,size,activation_function,input_size=False,neuron_weights=False,neuron_biases=False):
    self.size = size
    self.activation_function = activation_function
    self.input_size = input_size
    self.neuron_weights = neuron_weights
    self.neuron_biases = neuron_biases

  def init_neurons(self,previous_layer_size):
    if self.neuron_weights == False: self.neuron_weights = [[self.random() for weight_index in range(previous_layer_size)] for neuron_index in range(self.size)]
    if self.neuron_biases  == False: self.neuron_biases  = [0 for neuron_index in range(self.size)]
    self.neurons = [Neuron(self.neuron_weights[neuron_index],self.neuron_biases[neuron_index],self.activation_function) for neuron_index in range(self.size)]

  def layer_output(self,inputs):
    layer_output = []
    for neuron in self.neurons:
      neuron_input = neuron.neuron_input_propogation(inputs)
      neuron.neuron_input = neuron_input
      neuron_output = neuron.neuron_output_activation(neuron_input)
      neuron.neuron_output = neuron_output
      layer_output.append(neuron_output)
    return layer_output
  
  def update_layer(self,learning_rate):
    for neuron in self.neurons:
      neuron.update_neuron(learning_rate)

  @staticmethod    
  def random():
    value = 1 - random.random() * 2
    return value

class Neuron:
  def __init__(self,weights,bias,activation_function):
    self.weights = weights
    self.bias = bias
    self.activation_function = activation_function
    self.neuron_input = None
    self.neuron_output = None
    self.cost_derivative = None
    self.activation_derivative = None
    self.weight_derivative = None
    self.stored_weight_deltas = [0 for weight_index in range(len(self.weights))]
    self.stored_bias_delta = 0

  def neuron_input_propogation(self,inputs):
    weight_values = [(self.weights[weight_index] * input_val) for weight_index,input_val in enumerate(inputs)]
    summed_weight_values = sum(weight_values) + self.bias
    return summed_weight_values
    
  def neuron_output_activation(self,input_val):
    return self.activation_function(input_val)

  def update_neuron(self, learning_rate):
    for weight_index,weight_delta in enumerate(self.stored_weight_deltas):
      self.weights[weight_index] += weight_delta * learning_rate
    self.bias += self.stored_bias_delta * learning_rate
    self.stored_weight_deltas = [0 for weight_index in range(len(self.weights))]
    self.stored_bias_delta = 0

def main():
  nn = Network([Layer(2,activation_function=relu,input_size=1),
                Layer(2,activation_function=linear)],
                output_mapping={0:'cat',1:'dog'},
                network_type='classification')

  labels = [1,1,0,0]
  one_hot_encoded_labels  = nn.encode_one_hot_vectors(labels,2)
  features = [[10],[20],[-30],[-10]]
  training_set = list(zip(one_hot_encoded_labels,features))
  valid_set = list(zip(one_hot_encoded_labels,features))
  testing_set = list(zip(one_hot_encoded_labels,features))
  nn.evaluate_network(testing_set)
  nn.train_network(training_set,valid_set,learning_rate=0.01,epochs=100,batch_size=1)
  nn.evaluate_network(testing_set)

if __name__ == '__main__': main()

'''
Regression
Assigning quantities to data
ie the price of a house in sydney

Classification
assigning identites to data
ie whether someone is female or not

TODO LIST
Fix weight initilisation.
Implement Softmax functionality
'''