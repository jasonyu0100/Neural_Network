import random
import numpy as np
from math import exp
random.seed(1)

def relu(x,derivative=False):
  if derivative == False:
    return max(x,0)
  else:
    if x >= 0: return 1
    else: return 0

def leaky_relu(x,derivative=False):
  if derivative == False:
    if x >= 0: return x
    else: return 0.1 * x
  else:
    if x >= 0: return 1
    else: return 0.1

def linear(x,derivative=False):
  if derivative == False: return x
  else: return 1

def sigmoid(x,derivative=False):
  if derivative == False: 
    return 1 / (1 + exp(-x))
  else: return x * (1-x)

def softmax(x,derivative=False):
  pass

class Network:
  def __init__(self,layers,network_type='classification'):
    self.layers = layers
    self.network_size = len(self.layers)
    self.input_size = layers[0].input_size
    self.output_size = layers[-1].size
    self.network_type = network_type
    self.init_layers()

  def init_layers(self):
    for layer_index,current_layer in enumerate(self.layers):
      if layer_index == 0: previous_layer_size = self.input_size
      else: previous_layer_size = self.layers[layer_index-1].size
      current_layer.init_neurons(previous_layer_size)

  def forward_propogation(self,input_features):
    assert(len(input_features) == self.input_size)
    layer_output = input_features
    layer_outputs = [layer_output]
    for layer_index in range(self.network_size):
      layer_output = self.layers[layer_index].layer_output(layer_output)
      layer_outputs.append(layer_output)
    return layer_outputs
  
  def predict(self,final_output_layer):
    prediction_index = max(range(len(final_output_layer)),key=lambda index:final_output_layer[index])
    return prediction_index
  
  def predict_verbose(self,final_output_layer,output_mapping):
    return output_mapping[self.predict(final_output_layer)]

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
      total_cost += self.cost_function(output_value,label[output_index])
    return total_cost

  def back_propogation(self,label):
    for layer_index,layer in reversed(list(enumerate(self.layers))):
      for neuron_index,neuron in enumerate(layer.neurons):
        for weight_index,weight in enumerate(neuron.weights):
          if layer_index == (self.network_size - 1):
            neuron.cost_derivative = self.cost_function(neuron.neuron_output,label[neuron_index],derivative=True)
          else:
            neuron.cost_derivative = self.cost_derivative(layer_index,neuron_index)
          neuron.activation_derivative = neuron.activation_function(neuron.neuron_input,derivative=True)
          neuron.weight_derivative = weight
          cost_weight_delta = -1 * neuron.cost_derivative * neuron.activation_derivative * neuron.weight_derivative
          neuron.stored_weight_deltas[weight_index] += cost_weight_delta

  def train_network(self,training_set,valid_set,learning_rate,epochs,batch_size,display_epoch):
    for epoch_count in range(epochs):
      training_cost,training_correct = 0.0,0.0
      for counter,data in enumerate(training_set):
        label,input_features = data
        layer_activations = self.forward_propogation(input_features)
        final_output_layer = layer_activations[-1]
        prediction = self.predict(final_output_layer)
        actual_label = self.predict(label)
        if prediction == actual_label: training_correct += 1
        current_cost = self.total_cost(final_output_layer,label)
        training_cost += current_cost
        self.back_propogation(label)
        if counter % batch_size == 0:
          self.update_network(learning_rate)
      self.update_network(learning_rate)

      valid_cost,valid_correct = self.data_set_passover(valid_set)
      if epoch_count % display_epoch == 0:
        training_data = self.get_network_data(training_cost,training_correct,len(training_set))
        valid_data = self.get_network_data(valid_cost,valid_correct,len(valid_set))
        print('Epoch: {} Training Data: {} Valid Data: {}'.format(epoch_count,training_data,valid_data))

  def data_set_passover(self,data_set):
    cost,correct = 0.0,0.0
    for counter,data in enumerate(data_set):
      label,input_features = data
      final_output_layer = self.forward_propogation(input_features)[-1]
      prediction = self.predict(final_output_layer)
      actual_label = self.predict(label)
      if prediction == actual_label: correct += 1
      current_cost = self.total_cost(final_output_layer,label)
      cost += current_cost
    return cost,correct

  def get_network_data(self,total_cost,total_correct,data_set_length):
    accuracy = round(float(total_correct) / float(data_set_length),3)
    return ("Total Cost: {} Accuracy: {}".format(total_cost,accuracy))
      
  def evaluate_network(self,testing_set):
    if self.network_type == 'classification':
      total_cost,total_correct = self.data_set_passover(testing_set)
      print('Network Information: {}'.format(self.get_network_data(total_cost,total_correct,len(testing_set))))
      self.show_confusion_matrix(testing_set)
    else:
      total_cost = 0
      for label,input_features in testing_set:
        final_output_layer = self.forward_propogation(input_features)[-1]
        total_cost += self.total_cost(final_output_layer,label)
      average_cost = total_cost / len(testing_set)
      print("Average cost: {}".format(average_cost))

  def show_confusion_matrix(self,data_set):
    confusion_matrix = [[0 for j in range(self.output_size)] for i in range(self.output_size)]
    for label,input_features in data_set:
      final_output_layer = self.forward_propogation(input_features)[-1]
      prediction = self.predict(final_output_layer)
      actual_label = self.predict(label)
      confusion_matrix[actual_label][prediction] += 1
    print('Confusion Matrix: ')
    for line in confusion_matrix: print(line)

  def show_weights(self):
    for layer_index,layer in enumerate(self.layers):
      weights = [[round(weight,3) for weight in neuron.weights] for neuron in layer.neurons]
      bias = [round(neuron.bias,3) for neuron in layer.neurons]
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
    if self.neuron_weights == False: self.neuron_weights = [self.xavier_weight_initialization(previous_layer_size) for neuron_index in range(self.size)]
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

  def xavier_weight_initialization(self,previous_layer_size):
    x = 2 if self.activation_function == relu else 1 #Relu initialization has sd of 2/n while other of 1/n
    weights = [random.gauss(0,float(x)/float(previous_layer_size)) for i in range(previous_layer_size)]
    return weights

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
    self.reset_deltas()

  def reset_deltas(self):
    self.stored_weight_deltas = [0 for weight_index in range(len(self.weights))]
    self.stored_bias_delta = 0

def main():
  nn = Network([Layer(2,activation_function=relu,input_size=1),
                Layer(2,activation_function=linear)],
                network_type='classification')

  labels = [1,1,0,0]
  one_hot_encoded_labels = nn.encode_one_hot_vectors(labels,2)
  features = [[1],[2],[-1],[-2]]
  training_set = list(zip(one_hot_encoded_labels,features))
  valid_set = training_set
  nn.train_network(training_set,valid_set,learning_rate=0.01,epochs=1000,batch_size=1,display_epoch=10)
  # nn.forward_propogation([1])
  # nn.back_propogation([0,1])

main()